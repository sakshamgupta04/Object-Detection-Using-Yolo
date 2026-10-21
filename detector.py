"""
================================================================================
 YOLOv3 Object Detection Engine
 Author: Saksham Gupta (https://github.com/sakshamgupta04)
 Repository: https://github.com/sakshamgupta04/Object-Detection-Using-Yolo
================================================================================
"""

import os
import sys
import time
from typing import List, Dict, Tuple, Optional
import cv2
import numpy as np


class YOLODetector:
    """
    High-performance YOLOv3 detector with automated scaling,
    hardware acceleration fallback, and modern visual styling.
    """

    def __init__(
        self,
        yolo_dir: str = "yolo-coco",
        confidence: float = 0.5,
        threshold: float = 0.3,
        use_cuda: bool = True,
    ):
        self.yolo_dir = yolo_dir
        self.default_conf = confidence
        self.default_nms = threshold
        self.use_cuda = use_cuda

        # Resolve paths
        base_dir = os.path.dirname(os.path.abspath(__file__))
        if not os.path.exists(self.yolo_dir):
            alt_dir = os.path.join(base_dir, self.yolo_dir)
            if os.path.exists(alt_dir):
                self.yolo_dir = alt_dir

        self.labels_path = os.path.join(self.yolo_dir, "coco.names")
        self.config_path = os.path.join(self.yolo_dir, "yolov3.cfg")
        self.weights_path = os.path.join(self.yolo_dir, "yolov3.weights")

        self._validate_model_files()
        self._load_labels()
        self._load_network()

    def _validate_model_files(self) -> None:
        """Ensure all required model files are present and valid."""
        for path, min_size, desc in [
            (self.labels_path, 200, "Class labels"),
            (self.config_path, 1000, "Network configuration"),
            (self.weights_path, 100 * 1024 * 1024, "Model weights"),
        ]:
            if not os.path.exists(path):
                raise FileNotFoundError(
                    f"[ERROR] {desc} not found at: {path}\n"
                    f"Please make sure {os.path.basename(path)} is present in {self.yolo_dir}."
                )
            if os.path.getsize(path) < min_size:
                raise ValueError(
                    f"[ERROR] {desc} file at {path} seems to be a placeholder or corrupted ({os.path.getsize(path)} bytes).\n"
                    f"Please ensure full model weights are present."
                )


    def _load_labels(self) -> None:
        """Load COCO class labels and assign vibrant, deterministic colors."""
        with open(self.labels_path, "r", encoding="utf-8") as f:
            self.labels = [line.strip() for line in f.readlines() if line.strip()]

        # Generate vibrant, deterministic colors for each class
        np.random.seed(42)
        self.colors = np.random.randint(40, 255, size=(len(self.labels), 3), dtype="uint8").tolist()

    def _load_network(self) -> None:
        """Load YOLO Darknet network and configure hardware acceleration."""
        print(f"[INFO] Initializing YOLOv3 engine from '{self.yolo_dir}'...")
        start_t = time.time()
        self.net = cv2.dnn.readNetFromDarknet(self.config_path, self.weights_path)

        # Attempt hardware acceleration (CUDA) if requested
        if self.use_cuda:
            try:
                self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
                self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
                print("[INFO] Hardware acceleration: CUDA enabled.")
            except Exception:
                self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
                self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
                print("[INFO] CUDA unavailable, falling back to optimized CPU engine.")
        else:
            self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
            self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

        # Cache unconnected output layer names once for maximum FPS
        layer_names = self.net.getLayerNames()
        out_layers = self.net.getUnconnectedOutLayers()
        try:
            self.output_layers = [layer_names[i - 1] for i in out_layers.flatten()]
        except Exception:
            self.output_layers = [layer_names[i[0] - 1] for i in out_layers]

        load_dur = time.time() - start_t
        print(f"[INFO] YOLOv3 loaded successfully in {load_dur:.2f}s ({len(self.labels)} classes).")

    def detect(
        self,
        image: np.ndarray,
        confidence: Optional[float] = None,
        threshold: Optional[float] = None,
    ) -> Tuple[List[Dict], float]:
        """
        Run object detection inference on an input BGR image.

        Returns:
            Tuple of (detections list, inference time in seconds)
        """
        conf_thresh = confidence if confidence is not None else self.default_conf
        nms_thresh = threshold if threshold is not None else self.default_nms

        h, w = image.shape[:2]

        # Construct blob (416x416 standard Darknet input size)
        blob = cv2.dnn.blobFromImage(
            image, 1 / 255.0, (416, 416), swapRB=True, crop=False
        )
        self.net.setInput(blob)

        # Perform forward inference
        start_t = time.time()
        layer_outputs = self.net.forward(self.output_layers)
        inference_time = time.time() - start_t

        boxes = []
        confidences = []
        class_ids = []

        for output in layer_outputs:
            for detection in output:
                scores = detection[5:]
                class_id = int(np.argmax(scores))
                score = float(scores[class_id])

                if score > conf_thresh:
                    # Scale center coordinates and box dimensions
                    box = detection[0:4] * np.array([w, h, w, h])
                    center_x, center_y, box_w, box_h = box.astype("int")

                    x = int(center_x - (box_w / 2))
                    y = int(center_y - (box_h / 2))

                    boxes.append([x, y, int(box_w), int(box_h)])
                    confidences.append(score)
                    class_ids.append(class_id)

        # Apply Non-Maxima Suppression
        indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_thresh, nms_thresh)

        detections = []
        if len(indices) > 0:
            for i in indices.flatten():
                detections.append(
                    {
                        "class_id": class_ids[i],
                        "class_name": self.labels[class_ids[i]],
                        "confidence": confidences[i],
                        "box": boxes[i],  # [x, y, w, h]
                        "color": self.colors[class_ids[i]],
                    }
                )

        return detections, inference_time

    def draw_detections(
        self,
        image: np.ndarray,
        detections: List[Dict],
        display_scale: float = 1.0,
    ) -> np.ndarray:
        """
        Render crisp bounding boxes and filled badge tags on the image.
        Uses scale-aware line thickness and typography to prevent border clipping.
        """
        annotated = image.copy()
        h, w = annotated.shape[:2]

        # Calculate scale-invariant visual stroke and typography sizes
        line_thickness = max(2, int(3.5 / display_scale))
        font_scale = max(0.5, 0.65 / display_scale)
        font_thickness = max(1, int(1.8 / display_scale))
        pad = int(4 / display_scale)
        half_thick = line_thickness // 2 + 1

        for det in detections:
            x, y, bw, bh = det["box"]
            color = det["color"]
            label_text = f"{det['class_name']}: {det['confidence']:.1%}"

            # Clamping box strictly inside canvas to avoid edge cut-offs
            x1 = max(half_thick, min(w - half_thick, x))
            y1 = max(half_thick, min(h - half_thick, y))
            x2 = max(x1 + half_thick, min(w - half_thick, x + bw))
            y2 = max(y1 + half_thick, min(h - half_thick, y + bh))

            # Draw outer bounding box
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, line_thickness)

            # Compute label dimensions
            (text_w, text_h), baseline = cv2.getTextSize(
                label_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness
            )

            # Draw filled header badge
            if y1 - text_h - baseline - pad * 2 < 0:
                badge_y1 = y1
                badge_y2 = y1 + text_h + baseline + pad * 2
                text_y = badge_y2 - baseline - pad
            else:
                badge_y1 = y1 - text_h - baseline - pad * 2
                badge_y2 = y1
                text_y = y1 - baseline - pad

            badge_x2 = min(w, x1 + text_w + pad * 2)
            cv2.rectangle(annotated, (x1, badge_y1), (badge_x2, badge_y2), color, -1)

            # Compute luminance for adaptive text contrast (black vs white)
            luminance = 0.114 * color[0] + 0.587 * color[1] + 0.299 * color[2]
            text_color = (255, 255, 255) if luminance < 140 else (0, 0, 0)

            cv2.putText(
                annotated,
                label_text,
                (x1 + pad, text_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,
                text_color,
                font_thickness,
            )

        return annotated

    def create_display_image(
        self,
        image: np.ndarray,
        max_width: int = 500,
        max_height: int = 360,
    ) -> Tuple[np.ndarray, float]:
        """Resize an image to fit nicely within screen bounds while preserving aspect ratio."""
        h, w = image.shape[:2]
        scale = min(max_width / w, max_height / h, 1.0)
        if scale < 1.0:
            target_w = int(w * scale)
            target_h = int(h * scale)
            display_img = cv2.resize(image, (target_w, target_h), interpolation=cv2.INTER_AREA)
        else:
            display_img = image.copy()
        return display_img, scale
