"""
================================================================================
 YOLOv3 Object Detection CLI
 Author: Saksham Gupta (https://github.com/sakshamgupta04)
 Repository: https://github.com/sakshamgupta04/Object-Detection-Using-Yolo
================================================================================
"""

import argparse
import glob
import os
import sys
import time
import cv2

from detector import YOLODetector


BANNER = r"""
====================================================================
     __     ______  _      ____   ____       _            _   
     \ \   / / __ \| |    / __ \ |  _ \  ___| |_ ___  ___| |_ 
      \ \ / / |  | | |   | |  | || | | |/ _ \ __/ _ \/ __| __|
       \ V /| |__| | |___| |__| || |_| |  __/ ||  __/ (__| |_ 
        \_/  \____/|______\____/ |____/ \___|\__\___|\___|\__|
                                                              
                  Object Detection with OpenCV
                   Maintained by Saksham Gupta
====================================================================
"""


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Run YOLOv3 Object Detection on images, image directories, videos, or live webcam."
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("-i", "--image", type=str, help="Path to input image")
    group.add_argument("--image-dir", type=str, help="Directory containing images to process")
    group.add_argument("-v", "--video", type=str, help="Path to input video file")
    group.add_argument("--webcam", action="store_true", help="Use live webcam feed for real-time detection")

    parser.add_argument("-o", "--output", type=str, default=None, help="Path to save output image/video")
    parser.add_argument("--output-dir", type=str, default="results", help="Directory to save batch processed images")
    parser.add_argument("-y", "--yolo", type=str, default="yolo-coco", help="Directory containing YOLO configuration and weights")
    parser.add_argument("-c", "--confidence", type=float, default=0.5, help="Minimum probability threshold for weak detections")
    parser.add_argument("-t", "--threshold", type=float, default=0.3, help="Non-Maxima Suppression (NMS) threshold")
    parser.add_argument("--display-width", type=int, default=500, help="Maximum preview window width (default: 500)")
    parser.add_argument("--display-height", type=int, default=360, help="Maximum preview window height (default: 360)")
    parser.add_argument("--no-display", action="store_true", help="Run without showing preview GUI window")
    parser.add_argument("--no-cuda", action="store_true", help="Disable CUDA acceleration and run on CPU only")

    return parser.parse_args()


def print_summary(detections, inference_time):
    print(f"\n[SUMMARY] Inference Time: {inference_time*1000:.1f}ms (~{1.0/max(1e-5, inference_time):.1f} FPS)")
    print(f"[SUMMARY] Total Detections: {len(detections)}")
    if detections:
        for idx, det in enumerate(detections, 1):
            print(f"  [{idx:02d}] {det['class_name']:<15} | Confidence: {det['confidence']*100:.1f}%")
    else:
        print("  No objects detected meeting confidence criteria.")
    print("--------------------------------------------------------------------")


def process_single_image(detector, image_path, args, is_batch=False):
    if not os.path.exists(image_path):
        print(f"[ERROR] Image not found: {image_path}")
        return

    image = cv2.imread(image_path)
    if image is None:
        print(f"[ERROR] Could not decode image: {image_path}")
        return

    print(f"\n[INFO] Processing image: {image_path} ({image.shape[1]}x{image.shape[0]})")
    detections, infer_time = detector.detect(image, args.confidence, args.threshold)
    print_summary(detections, infer_time)

    # Compute display scaling factor
    _, display_scale = detector.create_display_image(
        image, max_width=args.display_width, max_height=args.display_height
    )

    annotated = detector.draw_detections(image, detections, display_scale=display_scale)

    # Save output if specified
    out_path = args.output
    if is_batch:
        os.makedirs(args.output_dir, exist_ok=True)
        filename = os.path.basename(image_path)
        out_path = os.path.join(args.output_dir, f"detected_{filename}")

    if out_path:
        cv2.imwrite(out_path, annotated)
        print(f"[INFO] Annotated image saved to: {out_path}")

    # Display preview window if enabled
    if not args.no_display:
        display_img, _ = detector.create_display_image(
            annotated, max_width=args.display_width, max_height=args.display_height
        )
        win_title = f"YOLOv3 Detection - {os.path.basename(image_path)}"
        cv2.namedWindow(win_title, cv2.WINDOW_NORMAL)
        cv2.imshow(win_title, display_img)
        print("[INFO] Press any key in the image window to continue (or 'q' to exit batch)...")
        key = cv2.waitKey(0) & 0xFF
        cv2.destroyWindow(win_title)
        if key == ord('q'):
            return False
    return True


def process_video_stream(detector, source, args, is_webcam=False):
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"[ERROR] Could not open video source: {source}")
        return

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0 or fps > 120:
        fps = 25.0

    writer = None
    if args.output:
        fourcc = cv2.VideoWriter_fourcc(*"MJPG")
        writer = cv2.VideoWriter(args.output, fourcc, fps, (width, height), True)
        print(f"[INFO] Writing output video to: {args.output}")

    source_desc = "Live Webcam" if is_webcam else f"Video file: {source}"
    print(f"\n[INFO] Starting video processing for {source_desc}...")
    print("[INFO] Press 'q' or 'ESC' in the display window to stop.\n")

    frame_count = 0
    start_total = time.time()

    win_title = "YOLOv3 Real-Time Stream - Saksham Gupta"
    if not args.no_display:
        cv2.namedWindow(win_title, cv2.WINDOW_NORMAL)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        detections, infer_time = detector.detect(frame, args.confidence, args.threshold)

        _, display_scale = detector.create_display_image(
            frame, max_width=args.display_width, max_height=args.display_height
        )
        annotated = detector.draw_detections(frame, detections, display_scale=display_scale)

        # Overlay FPS on the video frame
        current_fps = 1.0 / max(1e-5, infer_time)
        fps_text = f"FPS: {current_fps:.1f} | Detections: {len(detections)}"
        cv2.putText(
            annotated, fps_text, (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2
        )

        if writer is not None:
            writer.write(annotated)

        if not args.no_display:
            disp_frame, _ = detector.create_display_image(
                annotated, max_width=args.display_width, max_height=args.display_height
            )
            cv2.imshow(win_title, disp_frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q") or key == 27:
                print("[INFO] Stream interrupted by user.")
                break

    cap.release()
    if writer:
        writer.release()
    if not args.no_display:
        cv2.destroyAllWindows()

    total_time = time.time() - start_total
    avg_fps = frame_count / max(1e-5, total_time)
    print(f"\n[INFO] Finished! Processed {frame_count} frames in {total_time:.1f}s ({avg_fps:.1f} avg FPS).")


def main():
    print(BANNER)
    args = parse_arguments()

    # Check for weights and auto-download if missing
    weights_path = os.path.join(args.yolo, "yolov3.weights")
    if not os.path.exists(weights_path) or os.path.getsize(weights_path) < 1000:
        print("[WARNING] yolov3.weights is missing or incomplete.")
        import download_weights
        success = download_weights.ensure_weights()
        if not success:
            sys.exit(1)

    detector = YOLODetector(
        yolo_dir=args.yolo,
        confidence=args.confidence,
        threshold=args.threshold,
        use_cuda=not args.no_cuda,
    )

    if args.image:
        process_single_image(detector, args.image, args)

    elif args.image_dir:
        extensions = ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.webp")
        files = []
        for ext in extensions:
            files.extend(glob.glob(os.path.join(args.image_dir, ext)))
            files.extend(glob.glob(os.path.join(args.image_dir, ext.upper())))
        files = sorted(list(set(files)))

        if not files:
            print(f"[ERROR] No valid images found in directory: {args.image_dir}")
            sys.exit(1)

        print(f"[INFO] Found {len(files)} image(s) to process in {args.image_dir}.")
        for f in files:
            continue_processing = process_single_image(detector, f, args, is_batch=True)
            if continue_processing is False:
                print("[INFO] Batch processing cancelled by user.")
                break

    elif args.video:
        process_video_stream(detector, args.video, args, is_webcam=False)

    elif args.webcam:
        process_video_stream(detector, 0, args, is_webcam=True)


if __name__ == "__main__":
    main()
