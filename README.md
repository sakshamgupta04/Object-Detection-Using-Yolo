# 🚀 YOLOv3 Object Detection with OpenCV

[![Python Version](https://img.shields.io/badge/python-3.8%20%7C%203.9%20%7C%203.10%20%7C%203.11%20%7C%203.12-blue.svg)](https://www.python.org/)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.x-brightgreen.svg)](https://opencv.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Maintained By](https://img.shields.io/badge/Maintained%20by-Saksham%20Gupta-orange.svg)](https://github.com/sakshamgupta04)

An optimized, production-ready implementation of **YOLOv3 (You Only Look Once)** real-time object detection built with **OpenCV DNN** and **Python**.

Supports **images**, **batch directory inference**, **video files**, and **live webcam streams** with automated resolution scaling, high-contrast label badges, and intelligent bounding-box clipping.

---

## ✨ Key Features & Enhancements

- 🎯 **Multi-Input Pipeline**: Process single images, entire image directories, video files, or real-time webcam streams.
- ⚡ **OpenCV DNN Acceleration**: Automatically detects and uses NVIDIA CUDA GPU acceleration when available, with a fast CPU fallback.
- 📐 **Adaptive Display & Scaling**: Smart auto-scaling keeps previews neatly framed on any monitor resolution (no truncated or oversized windows).
- 🏷️ **Modern High-Contrast UI**: High-contrast filled badge tags and scale-aware line strokes keep bounding boxes bold and readable across all image sizes (from 360p to 4K).
- 🛡️ **Edge-Safe Clamping**: Prevents OpenCV bounding boxes and text from clipping outside the canvas when objects hit the screen border.
- 📦 **Automated Weight Downloader**: Built-in `download_weights.py` helper checks and downloads official Darknet YOLOv3 weights with a progress bar.
- 🧱 **Modular Object-Oriented Architecture**: Clean `YOLODetector` class in `detector.py` for effortless integration into other applications.

---

## 📸 Detection Previews

| Object Detection in Complex Scenes | Urban Motorcycle Detection |
| :---: | :---: |
| ![Baggage Claim Detection](Object%20dection%20using%20image/1.png) | ![Soccer Field Detection](Object%20dection%20using%20image/2.png) |

---

## 🛠️ Installation

### 1. Clone the Repository
```bash
git clone https://github.com/sakshamgupta04/Object-Detection-Using-Yolo.git
cd Object-Detection-Using-Yolo
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Download YOLOv3 Model Weights
You can download the pre-trained COCO weights (~237 MB) automatically:
```bash
python download_weights.py
```
*(Or let `python yolo.py` automatically detect and prompt download on first run).*

---

## 🚀 Quick Start & Usage

### 🖼️ 1. Detect Objects in an Image
```bash
python yolo.py --image images/Temo.jpeg
```
**Save the annotated detection:**
```bash
python yolo.py --image images/baggage_claim.jpg --output results/output.jpg
```

---

### 📂 2. Batch Process an Entire Directory
```bash
python yolo.py --image-dir images/ --output-dir results/
```

---

### 🎥 3. Detect Objects in a Video File
```bash
python yolo.py --video "Object detection using video/videos/airport.mp4" --output results/airport_detected.avi
```

---

### 📹 4. Real-Time Detection with Live Webcam
```bash
python yolo.py --webcam
```
*(Press `q` or `ESC` in the display window at any time to exit).*

---

## ⚙️ Command-Line Arguments Reference

| Argument | Type | Default | Description |
| :--- | :---: | :---: | :--- |
| `-i, --image` | `str` | `None` | Path to single input image |
| `--image-dir` | `str` | `None` | Directory of images for batch processing |
| `-v, --video` | `str` | `None` | Path to input video file |
| `--webcam` | `flag` | `False` | Run live stream from default webcam |
| `-o, --output` | `str` | `None` | Path to save output image or video |
| `--output-dir` | `str` | `results` | Directory to save batch results |
| `-c, --confidence`| `float` | `0.5` | Minimum confidence threshold to filter weak detections |
| `-t, --threshold` | `float` | `0.3` | Non-Maxima Suppression (NMS) overlap threshold |
| `--display-width` | `int` | `500` | Maximum preview window width |
| `--display-height`| `int` | `360` | Maximum preview window height |
| `--no-display` | `flag` | `False` | Run headlessly without opening a GUI window |
| `--no-cuda` | `flag` | `False` | Force CPU inference even if CUDA is available |

---

## 🧠 Python API Integration

You can easily import `YOLODetector` into your own Python projects:

```python
import cv2
from detector import YOLODetector

# Initialize detector
detector = YOLODetector(yolo_dir="yolo-coco", confidence=0.5, threshold=0.3)

# Load image
img = cv2.imread("images/soccer.jpg")

# Run inference
detections, inference_time = detector.detect(img)

for det in detections:
    print(f"Detected {det['class_name']} with {det['confidence']:.1%} confidence at {det['box']}")

# Draw sleek bounding boxes
annotated = detector.draw_detections(img, detections)
cv2.imwrite("annotated.jpg", annotated)
```

---

## 🏷️ Supported Classes (COCO Dataset)

YOLOv3 detects 80 everyday object categories including:
- **People & Vehicles:** Person, bicycle, car, motorbike, aeroplane, bus, train, truck, boat, traffic light.
- **Animals:** Bird, cat, dog, horse, sheep, cow, elephant, bear, zebra, giraffe.
- **Accessories & Sports:** Backpack, umbrella, handbag, suitcase, sports ball, skateboard, tennis racket.
- **Indoor Objects:** Chair, sofa, potted plant, bed, dining table, laptop, mouse, cell phone, refrigerator, clock, book.

---

## 👤 Author

**Saksham Gupta**
- GitHub: [@sakshamgupta04](https://github.com/sakshamgupta04)
- Repository: [Object-Detection-Using-Yolo](https://github.com/sakshamgupta04/Object-Detection-Using-Yolo)

---

## 📄 License
This project is open-sourced under the [MIT License](LICENSE).
