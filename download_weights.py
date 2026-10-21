"""
================================================================================
 YOLOv3 Weights Downloader
 Author: Saksham Gupta (https://github.com/sakshamgupta04)
 Repository: https://github.com/sakshamgupta04/Object-Detection-Using-Yolo
================================================================================
"""

import os
import sys
import urllib.request

WEIGHTS_URL = "https://pjreddie.com/media/files/yolov3.weights"
DEST_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "yolo-coco")
DEST_FILE = os.path.join(DEST_DIR, "yolov3.weights")
MIN_VALID_SIZE = 240 * 1024 * 1024  # ~240 MB


def download_progress(block_num, block_size, total_size):
    downloaded = block_num * block_size
    if total_size > 0:
        percent = min(100.0, downloaded * 100.0 / total_size)
        mb_down = downloaded / (1024 * 1024)
        mb_total = total_size / (1024 * 1024)
        progress_bar = ("=" * int(percent / 2)).ljust(50, "-")
        sys.stdout.write(f"\r[{progress_bar}] {percent:5.1f}% ({mb_down:6.1f} / {mb_total:6.1f} MB)")
        sys.stdout.flush()


def ensure_weights():
    os.makedirs(DEST_DIR, exist_ok=True)

    if os.path.exists(DEST_FILE) and os.path.getsize(DEST_FILE) >= MIN_VALID_SIZE:
        size_mb = os.path.getsize(DEST_FILE) / (1024 * 1024)
        print(f"[OK] yolov3.weights already exists and verified ({size_mb:.1f} MB).")
        return True

    print("[INFO] yolov3.weights not found or invalid.")
    print(f"[INFO] Downloading official YOLOv3 weights from: {WEIGHTS_URL}")
    print(f"[INFO] Target: {DEST_FILE}")

    try:
        urllib.request.urlretrieve(WEIGHTS_URL, DEST_FILE, reporthook=download_progress)
        print("\n[SUCCESS] Download completed successfully!")
        return True
    except Exception as e:
        print(f"\n[ERROR] Failed to download weights: {e}")
        print("You can manually download yolov3.weights and place it in the 'yolo-coco/' directory.")
        return False


if __name__ == "__main__":
    ensure_weights()
