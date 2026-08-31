# detector.py — YOLOv8-based face detection (ultralytics)
# If local weights are missing, detection gracefully returns no faces (no auto-download).

import os
from typing import List

try:
    from ultralytics import YOLO
except Exception:
    YOLO = None  # optional

import cv2
from PIL import Image
import numpy as np

# Using a pretrained YOLOv8n model for face detection.
# For better accuracy, replace with a face-specific YOLO model. Place weights locally to avoid network fetch.

MODEL_PATH = os.getenv('YOLO_WEIGHTS', os.path.join('models', 'yolov8n.pt'))
_model = None

def _load_model():
    global _model
    if _model is not None:
        return _model
    if YOLO is None:
        return None
    if not os.path.exists(MODEL_PATH):
        return None
    _model = YOLO(MODEL_PATH)
    return _model

def detect_faces(image_path) -> List[Image.Image]:
    '''
    Runs YOLOv8 detection and returns a list of cropped PIL.Image faces.
    If weights/model are unavailable, returns an empty list.
    '''
    model = _load_model()
    if model is None:
        return []
    results = model.predict(image_path, conf=0.5)  # Higher confidence for better face detection
    img = Image.open(image_path).convert('RGB')
    w, h = img.size
    crops = []
    for r in results:
        for box in r.boxes:
            cls_id = int(box.cls)
            conf = float(box.conf)
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            # Heuristic filter: only keep boxes not too large and near top/middle of frame.
            if conf < 0.5:  # Higher confidence threshold
                continue
            # crop
            x1, y1, x2, y2 = [max(0, int(x1)), max(0, int(y1)), int(x2), int(y2)]
            face_crop = img.crop((x1, y1, x2, y2))
            crops.append(face_crop)
    return crops

if __name__ == '__main__':
    print("YOLOv8 face detection test: run detect_faces('path/to/image.jpg')")
