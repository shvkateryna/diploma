from .base_detector import BaseDetector, Detection
import numpy as np

class YOLODetector(BaseDetector):
    def __init__(self, weights: str, name: str = None):
        from ultralytics import YOLO
        self.model = YOLO(weights)
        self.name = name or f"YOLO({weights})"

    def predict(self, image_path: str, conf: float) -> Detection:
        res = self.model(image_path, conf=conf, verbose=False)[0]
        boxes = res.boxes.xyxy.cpu().numpy() if len(res.boxes) else np.zeros((0,4))
        scores = res.boxes.conf.cpu().numpy() if len(res.boxes) else np.zeros(0)
        return Detection(boxes, scores)