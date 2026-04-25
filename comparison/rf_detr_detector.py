from .base_detector import BaseDetector, Detection
import numpy as np

class RFDETRDetector(BaseDetector):
    def __init__(self, weights: str, name: str = None):
        from rfdetr import RFDETRMedium
        self.model = RFDETRMedium(pretrain_weights=weights)
        self.name = name or f"RFDETR({weights})"

    def predict(self, image_path: str, conf: float) -> Detection:
        p = self.model.predict(images=[image_path], threshold=conf)
        p = p[0] if isinstance(p, list) else p
        boxes = np.array(p.xyxy) if p.xyxy is not None and len(p.xyxy) else np.zeros((0,4))
        scores = np.array(p.confidence) if p.confidence is not None else np.zeros(0)
        return Detection(boxes, scores)