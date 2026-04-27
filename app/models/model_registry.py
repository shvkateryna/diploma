from pathlib import Path

import numpy as np
import yaml
from PIL import Image


def load_config() -> dict:
    with open("config.yaml") as f:
        return yaml.safe_load(f)


def list_models() -> dict[str, str]:
    """Returns {model_key: display_label} for the sidebar selector."""
    return {name: info["label"] for name, info in load_config()["models"].items()}


class YOLOModel:
    def __init__(self, weights_path: str):
        from ultralytics import YOLO
        self.model = YOLO(weights_path)

    def predict_tile(
        self, tile_rgb: np.ndarray, conf: float
    ) -> tuple[list[tuple[int, int, int, int]], list[float]]:
        return self.predict_batch([tile_rgb], conf)[0]

    def predict_batch(
        self, tiles: list[np.ndarray], conf: float
    ) -> list[tuple[list[tuple[int, int, int, int]], list[float]]]:
        results = self.model.predict(tiles, conf=conf, verbose=False)
        output = []
        for r in results:
            b, s = [], []
            if r.boxes is not None:
                for box in r.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                    b.append((x1, y1, x2, y2))
                    s.append(float(box.conf[0]))
            output.append((b, s))
        return output


class RFDETRModel:
    def __init__(self, weights_path: str):
        from rfdetr import RFDETRMedium
        self.model = RFDETRMedium(pretrain_weights=weights_path)

    def predict_tile(
        self, tile_rgb: np.ndarray, conf: float
    ) -> tuple[list[tuple[int, int, int, int]], list[float]]:
        pil = Image.fromarray(tile_rgb)
        result = self.model.predict(pil, threshold=conf)
        boxes = [
            (int(x1), int(y1), int(x2), int(y2))
            for x1, y1, x2, y2 in result.xyxy
        ]
        return boxes, result.confidence.tolist()

    def predict_batch(
        self, tiles: list[np.ndarray], conf: float
    ) -> list[tuple[list[tuple[int, int, int, int]], list[float]]]:
        # RF-DETR does not expose a native batch API; process tiles sequentially
        return [self.predict_tile(t, conf) for t in tiles]


def load(name: str) -> YOLOModel | RFDETRModel:
    cfg = load_config()
    info = cfg["models"][name]
    path = info["path"]

    if not Path(path).exists():
        raise FileNotFoundError(f"Weights not found: {path}")

    if info["type"] == "ultralytics":
        return YOLOModel(path)
    if info["type"] == "rfdetr":
        return RFDETRModel(path)
    raise ValueError(f"Unknown model type: {info['type']}")
