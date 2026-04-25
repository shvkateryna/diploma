from .base_detector import BaseDetector, Detection
import numpy as np


class BirdDetector(BaseDetector):
    def __init__(
        self,
        name: str = "BirdDetector",
        use_release: bool = True,
        checkpoint_path: str = None,
        config_args: dict = None,
    ):
        from deepforest import main as df_main

        self.name = name

        if config_args is None:
            config_args = {
                "num_classes": 1,
                "label_dict": {"Penguine": 0}
            }

        if use_release:
            self.model = df_main.deepforest(label_dict=config_args["label_dict"])
            self.model.use_bird_release()
        else:
            if checkpoint_path is None:
                raise ValueError("checkpoint_path must be provided")

            self.model = df_main.deepforest.load_from_checkpoint(
                checkpoint_path
            )

    def predict(self, image_path: str, conf: float) -> Detection:
        df = self.model.predict_image(path=image_path)

        if df is None or len(df) == 0:
            return Detection(
                np.zeros((0, 4), dtype=float),
                np.zeros((0,), dtype=float)
            )

        if "score" in df.columns:
            df = df[df["score"] >= conf]

        if len(df) == 0:
            return Detection(
                np.zeros((0, 4), dtype=float),
                np.zeros((0,), dtype=float)
            )

        boxes = df[["xmin", "ymin", "xmax", "ymax"]].values.astype(float)
        scores = (
            df["score"].values.astype(float)
            if "score" in df.columns
            else np.ones(len(df), dtype=float)
        )

        return Detection(boxes, scores)