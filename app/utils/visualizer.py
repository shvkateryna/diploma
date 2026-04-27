from io import BytesIO

import cv2
import numpy as np
from PIL import Image

_BOX_COLOR = (0, 0, 255)
_TXT_BG    = (0, 0, 200)
_TXT_COLOR = (255, 255, 255)


def draw_boxes(
    image_rgb: np.ndarray,
    boxes: list[tuple[int, int, int, int]],
    scores: list[float] | None = None,
) -> np.ndarray:
    out = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

    for i, (x1, y1, x2, y2) in enumerate(boxes):
        cv2.rectangle(out, (x1, y1), (x2, y2), _BOX_COLOR, 3)

        if scores:
            label = f"{scores[i]:.2f}"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.40, 1)
            cv2.rectangle(out, (x1, y1 - th - 7), (x1 + tw + 5, y1), _TXT_BG, -1)
            cv2.putText(
                out, label, (x1 + 2, y1 - 3),
                cv2.FONT_HERSHEY_SIMPLEX, 0.40, _TXT_COLOR, 1,
            )

    return cv2.cvtColor(out, cv2.COLOR_BGR2RGB)


def to_bytes(image_rgb: np.ndarray) -> bytes:
    buf = BytesIO()
    Image.fromarray(image_rgb).save(buf, format="PNG")
    return buf.getvalue()
