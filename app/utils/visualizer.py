from io import BytesIO

import cv2
import numpy as np
from PIL import Image

_BOX_COLOR = (255, 0, 0)       # RGB red
_TXT_COLOR = (255, 255, 255)   # white number...
_TXT_OUTLINE = (0, 0, 0)       # ...with a thin dark outline, readable on snow and rock
_FONT = cv2.FONT_HERSHEY_SIMPLEX
_FONT_SCALE = 0.35


def draw_boxes(
    image_rgb: np.ndarray,
    boxes: list[tuple[int, int, int, int]],
    labels: list[str] | None = None,
    inplace: bool = False,
) -> np.ndarray:
    """
    Draw thin 1 px boxes so the birds stay visible. Each label (the penguin's
    number) is written just above its box, not over the bird, without a filled
    background; below the box if there is no room above.
    With inplace=True the input array is drawn on directly (no copy), which
    matters for full-resolution images of hundreds of megabytes.
    """
    out = image_rgb if inplace else image_rgb.copy()

    for (x1, y1, x2, y2) in boxes:
        cv2.rectangle(out, (x1, y1), (x2, y2), _BOX_COLOR, 1)

    # Labels after all boxes, so a neighbour's box never crosses a number
    if labels:
        for (x1, y1, _, y2), label in zip(boxes, labels):
            (_, th), _ = cv2.getTextSize(label, _FONT, _FONT_SCALE, 1)
            y = y1 - 2 if y1 - th - 2 >= 0 else y2 + th + 2
            for color, thickness in ((_TXT_OUTLINE, 2), (_TXT_COLOR, 1)):
                cv2.putText(out, label, (x1, y), _FONT, _FONT_SCALE,
                            color, thickness, cv2.LINE_AA)

    return out


def to_bytes(image_rgb: np.ndarray, compress_level: int = 6) -> bytes:
    buf = BytesIO()
    Image.fromarray(image_rgb).save(buf, format="PNG", compress_level=compress_level)
    return buf.getvalue()
