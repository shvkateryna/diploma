import numpy as np


def apply_nms(
    boxes: list[tuple[int, int, int, int]],
    scores: list[float],
    iou_threshold: float,
) -> tuple[list, list]:
    if not boxes:
        return [], []

    b = np.array(boxes, dtype=np.float32)
    s = np.array(scores, dtype=np.float32)
    order = s.argsort()[::-1]
    keep: list[int] = []

    while order.size > 0:
        i = int(order[0])
        keep.append(i)
        if order.size == 1:
            break
        rest = order[1:]
        x1 = np.maximum(b[i, 0], b[rest, 0])
        y1 = np.maximum(b[i, 1], b[rest, 1])
        x2 = np.minimum(b[i, 2], b[rest, 2])
        y2 = np.minimum(b[i, 3], b[rest, 3])
        inter = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)
        area_i = (b[i, 2] - b[i, 0]) * (b[i, 3] - b[i, 1])
        area_r = (b[rest, 2] - b[rest, 0]) * (b[rest, 3] - b[rest, 1])
        iou = inter / (area_i + area_r - inter + 1e-6)
        order = rest[iou < iou_threshold]

    return [boxes[i] for i in keep], [scores[i] for i in keep]
