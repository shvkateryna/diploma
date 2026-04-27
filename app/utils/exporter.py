import io
import json
import struct
import zipfile

def to_label_studio_json(
    boxes: list[tuple[int, int, int, int]],
    scores: list[float],
    img_w: int,
    img_h: int,
    filename: str = "image.jpg",
) -> bytes:
    results = []
    for i, ((x1, y1, x2, y2), score) in enumerate(zip(boxes, scores)):
        results.append({
            "id": f"result_{i + 1}",
            "type": "rectanglelabels",
            "from_name": "label",
            "to_name": "image",
            "original_width": img_w,
            "original_height": img_h,
            "value": {
                "x": round(x1 / img_w * 100, 4),
                "y": round(y1 / img_h * 100, 4),
                "width": round((x2 - x1) / img_w * 100, 4),
                "height": round((y2 - y1) / img_h * 100, 4),
                "rotation": 0,
                "rectanglelabels": ["penguin"],
            },
        })
    task = [{
        "data": {"image": filename},
        "predictions": [{
            "model_version": "penguin-detector",
            "result": results,
        }],
    }]
    return json.dumps(task, indent=2).encode("utf-8")


_MAGIC   = b'Iroi'
_VERSION = 228
_RECT    = 1


def _rect_roi(x1: int, y1: int, x2: int, y2: int) -> bytes:
    h = bytearray(64)
    h[0:4] = _MAGIC
    struct.pack_into('>H', h, 4,  _VERSION)
    h[6] = _RECT
    struct.pack_into('>H', h, 8,  min(65535, max(0, y1)))
    struct.pack_into('>H', h, 10, min(65535, max(0, x1)))
    struct.pack_into('>H', h, 12, min(65535, max(0, y2)))
    struct.pack_into('>H', h, 14, min(65535, max(0, x2)))
    return bytes(h)


def to_roi_zip(boxes: list[tuple[int, int, int, int]]) -> bytes:
    """Pack every bounding box into an ImageJ-compatible RoiSet.zip."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for i, (x1, y1, x2, y2) in enumerate(boxes):
            zf.writestr(f"{i + 1:04d}-penguin.roi", _rect_roi(x1, y1, x2, y2))
    return buf.getvalue()
