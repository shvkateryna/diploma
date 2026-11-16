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


# ImageJ ROI binary format (ij/io/RoiDecoder.java)
_MAGIC   = b'Iout'  # ImageJ rejects anything else: "This is not an ImageJ ROI"
_VERSION = 228
_RECT    = 1
_SUB_PIXEL_RESOLUTION = 128  # options flag: read float bounds at XD..HEIGHTD
_SHORT_MAX = 32767           # integer bounds are signed 16-bit


def _rect_roi(x1: int, y1: int, x2: int, y2: int) -> bytes:
    h = bytearray(64)
    h[0:4] = _MAGIC
    struct.pack_into('>h', h, 4,  _VERSION)
    h[6] = _RECT
    # Integer bounds (top, left, bottom, right) only fit images up to 32767 px;
    # the float bounds below carry the exact values for larger orthomosaics.
    struct.pack_into('>hhhh', h, 8, *(min(_SHORT_MAX, max(0, v)) for v in (y1, x1, y2, x2)))
    struct.pack_into('>ffff', h, 18, float(x1), float(y1), float(x2 - x1), float(y2 - y1))
    struct.pack_into('>h', h, 50, _SUB_PIXEL_RESOLUTION)
    return bytes(h)


def to_roi_zip(boxes: list[tuple[int, int, int, int]]) -> bytes:
    """Pack every bounding box into an ImageJ-compatible RoiSet.zip."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for i, (x1, y1, x2, y2) in enumerate(boxes):
            zf.writestr(f"{i + 1:04d}-penguin.roi", _rect_roi(x1, y1, x2, y2))
    return buf.getvalue()
