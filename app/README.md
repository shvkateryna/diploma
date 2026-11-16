# Penguin Detector

A Streamlit web application for detecting penguins in aerial and satellite imagery using deep learning object detection models. Supports large geospatial images through tile-based processing and exports results to multiple formats.

---

## Quick Start

### Prerequisites

- [Docker](https://docs.docker.com/get-docker/) and [Docker Compose](https://docs.docker.com/compose/install/)
- At least 8 GB RAM available to the Docker daemon

### Run

```bash
git clone <repo-url>
cd app
docker compose up --build
```

Open [http://localhost:8501](http://localhost:8501) in your browser.

> **First build** downloads PyTorch and all dependencies — takes 5–10 minutes. Subsequent starts are instant.

### Stop

```bash
docker compose down
```

---

## Features

| Feature | Details |
|---|---|
| **Two detection models** | YOLOv11 (augmented) and RF-DETR (baseline) selectable from the sidebar |
| **Large image support** | Tile-based streaming — processes arbitrarily large GeoTIFF/PNG/JPEG without loading the full image into RAM |
| **Interactive confidence** | Slider updates detections and visualisation instantly without re-running inference |
| **Progress tracking** | Live progress bar with tile-by-tile status |
| **Metrics** | Total count, average confidence, max confidence, processing time |
| **Three export formats** | ImageJ ROI zip, annotated PNG, Label Studio JSON |

---

## How to Use

1. **Select a model** in the left sidebar (YOLOv11 or RF-DETR).
2. **Set the confidence threshold** — only detections above this score are shown.
3. **Set the image resolution** (see below).
4. **Upload an aerial image** (JPEG, PNG, TIFF, GeoTIFF).
5. Click **Detect Penguins**.
6. Adjust the confidence slider after detection — the results update without re-running the model.
7. Download results in your preferred format.

> If you lower the slider below the threshold used during detection, a notice appears asking you to re-run with the new threshold.

### Image resolution

The models were trained on photos taken from 80–100 m (about 2.2 cm of ground per pixel). From higher up penguins look smaller in pixels and are missed, so the app rescales every image to the training resolution. Tell it the resolution in the sidebar:

| Option | Use for |
|---|---|
| **Auto (GeoTIFF)** | Georeferenced GeoTIFF orthomosaics — the pixel size is read from the file. JPEG/PNG are processed at original scale. |
| **Flight altitude** | Raw drone photos (DJI Phantom 4 Pro camera). 120 m → 3.3 cm/px → image enlarged ×1.5. |
| **Pixel size** | Orthomosaics exported as JPEG/PNG: enter the pixel size chosen at export (Metashape: *Export Orthomosaic → Pixel size*). |

The pixel size used and the rescale factor are shown above the results.

---

## Project Structure

```
app/
├── app.py                  # Streamlit entry point
├── config.yaml             # Pipeline and model configuration
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
│
├── core/
│   ├── config.py           # Constants and config loader
│   ├── styles.py           # Custom CSS
│   └── components.py       # Reusable UI components
│
├── models/
│   ├── model_registry.py   # Model factory (YOLOModel, RFDETRModel)
│   ├── yolo_aug.pt         # YOLOv11 weights
│   └── rfdetr.pth          # RF-DETR weights
│
├── pipeline/
│   ├── gsd.py              # Ground resolution from GeoTIFF metadata / altitude
│   ├── tiler.py            # Splits images into overlapping tiles
│   ├── inference.py        # Batch inference orchestration
│   ├── nms.py              # Non-Maximum Suppression
│   └── counter.py          # Detection metrics
│
└── utils/
    ├── exporter.py         # ROI zip and Label Studio JSON export
    └── visualizer.py       # Bounding box drawing and image encoding
```

---

## Modules

### `core/`

- **`config.py`** — loads `config.yaml` and provides app-wide constants (penguin facts, etc.)
- **`styles.py`** — injects custom CSS (Inter font, gradient header, badge pills, metric cards)
- **`components.py`** — renders the hero banner at the top of the page

### `models/model_registry.py`

Factory that loads and wraps detection models behind a common interface:

```
load(name) → YOLOModel | RFDETRModel
```

Both classes expose:
- `predict_tile(tile_rgb, conf)` — single tile
- `predict_batch(tiles, conf)` — list of tiles

**`YOLOModel`** uses the `ultralytics` library and supports native batching.

**`RFDETRModel`** uses the `rfdetr` library. GPU is used automatically when available, otherwise CPU. Batch processing is emulated (sequential tiles).

Models are registered in `config.yaml`:

```yaml
models:
  yolo_augmentation:
    path: models/yolo_aug.pt
    type: ultralytics
    label: "YOLOv11 — Augmented"
  rfdetr_baseline:
    path: models/rfdetr.pth
    type: rfdetr
    label: "RF-DETR — Baseline"
```

### `pipeline/`

**`tiler.py`** — Splits the source image into 1024×1024 tiles with 20% overlap.
- GeoTIFF images are streamed via rasterio windows (constant RAM regardless of file size).
- JPEG/PNG images are first converted to a temporary GeoTIFF, then tiled the same way.
- Each tile carries its `(x, y)` origin so detections can be mapped back to original coordinates.
- With a `scale` ≠ 1 it reads a `tile_size / scale` window and resamples it to `tile_size`, so penguins appear at the training size.

**`gsd.py`** — Ground sampling distance (cm/px): read from GeoTIFF georeferencing (projected or lat/lon CRS) or computed from flight altitude and camera.

**`inference.py`** — Orchestrates tiling and model inference.
- Tiles are accumulated into batches (default: 4) and flushed through the model.
- Detected box coordinates are scaled and offset back to the full image coordinate system.
- Accepts an optional `on_tile` progress callback.

**`nms.py`** — Standard greedy Non-Maximum Suppression.
- Sorts boxes by confidence, suppresses overlapping boxes with IoU > threshold.
- IoU threshold is set to `0.5` in `config.yaml`.

**`counter.py`** — Aggregates detection statistics: total count, average confidence, max confidence.

### `utils/`

**`visualizer.py`**
- `draw_boxes(image, boxes, labels)` — draws thin red boxes with each penguin's number above it. Penguins are numbered top to bottom, left to right; the numbers match the ROI names in `RoiSet.zip` (`0001-penguin`, …).
- `to_bytes(image)` — encodes a numpy array to PNG bytes for download.

**`exporter.py`**
- `to_roi_zip(boxes)` — writes boxes to an ImageJ-compatible ROI zip archive.
- `to_label_studio_json(boxes, scores, w, h, filename)` — serialises detections to Label Studio task JSON with normalised coordinates.

---

## Configuration

All pipeline parameters live in `config.yaml`:

```yaml
pipeline:
  tile_size: 1024       # pixels per tile side
  overlap: 0.2          # fraction overlap between adjacent tiles
  iou_threshold: 0.5    # NMS IoU threshold
  batch_size: 4         # tiles per inference batch

gsd:
  reference_cm: 2.2     # GSD of the training tiles (Phantom 4 Pro, 80 m)
  camera:               # converts "Flight altitude" to GSD
    sensor_width_mm: 13.2
    focal_length_mm: 8.8
    image_width_px: 5472
  min_scale: 0.25       # refuse rescaling outside this range
  max_scale: 4.0
```

---

## Export Formats

| Format | Use case |
|---|---|
| **RoiSet.zip** | Open in Fiji / ImageJ: `Analyze → Tools → ROI Manager → More → Open` |
| **PNG** | Annotated image for reports and presentations |
| **Label Studio JSON** | Import into Label Studio for manual review or re-training: `Tasks → Import → JSON` |

---

## Memory and Performance

- **RAM limit**: 8 GB (set in `docker-compose.yml`, adjust `mem_limit` for larger images).
- **Tile streaming** keeps peak RAM proportional to `batch_size × tile_size²`, not image size.
- To process faster, increase `batch_size` in `config.yaml` (requires more RAM).
- The model is loaded once and cached for the session — switching between images does not reload it.

---

## Supported Input Formats

| Format | Notes |
|---|---|
| GeoTIFF / TIFF | Natively streamed, any size |
| JPEG | Converted to temporary TIFF before tiling |
| PNG | Converted to temporary TIFF before tiling |

---

## Troubleshooting

**Not enough RAM**
```
Increase mem_limit in docker-compose.yml, e.g. mem_limit: 16g
```

**Model weights not found**
```
FileNotFoundError: Weights not found: models/rfdetr.pth
```
Place the `.pt` / `.pth` weight files in the `models/` directory and rebuild.

**Confidence slider shows "Re-run detection"**  
The slider was moved below the threshold used when detection was last run. Click **Detect Penguins** again to fetch detections at the lower threshold.
