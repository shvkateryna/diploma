# Developing a Computer Vision System for Penguin Counting at the Vernadsky Research Base

Bachelor thesis project — Ukrainian Catholic University, 2025

**Author:** Kateryna Shvahuliak  
**Supervisors:** Ph.D. Taras Firman, Svitozar Davydenko  
**Department:** Computer Sciences and Information Technologies, Faculty of Applied Sciences

---

## Overview

Penguin populations are important biological indicators of environmental change in Antarctica. At Ukraine's Vernadsky Research Base, colony size is currently estimated through manual counting of UAV orthomosaics — a process that can take several hours per survey. This project develops a computer vision system that automates penguin detection and counting from high-resolution UAV imagery, reducing analysis time to under 15 minutes.

Four object detection architectures are evaluated on a multi-site Antarctic dataset:

| Model | Type | Best mAP@0.5 | Count Error |
|---|---|---|---|
| BirdDetector | Zero-shot RetinaNet baseline | 0.07 (uk_test) | −3.6% to +77.4% |
| YOLOv11 (augmented) | CNN one-stage | 0.834 (uk_test) | **−2.0%** |
| RT-DETR | Transformer | 0.810 (uk_test) | +14.8% |
| RF-DETR | Transformer | **0.942** (uk_test) | +9.8% |

The two best-performing models — `YOLO_augmentation` (most accurate counting) and `RF-DETR` (highest detection sensitivity) — are integrated into a deployable Streamlit web application.

---

## Demo

<!-- VIDEO DEMO PLACEHOLDER — replace this section with the actual video when ready -->
> **Demo video coming soon.**

---

## Repository Structure

```
diploma/
├── app/                        # Streamlit web application (see app/README.md)
├── comparison/                 # Scripts for cross-model evaluation
├── data_preparation/           # Notebooks for raw data processing and tiling
├── evaluation/                 # Notebooks for model validation and testing
├── experiments/                # Training notebooks for all model variants
├── utils/                      # Label conversion, image cutting, dataset utilities
└── yamls/                      # YAML configs for dataset splits
```

### `app/`

The end-to-end web application for processing orthomosaic images. It handles memory-efficient tiled inference, NMS post-processing, and result export. See [`app/README.md`](app/README.md) for full documentation, setup instructions, and how to run it.

### `comparison/`

Python scripts implementing a unified evaluation framework for cross-model comparison. Each model family has a dedicated detector wrapper that shares the same interface:

- `base_detector.py` — abstract base class
- `yolo_detector.py` — YOLOv11 wrapper
- `rt_detr_detector.py` — RT-DETR wrapper
- `rf_detr_detector.py` — RF-DETR wrapper
- `deep_forest_detector.py` — BirdDetector wrapper
- `metrics.py` — mAP, precision, recall, F1, counting error
- `load_dataset.py` — dataset loading utilities
- `visualizations.py` — prediction visualisation helpers

### `data_preparation/`

Notebooks covering raw image processing and dataset construction:

- `data_preparation.ipynb` — full pipeline from raw UAV photos to tiled, annotated datasets
- `uk_penguins_cut.ipynb` — processing of the UK Polar Data Centre external dataset

### `evaluation/`

Notebooks for evaluating trained models:

- `testing_models.ipynb` — evaluation on the primary test set (`uk_test`)
- `validating_models.ipynb` — evaluation on the validation set (`moot`)
- `ood_tuxon.ipynb` — out-of-distribution evaluation on Tuxon Island

### `experiments/`

Training notebooks for each model and dataset configuration. Numbered rounds (01–04) correspond to progressive dataset refinement iterations:

- `01_*` / `02_*` / `03_*` / `04_*` — sequential training experiments per architecture
- `datasets_overview.ipynb` — dataset statistics and composition summary
- `zero_shot_bird_detector.ipynb` — BirdDetector zero-shot baseline evaluation

### `utils/`

Utility notebooks for dataset preparation and tooling:

- `convert_labels_json.ipynb` — converts Label Studio JSON annotations to YOLO/COCO format
- `cut_images.ipynb` — tiles raw images into 1024×1024 px tiles
- `sahi_cut.ipynb` — SAHI-based sliced inference utilities
- `labels_for_unlabeled.ipynb` — model-assisted pre-labelling for unlabeled images
- `yolo_to_coco_dataset.ipynb` — format conversion: YOLO → COCO
- `yolo_to_deep_forest_dataset.ipynb` — format conversion: YOLO → DeepForest
- `log_to_tensorboard.ipynb` — TensorBoard logging utilities

### `yamls/`

YAML files defining dataset splits used across training and evaluation:

- `raw_dataset.yaml` — raw image dataset config
- `cut_dataset.yaml` — yalour_green train, moot validation but only first round of refinement
- `cut_dataset_v2.yaml` — `dataset_v1` (yalour_green train, moot validation)
- `cut_dataset_uk_unlabeled.yaml` — `dataset_v2` (adds uk_train and additional yalour)

---

## Dataset

The dataset was collected in collaboration with the National Antarctic Scientific Center (NASC) of Ukraine using a DJI Phantom 4 Pro (FC6310S sensor). It covers four geographically distinct sites with two penguin species (*Pygoscelis adeliae* and *Pygoscelis papua*):

| Site | Input type | Images | Resolution | Altitude |
|---|---|---|---|---|
| Yalour Island | Raw photos | 807 | 5472 × 3648 px | 80 m |
| Green Island | Raw photos | 65 | 5472 × 3648 px | 100 m |
| Moot Island | Orthomosaic | 1 | 26848 × 21694 px | 100 m |
| Tuxon Island | Orthomosaic | 1 | 20289 × 11681 px | 150 m |

Additional data from the UK Polar Data Centre (Signy Island, South Orkney) were incorporated to improve generalization. All images were manually labeled and tiled into 1024×1024 px tiles with 20% overlap for training and inference.

Annotations were created in Label Studio and iteratively refined through three rounds of model-assisted annotation correction.

---

## Key Results

### Counting accuracy (primary metric)

`YOLO_augmentation` achieves the best colony size estimates, with a counting error of −1.9% on the validation set and −2.0% on the test set.

### Detection performance

`RF-DETR` (baseline) achieves the highest mAP@0.5 of 0.942 on the test set, but consistently overestimates colony size (+9.8% count error), making it more suitable for high-sensitivity detection than precise counting.

### Out-of-distribution generalization

All models degrade significantly on the Tuxon Island dataset (captured at 150 m altitude under strong sunlight). This is an inherent limitation when acquisition conditions differ substantially from training data.

### Processing time

| Image size | Manual | Automated |
|---|---|---|
| 5,472 × 3,648 px | 15–20 min | 15 s |
| 26,848 × 21,694 px | 1 h | 6.5 min |
| 34,645 × 24,179 px | 2 h | 10 min |

---

## Citation

If you use this work, please cite the thesis:

```
Shvahuliak, K. (2026). Developing a Computer Vision System for Penguin Counting
at the Vernadsky Research Base. Bachelor thesis, Ukrainian Catholic University,
Faculty of Applied Sciences.
```
