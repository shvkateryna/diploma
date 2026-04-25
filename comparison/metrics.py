from dataclasses import dataclass, field
from typing import List
import numpy as np
import torch
import time
from torchvision.ops import nms, box_iou
from .base_detector import BaseDetector
from .load_dataset import load_dataset

@dataclass
class EvalResult:
    name: str
    map50: float
    precision: float
    recall: float
    f1: float
    tp: int; fp: int; fn: int
    pr_curve: tuple
    conf: float
    iou_match: float
    pred_count: int = 0
    gt_count: int = 0
    #time
    inference_time_s: float = 0.0
    time_per_image_ms: float = 0.0
    fps: float = 0.0
    #count quality
    count_error: int = 0
    count_error_pct: float = 0.0
    count_abs_error_pct: float = 0.0


def evaluate_model(
    detector: BaseDetector,
    samples: list,
    conf: float = 0.2,
    iou_nms: float = 0.2,
    iou_match: float = 0.5,
) -> EvalResult:
    all_scores, all_tp, all_fp = [], [], []
    total_gt = 0
    total_inference_s = 0.0
    n_images = len(samples)

    for img_path, gt_np in samples:
        t0 = time.perf_counter()
        det = detector.predict(img_path, conf)
        total_inference_s += time.perf_counter() - t0

        gt = torch.tensor(gt_np, dtype=torch.float32)
        if len(det.boxes) == 0:
            total_gt += len(gt)
            continue

        pred = torch.tensor(det.boxes, dtype=torch.float32)
        scores = torch.tensor(det.scores, dtype=torch.float32)

        keep = nms(pred, scores, iou_nms)
        pred, scores = pred[keep], scores[keep]

        if len(gt) == 0:
            all_scores.extend(scores.tolist())
            all_tp.extend([0]*len(scores))
            all_fp.extend([1]*len(scores))
            continue

        iou = box_iou(pred, gt).numpy()
        order = np.argsort(-scores.numpy())
        matched = set()
        for pi in order:
            row = iou[pi].copy()
            row[list(matched)] = 0
            gi = row.argmax()
            if row[gi] >= iou_match:
                all_tp.append(1); all_fp.append(0)
                matched.add(int(gi))
            else:
                all_tp.append(0); all_fp.append(1)
            all_scores.append(float(scores[pi]))
        total_gt += len(gt)

    order = np.argsort(-np.array(all_scores))
    tp_arr = np.array(all_tp)[order]
    fp_arr = np.array(all_fp)[order]

    cum_tp = np.cumsum(tp_arr)
    cum_fp = np.cumsum(fp_arr)
    prec = cum_tp / (cum_tp + cum_fp + 1e-9)
    rec  = cum_tp / (total_gt + 1e-9)

    prec_env = np.concatenate([[1], prec, [0]])
    rec_env  = np.concatenate([[0], rec,  [1]])
    for i in range(len(prec_env)-2, -1, -1):
        prec_env[i] = max(prec_env[i], prec_env[i+1])
    map50 = float(np.sum((rec_env[1:] - rec_env[:-1]) * prec_env[1:]))

    tp = int(tp_arr.sum())
    fp = int(fp_arr.sum())
    fn = total_gt - tp
    p  = tp / (tp + fp + 1e-9)
    r  = tp / (tp + fn + 1e-9)
    f1 = 2*p*r / (p + r + 1e-9)

    time_per_image_ms = (total_inference_s / n_images * 1000) if n_images else 0.0
    fps = (n_images / total_inference_s) if total_inference_s > 0 else 0.0

    return EvalResult(
        name=detector.name, map50=map50,
        precision=p, recall=r, f1=f1,
        tp=tp, fp=fp, fn=fn,
        pr_curve=(prec, rec),
        conf=conf, iou_match=iou_match,
        pred_count=tp + fp,
        gt_count=total_gt,
        inference_time_s=total_inference_s,
        time_per_image_ms=time_per_image_ms,
        fps=fps,
        count_error=0,
        count_error_pct=0.0,
        count_abs_error_pct=0.0,
    )


def count_ground_truth(samples: list, iou_nms: float = 0.2) -> int:
    import re
    TILE_PATTERN = re.compile(r"_(\d+)_(\d+)_(\d+)_(\d+)\.")
    all_boxes = []

    for img_path, gt_np in samples:
        if len(gt_np) == 0:
            continue
        m = TILE_PATTERN.search(str(img_path))
        if not m:
            all_boxes.extend(gt_np.tolist())
            continue
        tx1, ty1, tx2, ty2 = map(int, m.groups())
        for x1, y1, x2, y2 in gt_np:
            all_boxes.append([x1 + tx1, y1 + ty1, x2 + tx1, y2 + ty1])

    if not all_boxes:
        return 0
    boxes = torch.tensor(all_boxes, dtype=torch.float32)
    keep  = nms(boxes, torch.ones(len(boxes)), iou_nms)
    return int(len(keep))


def count_predictions(
    detector: BaseDetector,
    samples: list,
    conf: float = 0.2,
    iou_nms: float = 0.2,
) -> int:
    import re
    TILE_PATTERN = re.compile(r"_(\d+)_(\d+)_(\d+)_(\d+)\.")
    all_boxes, all_scores = [], []

    for img_path, _ in samples:
        det = detector.predict(img_path, conf)
        if len(det.boxes) == 0:
            continue
        m = TILE_PATTERN.search(str(img_path))
        if not m:
            all_boxes.extend(det.boxes)
            all_scores.extend(det.scores)
            continue
        tx1, ty1 = int(m.group(1)), int(m.group(2))
        for (x1, y1, x2, y2), score in zip(det.boxes, det.scores):
            all_boxes.append([x1 + tx1, y1 + ty1, x2 + tx1, y2 + ty1])
            all_scores.append(float(score))

    if not all_boxes:
        return 0
    boxes  = torch.tensor(all_boxes,  dtype=torch.float32)
    scores = torch.tensor(all_scores, dtype=torch.float32)
    keep   = nms(boxes, scores, iou_nms)
    return int(len(keep))


def attach_count_metrics(result: EvalResult, true_gt: int, true_pred: int) -> EvalResult:
    """
    Call this after count_ground_truth + count_predictions to fill in
    the count comparison fields on an existing EvalResult.
    """
    result.count_error         = true_pred - true_gt
    result.count_error_pct     = (true_pred - true_gt) / (true_gt + 1e-9) * 100
    result.count_abs_error_pct = abs(true_pred - true_gt) / (true_gt + 1e-9) * 100
    return result