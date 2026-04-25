import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
import pandas as pd
from typing import List
from .metrics import EvalResult

COLORS = ["#378ADD", "#1D9E75", "#D85A30", "#D4537E", "#BA7517", "#7F77DD"]


def plot_pr_curves(results: List[EvalResult], figsize=(8, 5)):
    fig, ax = plt.subplots(figsize=figsize)
    for r, c in zip(results, COLORS):
        prec, rec = r.pr_curve
        ax.plot(rec, prec, color=c, lw=2, label=f"{r.name}  mAP={r.map50:.3f}")
    ax.set_xlabel("Recall"); ax.set_ylabel("Precision")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1.02)
    ax.legend(loc="lower left"); ax.grid(alpha=0.3)
    ax.set_title(f"Precision–Recall curves  (IoU={results[0].iou_match})")
    plt.tight_layout(); plt.show()


def plot_comparison_table(results: List[EvalResult]):
    rows = [{
        "Model":            r.name,
        "mAP@50":           f"{r.map50:.4f}",
        "Precision":        f"{r.precision:.4f}",
        "Recall":           f"{r.recall:.4f}",
        "F1":               f"{r.f1:.4f}",
        "Count Err %":      f"{r.count_error_pct:+.1f}%",
        "ms/tile":          f"{r.time_per_image_ms:.1f}",
        "FPS":              f"{r.fps:.1f}",
    } for r in results]

    df = pd.DataFrame(rows).set_index("Model")
    fig, ax = plt.subplots(figsize=(len(df.columns) * 1.4 + 1, len(df) * 0.55 + 1))
    ax.axis("off")
    tbl = ax.table(
        cellText=df.values, colLabels=df.columns,
        rowLabels=df.index, loc="center", cellLoc="center",
    )
    tbl.auto_set_font_size(False); tbl.set_fontsize(10)
    tbl.scale(1, 1.6)

    for j in range(len(df.columns)):
        tbl[(0, j)].set_facecolor("#E6F1FB")
        tbl[(0, j)].set_text_props(weight="bold")

    best = max(range(len(results)), key=lambda i: results[i].map50)
    for j in range(len(df.columns)):
        tbl[(best + 1, j)].set_facecolor("#EAF3DE")

    best_count = min(range(len(results)), key=lambda i: abs(results[i].count_error_pct))
    for j in range(len(df.columns)):
        cell = tbl[(best_count + 1, j)]
        if best_count != best:
            cell.set_facecolor("#FFF8E1")

    best_fps = max(range(len(results)), key=lambda i: results[i].fps)
    for j in range(len(df.columns)):
        if best_fps != best and best_fps != best_count:
            tbl[(best_fps + 1, j)].set_facecolor("#F3E8FF")

    plt.tight_layout(); plt.show()
    return df


def plot_count_comparison(results: List[EvalResult], figsize=(10, 5)):
    """Bar chart: true GT count vs predicted count per model, with error % annotation."""
    names      = [r.name for r in results]
    gt_counts  = [r.gt_count  for r in results]
    pred_counts = [r.pred_count for r in results]

    x   = np.arange(len(names))
    w   = 0.35
    fig, ax = plt.subplots(figsize=figsize)

    bars_gt   = ax.bar(x - w/2, gt_counts,   w, label="Ground Truth", color="#6DB3E8", zorder=3)
    bars_pred = ax.bar(x + w/2, pred_counts, w, label="Predicted",    color="#F4A460", zorder=3)

    for bar, r in zip(bars_pred, results):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + max(gt_counts) * 0.01,
            f"{r.count_error_pct:+.1f}%",
            ha="center", va="bottom", fontsize=9,
            color="green" if r.count_error_pct >= 0 else "red",
        )

    ax.set_xticks(x); ax.set_xticklabels(names, rotation=15, ha="right")
    ax.set_ylabel("Object count")
    ax.set_title("Ground Truth vs Predicted Count")
    ax.legend(); ax.grid(axis="y", alpha=0.3, zorder=0)
    plt.tight_layout(); plt.show()


def plot_speed_comparison(results: List[EvalResult], figsize=(10, 4)):
    """Horizontal bar chart of ms/tile and FPS per model."""
    names = [r.name for r in results]
    ms    = [r.time_per_image_ms for r in results]
    fps   = [r.fps for r in results]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    bars = ax1.barh(names, ms, color=COLORS[:len(names)])
    ax1.set_xlabel("ms / tile")
    ax1.set_title("Inference time per tile")
    ax1.bar_label(bars, fmt="%.1f ms", padding=4, fontsize=9)
    ax1.grid(axis="x", alpha=0.3)

    bars2 = ax2.barh(names, fps, color=COLORS[:len(names)])
    ax2.set_xlabel("FPS")
    ax2.set_title("Throughput (FPS)")
    ax2.bar_label(bars2, fmt="%.1f", padding=4, fontsize=9)
    ax2.grid(axis="x", alpha=0.3)

    plt.tight_layout(); plt.show()


def plot_confusion_matrix(result: EvalResult, figsize=(4, 4)):
    data   = np.array([[result.tp, result.fn], [result.fp, 0]])
    labels = [["TP", "FN"], ["FP", "—"]]
    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(data, cmap="Blues", vmin=0)
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(["Predicted +", "Predicted −"])
    ax.set_yticklabels(["Actual +", "Actual −"])
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{labels[i][j]}\n{data[i,j]}",
                    ha="center", va="center", fontsize=13,
                    color="white" if data[i, j] > data.max() * 0.6 else "black")
    ax.set_title(result.name); plt.colorbar(im, ax=ax, fraction=0.04)
    plt.tight_layout(); plt.show()


def plot_all_confusion_matrices(results: List[EvalResult]):
    n = len(results)
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 4))
    if n == 1: axes = [axes]
    for ax, r in zip(axes, results):
        data   = np.array([[r.tp, r.fn], [r.fp, 0]])
        labels = [["TP", "FN"], ["FP", "—"]]
        im = ax.imshow(data, cmap="Blues", vmin=0)
        ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
        ax.set_xticklabels(["Pred +", "Pred −"])
        ax.set_yticklabels(["Act +", "Act −"])
        for i in range(2):
            for j in range(2):
                ax.text(j, i, f"{labels[i][j]}\n{data[i,j]}",
                        ha="center", va="center", fontsize=12,
                        color="white" if data[i, j] > data.max() * 0.6 else "black")
        ax.set_title(r.name); plt.colorbar(im, ax=ax, fraction=0.04)
    plt.suptitle("Confusion matrices", y=1.02)
    plt.tight_layout(); plt.show()