def compute_metrics(boxes: list, scores: list) -> dict:
    total = len(boxes)
    return {
        "total": total,
        "avg_confidence": sum(scores) / total if total else 0.0,
        "max_confidence": max(scores) if total else 0.0,
    }
