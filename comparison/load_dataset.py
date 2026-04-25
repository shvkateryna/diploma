from pathlib import Path
import numpy as np
import cv2

def load_dataset(image_dir: str, label_dir: str):
    """
    Returns list of (image_path, gt_boxes_xyxy) for every image that has a label.
    Labels must be YOLO format .txt files.
    """
    samples = []
    for img_path in sorted(Path(image_dir).glob("*.[jp][pn]g")):
        label_path = Path(label_dir) / (img_path.stem + ".txt")
        if not label_path.exists():
            continue
        img = cv2.imread(str(img_path))
        H, W = img.shape[:2]
        boxes = []
        for line in label_path.read_text().splitlines():
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            _, xc, yc, bw, bh = map(float, parts)
            x1, y1 = (xc - bw/2)*W, (yc - bh/2)*H
            x2, y2 = (xc + bw/2)*W, (yc + bh/2)*H
            boxes.append([x1, y1, x2, y2])
        samples.append((str(img_path), np.array(boxes, dtype=float)))
    return samples