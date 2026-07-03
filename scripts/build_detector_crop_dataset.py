#!/usr/bin/env python
"""Build a stage-2 classifier dataset from the DETECTOR's own crops.

The deployed classifier trains on ground-truth crops but serves on YOLO crops
(looser boxes, real-world framing) - a train/serve skew. This script runs the
retrained YOLO26n over the hardcase train/val images, matches each predicted box
to a ground-truth box (IoU >= 0.5), and exports the predicted crop labeled with
the GT class. Unmatched predictions with confidence >= --bg-conf are exported as
Background (they are exactly the false positives the verifier must reject).

Run with the CUDA env:
.venv311\\Scripts\\python.exe scripts/build_detector_crop_dataset.py
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]
WEIGHTS = ROOT / "runs" / "detect" / "yolo26n_hardcase_v2_long" / "weights" / "best.pt"
DATASET = ROOT / "external_datasets" / "yolo26_hardcase_dataset_v1"
OUT = ROOT / "data" / "detector_crops_v1"
CLASSES = ["plastic", "glass", "metal", "paper", "cardboard", "organic"]
IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def iou(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    union = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter
    return inter / union


def load_gt(lbl_path: Path, w: int, h: int) -> list[tuple[int, tuple[float, float, float, float]]]:
    boxes = []
    if not lbl_path.exists():
        return boxes
    for line in lbl_path.read_text(encoding="utf-8").splitlines():
        p = line.split()
        if len(p) != 5:
            continue
        c = int(float(p[0]))
        cx, cy, bw, bh = (float(v) for v in p[1:5])
        x1 = (cx - bw / 2) * w
        y1 = (cy - bh / 2) * h
        boxes.append((c, (x1, y1, x1 + bw * w, y1 + bh * h)))
    return boxes


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--iou-match", type=float, default=0.5)
    ap.add_argument("--bg-conf", type=float, default=0.35, help="min conf for unmatched preds kept as Background")
    ap.add_argument("--max-per-class", type=int, default=6000)
    ap.add_argument("--max-bg-per-split", type=int, default=3000)
    ap.add_argument("--min-crop-px", type=int, default=20)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--splits", nargs="+", default=["train", "val"])
    args = ap.parse_args()

    import torch
    from ultralytics import YOLO

    assert torch.cuda.is_available(), "CUDA not available - use .venv311 python"
    model = YOLO(str(WEIGHTS))
    rng = random.Random(args.seed)

    summary: dict[str, dict] = {}
    for split in args.splits:
        img_dir = DATASET / split / "images"
        lbl_dir = DATASET / split / "labels"
        images = sorted(p for p in img_dir.iterdir() if p.suffix.lower() in IMG_EXT)
        rng.shuffle(images)

        counts: Counter = Counter()
        out_split = OUT / split
        for cname in CLASSES + ["Background"]:
            (out_split / cname).mkdir(parents=True, exist_ok=True)

        print(f"[INFO] {split}: {len(images)} images")
        batch = 32
        for start in range(0, len(images), batch):
            chunk = images[start : start + batch]
            results = model.predict(
                [str(p) for p in chunk], conf=args.conf, imgsz=960, device=0, verbose=False
            )
            for img_path, res in zip(chunk, results):
                img = cv2.imread(str(img_path))
                if img is None:
                    continue
                h, w = img.shape[:2]
                gt = load_gt(lbl_dir / (img_path.stem + ".txt"), w, h)
                for box, conf_t, cls_t in zip(
                    res.boxes.xyxy.cpu().numpy(),
                    res.boxes.conf.cpu().numpy(),
                    res.boxes.cls.cpu().numpy(),
                ):
                    x1, y1, x2, y2 = (float(v) for v in box)
                    if (x2 - x1) < args.min_crop_px or (y2 - y1) < args.min_crop_px:
                        continue
                    best_iou, best_cls = 0.0, None
                    for gc, gb in gt:
                        v = iou((x1, y1, x2, y2), gb)
                        if v > best_iou:
                            best_iou, best_cls = v, gc
                    if best_iou >= args.iou_match and best_cls is not None:
                        label = CLASSES[best_cls]
                    elif float(conf_t) >= args.bg_conf and best_iou < 0.2:
                        label = "Background"
                    else:
                        continue
                    cap = args.max_bg_per_split if label == "Background" else args.max_per_class
                    if counts[label] >= cap:
                        continue
                    crop = img[int(max(0, y1)) : int(min(h, y2)), int(max(0, x1)) : int(min(w, x2))]
                    if crop.size == 0:
                        continue
                    name = f"{img_path.stem}_{counts[label]:05d}.jpg"
                    cv2.imwrite(str(out_split / label / name), crop, [cv2.IMWRITE_JPEG_QUALITY, 92])
                    counts[label] += 1
            if start and start % (batch * 40) == 0:
                print(f"  ... {start}/{len(images)} images | {dict(counts)}")

        summary[split] = dict(counts)
        print(f"[OK] {split}: {dict(counts)}")

    (OUT / "build_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"[OK] Detector-crop dataset at {OUT}")


if __name__ == "__main__":
    main()
