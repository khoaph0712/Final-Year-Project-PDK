#!/usr/bin/env python
"""Evaluate a detector on the clean val+test with size- and class-broken-out recall.

Reports, for a given weights file, on BOTH the leakage-quarantined clean val and
clean test splits (the exact splits behind the 0.474 baseline):
- overall P / R / mAP50 / mAP50-95 (via model.val, comparable to validate_detector_clean.py)
- per-class recall (organic highlighted)
- recall on SMALL GT boxes (area < 1% of image) vs the rest

Custom matching is greedy, class-aware, IoU>=0.5, at a fixed conf so recall is
measured at a real operating point (not the PR-curve integral).

Run: .venv311\\Scripts\\python.exe scripts/eval_detector_sizeclass.py --weights <path> --tag <name>
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
CLEAN = ROOT / "external_datasets" / "yolo26_hardcase_clean_eval"
OUT = ROOT / "runs" / "audits"
CLASSES = ["plastic", "glass", "metal", "paper", "cardboard", "organic"]
IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
SMALL_AREA = 0.01


def iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    x1 = np.maximum(a[:, None, 0], b[None, :, 0])
    y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2])
    y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    aa = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    ab = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / (aa[:, None] + ab[None, :] - inter + 1e-9)


def load_gt(lbl: Path, w: int, h: int):
    boxes, cls = [], []
    if lbl.exists():
        for line in lbl.read_text(encoding="utf-8").splitlines():
            p = line.split()
            if len(p) == 5:
                c = int(float(p[0]))
                cx, cy, bw, bh = (float(v) for v in p[1:5])
                boxes.append([(cx - bw / 2) * w, (cy - bh / 2) * h, (cx + bw / 2) * w, (cy + bh / 2) * h])
                cls.append(c)
    return np.array(boxes, np.float32).reshape(-1, 4), np.array(cls, int)


def match(gt, gtc, pb, pc, iou_thr=0.5):
    matched = np.zeros(len(gt), bool)
    if len(gt) and len(pb):
        used = np.zeros(len(pb), bool)
        m = iou_matrix(gt, pb)
        for gi in np.argsort(-m.max(axis=1)):
            cand = np.where((m[gi] >= iou_thr) & ~used & (pc == gtc[gi]))[0]
            if len(cand):
                pj = cand[np.argmax(m[gi][cand])]
                matched[gi] = True
                used[pj] = True
    return matched


def size_class_recall(model, split: str, conf: float, imgsz: int, w_norm):
    img_dir = CLEAN / split / "images"
    lbl_dir = CLEAN / split / "labels"
    images = sorted(p for p in img_dir.iterdir() if p.suffix.lower() in IMG_EXT)
    per_cls_tp = np.zeros(len(CLASSES), int)
    per_cls_gt = np.zeros(len(CLASSES), int)
    small_tp = small_gt = big_tp = big_gt = 0
    for start in range(0, len(images), 32):
        chunk = images[start : start + 32]
        results = model.predict([str(p) for p in chunk], conf=conf, imgsz=imgsz, device=0, verbose=False)
        for p, res in zip(chunk, results):
            im = cv2.imread(str(p))
            if im is None:
                continue
            h, w = im.shape[:2]
            gt, gtc = load_gt(lbl_dir / (p.stem + ".txt"), w, h)
            pb = res.boxes.xyxy.cpu().numpy()
            pc = res.boxes.cls.cpu().numpy().astype(int)
            matched = match(gt, gtc, pb, pc)
            areas = (
                ((gt[:, 2] - gt[:, 0]) * (gt[:, 3] - gt[:, 1])) / (w * h + 1e-9)
                if len(gt)
                else np.zeros(0)
            )
            for i in range(len(gt)):
                per_cls_gt[gtc[i]] += 1
                per_cls_tp[gtc[i]] += int(matched[i])
                if areas[i] < SMALL_AREA:
                    small_gt += 1
                    small_tp += int(matched[i])
                else:
                    big_gt += 1
                    big_tp += int(matched[i])
    return {
        "per_class_recall": {
            CLASSES[i]: round(per_cls_tp[i] / max(per_cls_gt[i], 1), 4) for i in range(len(CLASSES))
        },
        "per_class_gt": {CLASSES[i]: int(per_cls_gt[i]) for i in range(len(CLASSES))},
        "small_box_recall": round(small_tp / max(small_gt, 1), 4),
        "small_box_gt": int(small_gt),
        "large_box_recall": round(big_tp / max(big_gt, 1), 4),
        "large_box_gt": int(big_gt),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--weights", type=Path, required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--conf", type=float, default=0.30, help="operating conf for size/class recall")
    args = ap.parse_args()

    import torch
    from ultralytics import YOLO

    assert torch.cuda.is_available(), "CUDA required - use .venv311 python"
    assert args.weights.exists(), f"missing {args.weights}"
    model = YOLO(str(args.weights))

    result = {"weights": str(args.weights), "imgsz": args.imgsz, "conf": args.conf, "splits": {}}
    for split in ("val", "test"):
        v = model.val(
            data=str(CLEAN / "data.yaml"), split=split, imgsz=args.imgsz, device=0, plots=False, verbose=False
        )
        overall = {
            "precision": float(v.box.mp),
            "recall": float(v.box.mr),
            "map50": float(v.box.map50),
            "map50_95": float(v.box.map),
        }
        breakdown = size_class_recall(model, split, args.conf, args.imgsz, None)
        result["splits"][split] = {"overall": overall, **breakdown}
        print(
            f"[{split}] mAP50={overall['map50']:.3f} R={overall['recall']:.3f} | "
            f"organic R={breakdown['per_class_recall']['organic']:.3f} | "
            f"small-box R={breakdown['small_box_recall']:.3f} (n={breakdown['small_box_gt']})"
        )

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"detector_sizeclass_{args.tag}.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"[OK] {OUT / f'detector_sizeclass_{args.tag}.json'}")


if __name__ == "__main__":
    main()
