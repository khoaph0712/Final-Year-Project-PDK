#!/usr/bin/env python
"""Class-agnostic detector recall on the clean TACO field test (F10 honest baseline).

Pipeline recall depends on whether YOLO finds the object at all (the classifier
re-labels the crop), so this matches boxes CLASS-AGNOSTICALLY at IoU>=0.5, greedy,
at a fixed serving operating point. Reports overall + small-box (<1% area) recall.

Run: .venv311\\Scripts\\python.exe scripts/eval_field_recall.py --weights <pt> --tag <name>
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]
IMG_EXT = {".jpg", ".jpeg", ".png"}


def iou_mat(a, b):
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    x1 = np.maximum(a[:, None, 0], b[None, :, 0]); y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2]); y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    aa = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1]); ab = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / (aa[:, None] + ab[None, :] - inter + 1e-9)


def load_gt(lbl, w, h):
    boxes, small = [], []
    if lbl.exists():
        for ln in lbl.read_text().splitlines():
            p = ln.split()
            if len(p) == 5:
                _, cx, cy, bw, bh = (float(v) for v in p)
                boxes.append([(cx-bw/2)*w, (cy-bh/2)*h, (cx+bw/2)*w, (cy+bh/2)*h])
                small.append(bw*bh < 0.01)
    return np.array(boxes, np.float32).reshape(-1, 4), np.array(small, bool)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--dataset", default="taco_field_clean_v1",
                    help="dataset dir under external_datasets/ (class-agnostic match ignores class)")
    ap.add_argument("--split", default="test")
    ap.add_argument("--conf", type=float, default=0.10)
    ap.add_argument("--imgsz", type=int, default=640)
    a = ap.parse_args()
    DS = ROOT / "external_datasets" / a.dataset
    model = YOLO(a.weights)
    img_dir = DS / a.split / "images"
    lbl_dir = DS / a.split / "labels"
    images = sorted(p for p in img_dir.iterdir() if p.suffix.lower() in IMG_EXT)
    tp = gt = stp = sgt = zero_det_imgs = 0
    for i in range(0, len(images), 16):
        batch = images[i:i+16]
        res = model.predict([str(p) for p in batch], conf=a.conf, imgsz=a.imgsz, iou=0.55, max_det=80, verbose=False)
        for p, r in zip(batch, res):
            import cv2
            im = cv2.imread(str(p)); h, w = im.shape[:2]
            g, small = load_gt(lbl_dir / (p.stem + ".txt"), w, h)
            pb = r.boxes.xyxy.cpu().numpy() if r.boxes is not None else np.zeros((0, 4))
            if len(pb) == 0 and len(g) > 0:
                zero_det_imgs += 1
            matched = np.zeros(len(g), bool)
            if len(g) and len(pb):
                m = iou_mat(g, pb); used = np.zeros(len(pb), bool)
                for gi in np.argsort(-m.max(axis=1)):
                    cand = np.where((m[gi] >= 0.5) & ~used)[0]
                    if len(cand):
                        pj = cand[np.argmax(m[gi][cand])]; matched[gi] = True; used[pj] = True
            gt += len(g); tp += int(matched.sum())
            sgt += int(small.sum()); stp += int(matched[small].sum()) if len(g) else 0
    out = {
        "tag": a.tag, "dataset": a.dataset, "split": a.split, "conf": a.conf, "imgsz": a.imgsz,
        "images": len(images), "gt_boxes": gt, "tp": tp,
        "recall": tp / gt if gt else 0.0,
        "small_gt": sgt, "small_tp": stp,
        "small_recall": stp / sgt if sgt else 0.0,
        "zero_detection_images": zero_det_imgs,
    }
    print(json.dumps(out, indent=2))
    dst = ROOT / "runs" / "audits" / f"field_recall_{a.tag}.json"
    json.dump(out, open(dst, "w"), indent=2)


if __name__ == "__main__":
    main()
