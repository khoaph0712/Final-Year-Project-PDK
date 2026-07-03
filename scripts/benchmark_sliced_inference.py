#!/usr/bin/env python
"""Benchmark sliced (SAHI-style) inference vs whole-image for tiny objects.

Tiny objects are the detector's main miss source and a 960px retrain was ruled
out. Slicing attacks the same limit at inference time: predict on 2x2
overlapping tiles plus the whole image, map boxes back, merge with NMS.
Measures per-size recall/precision on the leakage-quarantined val split so the
decision to deploy is evidence-based.

Run: .venv311\\Scripts\\python.exe scripts/benchmark_sliced_inference.py
"""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
WEIGHTS = ROOT / "models" / "trained" / "yolov11_detector" / "best.pt"
DATASET = ROOT / "external_datasets" / "yolo26_hardcase_clean_eval"
OUT = ROOT / "runs" / "audits"
IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
TINY_AREA = 0.01  # GT boxes under 1% of image area


def iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    x1 = np.maximum(a[:, None, 0], b[None, :, 0])
    y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2])
    y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / (area_a[:, None] + area_b[None, :] - inter + 1e-9)


def ios_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Intersection over the SMALLER box - merges tile-boundary fragments that
    plain IoU misses (partial box contained in a full box has low IoU)."""
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    x1 = np.maximum(a[:, None, 0], b[None, :, 0])
    y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2])
    y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    smaller = np.minimum(area_a[:, None], area_b[None, :])
    return inter / (smaller + 1e-9)


def nms(boxes: np.ndarray, scores: np.ndarray, thr: float = 0.55, ios_thr: float = 0.6) -> list[int]:
    order = scores.argsort()[::-1]
    keep: list[int] = []
    while len(order):
        i = order[0]
        keep.append(int(i))
        if len(order) == 1:
            break
        rest = order[1:]
        ious = iou_matrix(boxes[i : i + 1], boxes[rest])[0]
        ioss = ios_matrix(boxes[i : i + 1], boxes[rest])[0]
        order = rest[(ious < thr) & (ioss < ios_thr)]
    return keep


def load_gt(lbl: Path, w: int, h: int) -> tuple[np.ndarray, np.ndarray]:
    boxes, classes = [], []
    if lbl.exists():
        for line in lbl.read_text(encoding="utf-8").splitlines():
            p = line.split()
            if len(p) != 5:
                continue
            c = int(float(p[0]))
            cx, cy, bw, bh = (float(v) for v in p[1:5])
            boxes.append([(cx - bw / 2) * w, (cy - bh / 2) * h, (cx + bw / 2) * w, (cy + bh / 2) * h])
            classes.append(c)
    return np.array(boxes, dtype=np.float32).reshape(-1, 4), np.array(classes)


def predict_whole(model, img, conf, imgsz):
    res = model.predict(img, conf=conf, imgsz=imgsz, device=0, verbose=False)[0]
    return (
        res.boxes.xyxy.cpu().numpy(),
        res.boxes.conf.cpu().numpy(),
        res.boxes.cls.cpu().numpy().astype(int),
    )


def predict_sliced(model, img, conf, imgsz, overlap=0.2, slice_conf=None):
    h, w = img.shape[:2]
    tw, th = int(w / 2 * (1 + overlap)), int(h / 2 * (1 + overlap))
    origins = [(0, 0), (w - tw, 0), (0, h - th), (w - tw, h - th)]
    tiles = [img[max(0, y) : y + th, max(0, x) : x + tw] for x, y in origins]
    b0, s0, c0 = predict_whole(model, img, conf, imgsz)
    all_b, all_s, all_c = [b0], [s0], [c0]
    results = model.predict(tiles, conf=slice_conf or conf, imgsz=imgsz, device=0, verbose=False)
    for (ox, oy), res in zip(origins, results):
        b = res.boxes.xyxy.cpu().numpy()
        if len(b):
            b[:, [0, 2]] += max(0, ox)
            b[:, [1, 3]] += max(0, oy)
            all_b.append(b)
            all_s.append(res.boxes.conf.cpu().numpy())
            all_c.append(res.boxes.cls.cpu().numpy().astype(int))
    boxes = np.concatenate(all_b) if all_b else np.zeros((0, 4))
    scores = np.concatenate(all_s) if all_s else np.zeros(0)
    classes = np.concatenate(all_c) if all_c else np.zeros(0, int)
    keep = nms(boxes, scores)
    return boxes[keep], scores[keep], classes[keep]


def match(gt_boxes, gt_cls, pred_boxes, pred_cls, iou_thr=0.5):
    """Greedy class-aware matching; returns matched-GT mask and TP count."""
    matched = np.zeros(len(gt_boxes), bool)
    used = np.zeros(len(pred_boxes), bool)
    if len(gt_boxes) and len(pred_boxes):
        m = iou_matrix(gt_boxes, pred_boxes)
        for gi in np.argsort(-m.max(axis=1)):
            cand = np.where((m[gi] >= iou_thr) & ~used & (pred_cls == gt_cls[gi]))[0]
            if len(cand):
                pj = cand[np.argmax(m[gi][cand])]
                matched[gi] = True
                used[pj] = True
    return matched, used.sum()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--images", type=int, default=800)
    ap.add_argument("--conf", type=float, default=0.30)
    ap.add_argument("--slice-conf", type=float, default=0.45, help="stricter conf for tile detections")
    ap.add_argument("--imgsz", type=int, default=960)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    import torch
    from ultralytics import YOLO

    assert torch.cuda.is_available()
    model = YOLO(str(WEIGHTS))

    img_dir = DATASET / "val" / "images"
    lbl_dir = DATASET / "val" / "labels"
    images = sorted(p for p in img_dir.iterdir() if p.suffix.lower() in IMG_EXT)
    random.Random(args.seed).shuffle(images)
    images = images[: args.images]

    stats = {
        mode: {"tp_tiny": 0, "gt_tiny": 0, "tp_rest": 0, "gt_rest": 0, "preds": 0, "tps": 0, "ms": 0.0}
        for mode in ("whole", "sliced")
    }
    for i, path in enumerate(images):
        img = cv2.imread(str(path))
        if img is None:
            continue
        h, w = img.shape[:2]
        gt_boxes, gt_cls = load_gt(lbl_dir / (path.stem + ".txt"), w, h)
        areas = ((gt_boxes[:, 2] - gt_boxes[:, 0]) * (gt_boxes[:, 3] - gt_boxes[:, 1])) / (w * h + 1e-9)
        tiny = areas < TINY_AREA

        for mode in ("whole", "sliced"):
            t0 = time.time()
            if mode == "whole":
                pb, ps, pc = predict_whole(model, img, args.conf, args.imgsz)
            else:
                pb, ps, pc = predict_sliced(model, img, args.conf, args.imgsz, slice_conf=args.slice_conf)
            stats[mode]["ms"] += (time.time() - t0) * 1000
            matched, tp = match(gt_boxes, gt_cls, pb, pc)
            s = stats[mode]
            s["tp_tiny"] += int(matched[tiny].sum())
            s["gt_tiny"] += int(tiny.sum())
            s["tp_rest"] += int(matched[~tiny].sum())
            s["gt_rest"] += int((~tiny).sum())
            s["preds"] += len(pb)
            s["tps"] += int(tp)
        if i and i % 100 == 0:
            print(f"  {i}/{len(images)}")

    lines = [
        "# Sliced vs whole-image inference (clean val, deployed detector)",
        "",
        f"- {len(images)} images, conf={args.conf}, imgsz={args.imgsz}, tiny = GT box < 1% image area",
        "",
        "| mode | tiny recall | other recall | precision | avg ms/img |",
        "|---|---:|---:|---:|---:|",
    ]
    report = {}
    for mode, s in stats.items():
        tiny_r = s["tp_tiny"] / max(s["gt_tiny"], 1)
        rest_r = s["tp_rest"] / max(s["gt_rest"], 1)
        prec = s["tps"] / max(s["preds"], 1)
        ms = s["ms"] / max(len(images), 1)
        report[mode] = {"tiny_recall": tiny_r, "other_recall": rest_r, "precision": prec, "ms_per_img": ms}
        lines.append(f"| {mode} | {tiny_r:.3f} | {rest_r:.3f} | {prec:.3f} | {ms:.0f} |")
        print(f"[{mode}] tiny recall {tiny_r:.3f} | other {rest_r:.3f} | precision {prec:.3f} | {ms:.0f} ms/img")

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "SLICED_INFERENCE_BENCHMARK.md").write_text("\n".join(lines), encoding="utf-8")
    (OUT / "sliced_inference_benchmark.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"[OK] {OUT / 'SLICED_INFERENCE_BENCHMARK.md'}")


if __name__ == "__main__":
    main()
