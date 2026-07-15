#!/usr/bin/env python
"""Re-sweep detector conf/gate/alpha for yolo26m, matching the yolo26s_conf_gate_alpha_sweep.json methodology.

Field = taco_field_clean_v1/test (rebuilt via build_taco_field_clean.py, deterministic
from external_datasets/hard_case_full/taco_official - unseen-photo TACO test set).
Studio = yolo26_hardcase_clean_eval/val (leakage-quarantined studio-domain clean split).

Matching: class-agnostic IoU>=0.5, greedy, imgsz 640 - identical to the yolo26s sweep.

Run: .venv311\\Scripts\\python.exe scripts/sweep_detector_conf_gate_yolo26m.py
"""
from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]
WEIGHTS = Path(r"C:\kaggle\working\runs\yolo26m_hardcase_v1\weights\last.pt")
FIELD_DIR = ROOT / "external_datasets" / "taco_field_clean_v1" / "test"
STUDIO_DIR = ROOT / "external_datasets" / "yolo26_hardcase_clean_eval" / "val"
CLASSES = ["plastic", "glass", "metal", "paper", "cardboard", "organic"]
IMG_EXT = {".jpg", ".jpeg", ".png"}
IMGSZ = 640
CONF_GRID = [0.10, 0.07, 0.05, 0.04, 0.03, 0.02]
GATE_GRID = [0.20, 0.25, 0.30, 0.35, 0.40, 0.50]
OUT = ROOT / "runs" / "audits" / "yolo26m_conf_gate_alpha_sweep.json"


def iou_mat(a, b):
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    x1 = np.maximum(a[:, None, 0], b[None, :, 0]); y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2]); y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    aa = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1]); ab = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / (aa[:, None] + ab[None, :] - inter + 1e-9)


def load_gt(lbl, w, h):
    boxes, cls, small = [], [], []
    if lbl.exists():
        for ln in lbl.read_text(encoding="utf-8").splitlines():
            p = ln.split()
            if len(p) == 5:
                c = int(float(p[0]))
                cx, cy, bw, bh = (float(v) for v in p[1:5])
                boxes.append([(cx - bw / 2) * w, (cy - bh / 2) * h, (cx + bw / 2) * w, (cy + bh / 2) * h])
                cls.append(c)
                small.append(bw * bh < 0.01)
    return (np.array(boxes, np.float32).reshape(-1, 4), np.array(cls, int), np.array(small, bool))


def eval_domain(model, img_dir: Path, lbl_dir: Path, conf: float, need_class_bias: bool = False):
    images = sorted(p for p in img_dir.iterdir() if p.suffix.lower() in IMG_EXT)
    tp = gt = pred_total = pred_correct = stp = sgt = 0
    class_pred_count = {c: 0 for c in CLASSES}
    class_correct = {c: 0 for c in CLASSES}
    for i in range(0, len(images), 16):
        batch = images[i : i + 16]
        res = model.predict(
            [str(p) for p in batch], conf=conf, imgsz=IMGSZ, iou=0.55, max_det=80, device=0, verbose=False
        )
        for p, r in zip(batch, res):
            im = cv2.imread(str(p))
            if im is None:
                continue
            h, w = im.shape[:2]
            g, gc, small = load_gt(lbl_dir / (p.stem + ".txt"), w, h)
            pb = r.boxes.xyxy.cpu().numpy() if r.boxes is not None else np.zeros((0, 4))
            pc = r.boxes.cls.cpu().numpy().astype(int) if r.boxes is not None else np.zeros(0, int)
            matched = np.zeros(len(g), bool)
            matched_class_ok = np.zeros(len(pb), bool)
            if len(g) and len(pb):
                m = iou_mat(g, pb)
                used = np.zeros(len(pb), bool)
                for gi in np.argsort(-m.max(axis=1)):
                    cand = np.where((m[gi] >= 0.5) & ~used)[0]
                    if len(cand):
                        pj = cand[np.argmax(m[gi][cand])]
                        matched[gi] = True
                        used[pj] = True
                        if need_class_bias and pc[pj] == gc[gi]:
                            matched_class_ok[pj] = True
            gt += len(g)
            tp += int(matched.sum())
            sgt += int(small.sum())
            stp += int(matched[small].sum()) if len(g) else 0
            pred_total += len(pb)
            pred_correct += int(matched.sum())
            if need_class_bias:
                for j, c in enumerate(pc):
                    if 0 <= c < len(CLASSES):
                        class_pred_count[CLASSES[c]] += 1
                        if matched_class_ok[j]:
                            class_correct[CLASSES[c]] += 1
    out = {
        "images": len(images),
        "gt_boxes": gt,
        "tp": tp,
        "recall": tp / gt if gt else 0.0,
        "small_gt": sgt,
        "small_tp": stp,
        "small_recall": stp / sgt if sgt else 0.0,
        "pred_total": pred_total,
        "pred_correct": pred_correct,
        "precision": pred_correct / pred_total if pred_total else 0.0,
    }
    if need_class_bias:
        out["class_pred_count"] = class_pred_count
        out["class_precision"] = {
            c: (class_correct[c] / class_pred_count[c] if class_pred_count[c] else None) for c in CLASSES
        }
    return out


def main() -> None:
    import torch

    assert torch.cuda.is_available(), "CUDA required - use .venv311 python"
    assert WEIGHTS.exists(), f"missing {WEIGHTS}"
    model = YOLO(str(WEIGHTS))

    result = {"weights": str(WEIGHTS), "conf_grid": {}, "gate_grid": {}}

    for conf in CONF_GRID:
        field = eval_domain(model, FIELD_DIR / "images", FIELD_DIR / "labels", conf)
        studio = eval_domain(model, STUDIO_DIR / "images", STUDIO_DIR / "labels", conf)
        result["conf_grid"][str(conf)] = {"field": field, "studio": studio}
        print(f"conf={conf:<5} field R={field['recall']:.3f} P={field['precision']:.3f} | "
              f"studio R={studio['recall']:.3f} P={studio['precision']:.3f}")

    for gate in GATE_GRID:
        field = eval_domain(model, FIELD_DIR / "images", FIELD_DIR / "labels", gate)
        studio = eval_domain(model, STUDIO_DIR / "images", STUDIO_DIR / "labels", gate)
        result["gate_grid"][str(gate)] = {"field": field, "studio": studio}
        print(f"gate={gate:<5} field R={field['recall']:.3f} P={field['precision']:.3f} | "
              f"studio R={studio['recall']:.3f} P={studio['precision']:.3f}")

    bias = eval_domain(model, FIELD_DIR / "images", FIELD_DIR / "labels", 0.04, need_class_bias=True)
    result["field_class_bias_conf004"] = bias

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"[OK] {OUT}")


if __name__ == "__main__":
    main()
