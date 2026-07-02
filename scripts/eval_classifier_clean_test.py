#!/usr/bin/env python
"""Evaluate the EfficientNetB0 classifier on original vs leakage-quarantined test sets.

Forces CPU (CUDA_VISIBLE_DEVICES='') so it can run while YOLO trains on the GPU.
Produces the before/after leakage table for the final report.
"""

from __future__ import annotations

import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""  # must be set before TF import

import json
from pathlib import Path

import cv2
import numpy as np

try:
    import tf_keras as keras
except ImportError:
    from tensorflow import keras

ROOT = Path(__file__).resolve().parents[1]
MODEL = ROOT / "models" / "trained" / "efficientnet_classifier" / "best_efficientnet_tuned.h5"
CLASSES = ["plastic", "glass", "metal", "paper", "cardboard", "organic", "Background"]
SETS = {
    "original_test": ROOT / "data" / "merged_dataset_v5" / "test",
    "clean_test": ROOT / "data" / "merged_dataset_v5_clean_test" / "test",
}
OUT = ROOT / "runs" / "audits"
IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def load_set(root: Path) -> tuple[np.ndarray, np.ndarray]:
    xs, ys = [], []
    for ci, cname in enumerate(CLASSES):
        cdir = root / cname
        if not cdir.exists():
            continue
        for p in cdir.iterdir():
            if p.suffix.lower() not in IMG_EXT:
                continue
            img = cv2.imread(str(p))
            if img is None:
                continue
            img = cv2.cvtColor(cv2.resize(img, (224, 224)), cv2.COLOR_BGR2RGB)
            xs.append(img)
            ys.append(ci)
    x = keras.applications.efficientnet.preprocess_input(np.asarray(xs, np.float32))
    return x, np.asarray(ys, np.int64)


def main() -> None:
    from sklearn.metrics import accuracy_score, classification_report, f1_score

    model = keras.models.load_model(str(MODEL))
    results = {}
    for name, root in SETS.items():
        if not root.exists():
            print(f"[WARN] missing {root}")
            continue
        print(f"[INFO] loading {name} from {root}")
        x, y = load_set(root)
        print(f"[INFO] {name}: {len(y)} images; predicting on CPU...")
        preds = np.argmax(model.predict(x, batch_size=64, verbose=1), axis=1)
        acc = accuracy_score(y, preds)
        results[name] = {
            "n": int(len(y)),
            "accuracy": float(acc),
            "f1_macro": float(f1_score(y, preds, average="macro")),
            "report": classification_report(y, preds, target_names=CLASSES, zero_division=0),
        }
        print(f"[RESULT] {name}: accuracy {acc * 100:.2f}%")

    OUT.mkdir(parents=True, exist_ok=True)
    lines = ["# Classifier: original vs leakage-quarantined test set", ""]
    lines += ["| test set | images | accuracy | macro-F1 |", "|---|---:|---:|---:|"]
    for name, r in results.items():
        lines.append(f"| {name} | {r['n']} | {r['accuracy'] * 100:.2f}% | {r['f1_macro']:.4f} |")
    for name, r in results.items():
        lines += ["", f"## {name}", "", "```text", r["report"].rstrip(), "```"]
    (OUT / "CLASSIFIER_CLEAN_TEST_EVAL.md").write_text("\n".join(lines), encoding="utf-8")
    (OUT / "classifier_clean_test_eval.json").write_text(
        json.dumps({k: {kk: vv for kk, vv in v.items() if kk != "report"} for k, v in results.items()}, indent=2),
        encoding="utf-8",
    )
    print(f"[OK] {OUT / 'CLASSIFIER_CLEAN_TEST_EVAL.md'}")


if __name__ == "__main__":
    main()
