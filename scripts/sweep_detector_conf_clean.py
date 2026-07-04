#!/usr/bin/env python
"""Sweep the detector's confidence operating point on the clean val+test split.

Step 1 of the tiny-object plan (tile-augmented training, see F4c) made things
worse, not better - training-data diversity is the real bottleneck and isn't
fixable by retraining again. This is the zero-retrain lever from F5: `conf` sets
the deployed operating point directly, so lowering it trades precision for
recall - specifically the SMALL-box / organic recall the deployed model is
weakest on (see runs/audits/detector_sizeclass_baseline_v2long.json).

For each conf in the grid, reports (on both clean val and clean test):
- overall P / R / mAP50 / mAP50-95 (model.val at that conf, so P/R are the real
  operating-point numbers, not the threshold-independent PR-curve integral)
- organic recall and small-box (<1% image area) recall, via the same greedy
  IoU>=0.5 class-aware matching as eval_detector_sizeclass.py

Run: .venv311\\Scripts\\python.exe scripts/sweep_detector_conf_clean.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval_detector_sizeclass import CLEAN, size_class_recall  # noqa: E402

DEFAULT_WEIGHTS = ROOT / "runs" / "detect" / "yolo26n_hardcase_v2_long" / "weights" / "best.pt"
CONF_GRID = [0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50]
OUT = ROOT / "runs" / "audits"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--imgsz", type=int, default=960, help="960 matches web/server.py YOLO_IMG_SIZE default")
    args = ap.parse_args()

    import torch
    from ultralytics import YOLO

    assert torch.cuda.is_available(), "CUDA required - use .venv311 python"
    assert DEFAULT_WEIGHTS.exists(), f"missing {DEFAULT_WEIGHTS}"
    model = YOLO(str(DEFAULT_WEIGHTS))

    result = {"weights": str(DEFAULT_WEIGHTS), "imgsz": args.imgsz, "splits": {}}
    for split in ("val", "test"):
        rows = []
        for conf in CONF_GRID:
            v = model.val(
                data=str(CLEAN / "data.yaml"), split=split, imgsz=args.imgsz, conf=conf,
                device=0, plots=False, verbose=False,
            )
            breakdown = size_class_recall(model, split, conf, args.imgsz, None)
            row = {
                "conf": conf,
                "precision": float(v.box.mp),
                "recall": float(v.box.mr),
                "map50": float(v.box.map50),
                "map50_95": float(v.box.map),
                "organic_recall": breakdown["per_class_recall"]["organic"],
                "small_box_recall": breakdown["small_box_recall"],
                "large_box_recall": breakdown["large_box_recall"],
            }
            rows.append(row)
            print(
                f"[{split}] conf={conf:<5} P={row['precision']:.3f} R={row['recall']:.3f} "
                f"mAP50={row['map50']:.3f} | organic R={row['organic_recall']:.3f} "
                f"| small-box R={row['small_box_recall']:.3f}"
            )
        result["splits"][split] = rows

    OUT.mkdir(parents=True, exist_ok=True)
    out_path = OUT / f"detector_conf_sweep_baseline_v2long_imgsz{args.imgsz}.json"
    out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"[OK] {out_path}")


if __name__ == "__main__":
    main()
