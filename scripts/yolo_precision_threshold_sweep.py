#!/usr/bin/env python
"""Raise detector precision WITHOUT retraining by sweeping the confidence operating point.

mAP integrates over the whole PR curve, so it barely moves with `conf`. Precision and
recall, however, are read at a single operating point: raising `conf` drops false
positives (precision up) at the cost of recall. This script runs `model.val()` across a
grid of confidence thresholds, tabulates the P/R/F1 trade-off, and recommends the lowest
`conf` that reaches a target precision so recall is sacrificed as little as possible.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_WEIGHTS = ROOT / "runs" / "detect" / "yolo26n_hardcase_dataset_v1" / "weights" / "best.pt"
DEFAULT_DATA = ROOT / "external_datasets" / "yolo26_hardcase_dataset_v1" / "data.yaml"
DEFAULT_OUT = ROOT / "runs" / "detect" / "precision_threshold_sweep"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS)
    p.add_argument("--data", type=Path, default=DEFAULT_DATA)
    p.add_argument("--out", type=Path, default=DEFAULT_OUT)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--iou", type=float, default=0.7, help="NMS IoU threshold")
    p.add_argument("--split", default="val")
    p.add_argument(
        "--conf-grid",
        type=float,
        nargs="+",
        default=[0.001, 0.10, 0.20, 0.25, 0.30, 0.40, 0.50, 0.60, 0.70],
    )
    p.add_argument("--target-precision", type=float, default=0.85)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    import torch
    from ultralytics import YOLO

    if not args.weights.exists():
        raise SystemExit(f"[ERROR] weights not found: {args.weights}")
    device = 0 if torch.cuda.is_available() else "cpu"
    print(f"[INFO] Device: {device} | weights: {args.weights.name}")

    model = YOLO(str(args.weights))
    rows: list[dict] = []
    for conf in args.conf_grid:
        res = model.val(
            data=str(args.data),
            imgsz=args.imgsz,
            conf=conf,
            iou=args.iou,
            split=args.split,
            device=device,
            plots=False,
            verbose=False,
        )
        p = float(res.box.mp)
        r = float(res.box.mr)
        f1 = (2 * p * r / (p + r)) if (p + r) > 0 else 0.0
        rows.append(
            {
                "conf": conf,
                "precision": p,
                "recall": r,
                "f1": f1,
                "map50": float(res.box.map50),
                "map50_95": float(res.box.map),
            }
        )
        print(f"  conf={conf:<5} P={p:.3f} R={r:.3f} F1={f1:.3f} mAP50={res.box.map50:.3f}")

    reaching = [row for row in rows if row["precision"] >= args.target_precision]
    best_prec = min(reaching, key=lambda x: x["conf"]) if reaching else None
    best_f1 = max(rows, key=lambda x: x["f1"])

    lines = [
        "# YOLO Precision Threshold Sweep (no retrain)",
        "",
        f"- Weights: `{args.weights.relative_to(ROOT)}`",
        f"- Data: `{args.data.relative_to(ROOT)}` (split={args.split}), NMS IoU={args.iou}",
        f"- Target precision: {args.target_precision:.2f}",
        "",
        "| conf | Precision | Recall | F1 | mAP50 | mAP50-95 |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['conf']} | {row['precision']:.3f} | {row['recall']:.3f} | "
            f"{row['f1']:.3f} | {row['map50']:.3f} | {row['map50_95']:.3f} |"
        )
    lines.append("")
    lines.append(f"- **Best F1 operating point:** conf={best_f1['conf']} "
                 f"(P={best_f1['precision']:.3f}, R={best_f1['recall']:.3f}).")
    if best_prec:
        lines.append(
            f"- **Lowest conf reaching precision >= {args.target_precision:.2f}:** conf={best_prec['conf']} "
            f"(P={best_prec['precision']:.3f}, R={best_prec['recall']:.3f}) - use this to raise precision "
            f"with minimal recall loss."
        )
    else:
        lines.append(
            f"- No conf in the grid reaches precision >= {args.target_precision:.2f}; "
            f"precision ceiling needs a retrain, not a threshold change."
        )
    (args.out / "precision_threshold_sweep.md").write_text("\n".join(lines), encoding="utf-8")
    (args.out / "precision_threshold_sweep.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(f"[SUCCESS] Report: {args.out / 'precision_threshold_sweep.md'}")


if __name__ == "__main__":
    main()
