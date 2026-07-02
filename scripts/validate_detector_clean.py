#!/usr/bin/env python
"""Validate the retrained detector on original vs leakage-quarantined eval splits.

Produces the detector before/after-leakage table (DETECTOR_CLEAN_VAL.md).
Run with the CUDA env: .venv311\\Scripts\\python.exe scripts/validate_detector_clean.py
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WEIGHTS = ROOT / "runs" / "detect" / "yolo26n_hardcase_v2_long" / "weights" / "best.pt"
DATASETS = {
    "original": ROOT / "external_datasets" / "yolo26_hardcase_dataset_v1" / "data.yaml",
    "clean": ROOT / "external_datasets" / "yolo26_hardcase_clean_eval" / "data.yaml",
}
OUT = ROOT / "runs" / "audits"


def main() -> None:
    import torch
    from ultralytics import YOLO

    assert torch.cuda.is_available(), "CUDA not available - use .venv311 python"
    assert WEIGHTS.exists(), f"weights missing: {WEIGHTS}"

    model = YOLO(str(WEIGHTS))
    rows = []
    for ds_name, data_yaml in DATASETS.items():
        for split in ("val", "test"):
            res = model.val(data=str(data_yaml), split=split, device=0, plots=False, verbose=False)
            rows.append(
                {
                    "eval_set": ds_name,
                    "split": split,
                    "precision": float(res.box.mp),
                    "recall": float(res.box.mr),
                    "map50": float(res.box.map50),
                    "map50_95": float(res.box.map),
                }
            )
            print(
                f"[{ds_name}/{split}] P={res.box.mp:.3f} R={res.box.mr:.3f} "
                f"mAP50={res.box.map50:.3f} mAP50-95={res.box.map:.3f}"
            )

    OUT.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Detector: original vs leakage-quarantined eval (yolo26n_hardcase_v2_long best.pt)",
        "",
        "| eval set | split | Precision | Recall | mAP50 | mAP50-95 |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for r in rows:
        lines.append(
            f"| {r['eval_set']} | {r['split']} | {r['precision']:.3f} | {r['recall']:.3f} | "
            f"{r['map50']:.3f} | {r['map50_95']:.3f} |"
        )
    lines += [
        "",
        "The clean rows exclude eval images that near-duplicate training images "
        "(see runs/audits/quarantine_manifest.csv); they are the honest deployment numbers.",
    ]
    (OUT / "DETECTOR_CLEAN_VAL.md").write_text("\n".join(lines), encoding="utf-8")
    (OUT / "detector_clean_val.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(f"[OK] {OUT / 'DETECTOR_CLEAN_VAL.md'}")


if __name__ == "__main__":
    main()
