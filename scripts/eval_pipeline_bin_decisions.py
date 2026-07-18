#!/usr/bin/env python
"""Full-pipeline (server.predict_image) eval: material and bin-routing accuracy.

Ground truth: external_datasets/rf_garbage_cls/test (Roboflow "garbage-classification-3",
YOLO-format bboxes, 6 classes BIODEGRADABLE/CARDBOARD/GLASS/METAL/PAPER/PLASTIC). Not used
in training our detector or classifier, so it's an independent check on the fusion and
material->bin routing logic in web/server.py rather than the detector's own conf/gate sweep set.

The S6 waste-state decision was removed 2026-07-18 (no dataset to train it), so this no longer
measures waste-state accuracy - just whether the dominant material and its bin route are right.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "web"))

import server  # noqa: E402

DATA_DIR = ROOT / "external_datasets" / "rf_garbage_cls" / "test"
GT_NAMES = ["organic", "cardboard", "glass", "metal", "paper", "plastic"]  # index-matched to data.yaml
OUT = ROOT / "runs" / "audits"
TAG = sys.argv[1] if len(sys.argv) > 1 else "baseline"


def load_gt(label_path: Path) -> list[str]:
    if not label_path.exists():
        return []
    classes = []
    for line in label_path.read_text().strip().splitlines():
        if not line.strip():
            continue
        idx = int(line.split()[0])
        classes.append(GT_NAMES[idx])
    return classes


def main() -> None:
    images = sorted((DATA_DIR / "images").glob("*.*"))
    n_total = 0
    material_correct = 0
    bin_correct = 0
    secondary_hits = 0
    secondary_total = 0
    confusion: dict[str, dict[str, int]] = {}
    mixed_flagged_on_single_gt = 0
    mixed_flagged_on_mixed_gt = 0
    n_single_gt = 0
    n_mixed_gt = 0

    for i, img_path in enumerate(images):
        label_path = DATA_DIR / "labels" / (img_path.stem + ".txt")
        gt_classes = load_gt(label_path)
        if not gt_classes:
            continue

        gt_counts: dict[str, int] = {}
        for c in gt_classes:
            gt_counts[c] = gt_counts.get(c, 0) + 1
        gt_dominant = max(gt_counts, key=gt_counts.get)
        gt_distinct = set(gt_classes)

        result = server.predict_image(img_path.read_bytes())
        n_total += 1

        pred_material = result["className"].lower()
        pred_bin = result["bin"]
        gt_bin = server.ROUTES.get(gt_dominant, "Recycling")

        confusion.setdefault(gt_dominant, {})
        confusion[gt_dominant][pred_material] = confusion[gt_dominant].get(pred_material, 0) + 1

        if pred_material == gt_dominant:
            material_correct += 1
        if pred_bin == gt_bin:
            bin_correct += 1

        mixed_flagged = bool(result["model"].get("mixedLoadDetected"))
        if len(gt_distinct) > 1:
            n_mixed_gt += 1
            if mixed_flagged:
                mixed_flagged_on_mixed_gt += 1
            materials_present = {m.lower() for m in result["model"].get("materialsPresent", [])}
            for c in gt_distinct:
                secondary_total += 1
                if c in materials_present:
                    secondary_hits += 1
        else:
            n_single_gt += 1
            if mixed_flagged:
                mixed_flagged_on_single_gt += 1

        if (i + 1) % 100 == 0:
            print(f"[{i + 1}/{len(images)}] material_acc={material_correct / n_total:.3f}", flush=True)

    report = {
        "tag": TAG,
        "n": n_total,
        "materialAccuracy": material_correct / n_total,
        "binAccuracy": bin_correct / n_total,
        "secondaryMaterialRecall": (secondary_hits / secondary_total) if secondary_total else None,
        "secondaryMaterialN": secondary_total,
        "mixedFlagFalsePositiveRate": (mixed_flagged_on_single_gt / n_single_gt) if n_single_gt else None,
        "mixedFlagTruePositiveRate": (mixed_flagged_on_mixed_gt / n_mixed_gt) if n_mixed_gt else None,
        "confusion": confusion,
    }

    OUT.mkdir(parents=True, exist_ok=True)
    out_path = OUT / f"pipeline_bin_decision_eval_{TAG}.json"
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"\n[RESULT] tag={TAG} n={n_total}")
    print(f"  material accuracy:        {report['materialAccuracy']:.3f}")
    print(f"  bin accuracy:             {report['binAccuracy']:.3f}")
    if secondary_total:
        print(f"  secondary-material recall: {report['secondaryMaterialRecall']:.3f} (n={secondary_total} across {n_mixed_gt} mixed-class images)")
    print(f"  mixed-load flag: TP rate {report['mixedFlagTruePositiveRate']:.3f} (n={n_mixed_gt}), FP rate {report['mixedFlagFalsePositiveRate']:.3f} (n={n_single_gt})")
    print(f"[OK] {out_path}")


if __name__ == "__main__":
    main()
