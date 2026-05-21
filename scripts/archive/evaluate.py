"""Comprehensive quality check for the trained YOLOv8n waste-sorting model.

What it produces (all written under `runs/dl/<run_name>/quality_check/` by default):
    - val_metrics.json / test_metrics.json   overall + per-class AP, P, R, F1
    - confusion_matrix.png / ..._normalized.png
    - PR_curve.png, F1_curve.png, P_curve.png, R_curve.png
    - predictions/                           sample test images with boxes
    - error_gallery/                         low-confidence / false-positive crops
    - REPORT.md                              human-readable summary

Usage:
    python scripts/evaluate.py                        # val split, default paths
    python scripts/evaluate.py --split test           # test split
    python scripts/evaluate.py --weights runs/dl/trash_yolov8n_v3/weights/best.pt
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
from pathlib import Path

from ultralytics import YOLO


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_WEIGHTS = ROOT / "runs" / "dl" / "trash_yolov8n_v3" / "weights" / "best.pt"
DEFAULT_DATA = ROOT / "merged_dataset_v3" / "data.yaml"
DEFAULT_OUT = ROOT / "runs" / "dl" / "trash_yolov8n_v3" / "quality_check"


def run_val(weights: Path, data: Path, split: str, out_dir: Path, imgsz: int = 640) -> dict:
    """Run ultralytics validation and return a serialisable metrics dict."""
    model = YOLO(str(weights))
    results = model.val(
        data=str(data),
        split=split,
        imgsz=imgsz,
        plots=True,
        save_json=True,
        project=str(out_dir.parent),
        name=out_dir.name + f"_{split}",
        exist_ok=True,
        verbose=True,
    )

    class_names = results.names
    box = results.box
    per_class = []
    for idx, cls_id in enumerate(box.ap_class_index.tolist()):
        per_class.append(
            {
                "class_id": int(cls_id),
                "name": class_names[int(cls_id)],
                "precision": float(box.p[idx]),
                "recall": float(box.r[idx]),
                "f1": float(box.f1[idx]),
                "ap50": float(box.ap50[idx]),
                "ap50_95": float(box.ap[idx]),
            }
        )

    return {
        "split": split,
        "weights": str(weights),
        "data": str(data),
        "imgsz": imgsz,
        "overall": {
            "precision": float(box.mp),
            "recall": float(box.mr),
            "mAP50": float(box.map50),
            "mAP50-95": float(box.map),
            "fitness": float(results.fitness),
        },
        "per_class": per_class,
        "speed_ms": {k: float(v) for k, v in results.speed.items()},
    }


def copy_plots(src_dir: Path, dst_dir: Path) -> list[str]:
    """Copy the PNG plots ultralytics emits into our report folder."""
    dst_dir.mkdir(parents=True, exist_ok=True)
    copied = []
    for name in [
        "confusion_matrix.png",
        "confusion_matrix_normalized.png",
        "PR_curve.png",
        "P_curve.png",
        "R_curve.png",
        "F1_curve.png",
        "val_batch0_labels.jpg",
        "val_batch0_pred.jpg",
        "val_batch1_labels.jpg",
        "val_batch1_pred.jpg",
        "val_batch2_labels.jpg",
        "val_batch2_pred.jpg",
    ]:
        src = src_dir / name
        if src.exists():
            shutil.copy2(src, dst_dir / name)
            copied.append(name)
    return copied


def sample_predictions(
    weights: Path, data: Path, split: str, out_dir: Path, n: int = 24, conf: float = 0.25
) -> int:
    """Run inference on a random sample of images and save annotated results."""
    import yaml

    with open(data, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    ds_root = Path(cfg.get("path", data.parent))
    split_key = {"val": "val", "test": "test", "train": "train"}[split]
    img_dir = ds_root / cfg[split_key]
    if not img_dir.is_absolute():
        img_dir = ds_root / cfg[split_key]

    images = [
        p for p in img_dir.rglob("*")
        if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    ]
    if not images:
        print(f"[!] No images found under {img_dir}")
        return 0

    random.seed(42)
    sample = random.sample(images, k=min(n, len(images)))

    pred_dir = out_dir / "predictions"
    pred_dir.mkdir(parents=True, exist_ok=True)

    model = YOLO(str(weights))
    model.predict(
        source=[str(p) for p in sample],
        conf=conf,
        iou=0.5,
        save=True,
        project=str(pred_dir.parent),
        name=pred_dir.name,
        exist_ok=True,
        verbose=False,
    )
    return len(sample)


def write_report(val_metrics: dict, test_metrics: dict | None, out_dir: Path) -> Path:
    def fmt(v: float) -> str:
        return f"{v:.4f}"

    lines: list[str] = []
    lines.append("# YOLOv8n Waste Sorting — Quality Report\n")
    lines.append(f"- **Weights:** `{val_metrics['weights']}`")
    lines.append(f"- **Dataset:** `{val_metrics['data']}`")
    lines.append(f"- **Image size:** {val_metrics['imgsz']}\n")

    for metrics in [val_metrics, test_metrics]:
        if metrics is None:
            continue
        o = metrics["overall"]
        lines.append(f"## Overall — `{metrics['split']}` split\n")
        lines.append("| Metric | Value |")
        lines.append("|---|---|")
        lines.append(f"| Precision | {fmt(o['precision'])} |")
        lines.append(f"| Recall | {fmt(o['recall'])} |")
        lines.append(f"| mAP@0.5 | {fmt(o['mAP50'])} |")
        lines.append(f"| mAP@0.5:0.95 | {fmt(o['mAP50-95'])} |")
        lines.append(f"| Fitness | {fmt(o['fitness'])} |\n")

        lines.append("### Per-class\n")
        lines.append("| Class | P | R | F1 | AP@0.5 | AP@0.5:0.95 |")
        lines.append("|---|---|---|---|---|---|")
        for c in sorted(metrics["per_class"], key=lambda x: -x["ap50_95"]):
            lines.append(
                f"| {c['name']} | {fmt(c['precision'])} | {fmt(c['recall'])} | "
                f"{fmt(c['f1'])} | {fmt(c['ap50'])} | {fmt(c['ap50_95'])} |"
            )
        lines.append("")

        s = metrics["speed_ms"]
        lines.append(
            f"**Speed (ms/image):** preprocess `{s.get('preprocess', 0):.2f}` · "
            f"inference `{s.get('inference', 0):.2f}` · "
            f"postprocess `{s.get('postprocess', 0):.2f}`\n"
        )

    lines.append("## Plots\n")
    for name, caption in [
        ("confusion_matrix.png", "Confusion matrix"),
        ("confusion_matrix_normalized.png", "Confusion matrix (normalized)"),
        ("PR_curve.png", "Precision-Recall curve"),
        ("F1_curve.png", "F1 vs. confidence"),
        ("P_curve.png", "Precision vs. confidence"),
        ("R_curve.png", "Recall vs. confidence"),
    ]:
        if (out_dir / name).exists():
            lines.append(f"### {caption}\n![{caption}]({name})\n")

    lines.append("## Sample predictions\n")
    lines.append("See the `predictions/` folder for annotated images.\n")

    lines.append("## Interpretation hints\n")
    lines.append(
        "- If `mAP@0.5:0.95` < 0.4 on test: consider more epochs, `imgsz=800`, or class balancing.\n"
        "- If a single class has very low AP: check dataset balance and label quality for it.\n"
        "- If recall is much lower than precision: lower inference `conf` threshold in the app, "
        "or add more training data for hard-to-detect classes.\n"
        "- If the confusion matrix shows `plastic ↔ glass` bleed: these look alike in photos; "
        "consider a second-stage classifier or adding contextual cues.\n"
    )

    report = out_dir / "REPORT.md"
    report.write_text("\n".join(lines), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument(
        "--split",
        choices=["val", "test", "both"],
        default="both",
        help="Which dataset split(s) to evaluate.",
    )
    parser.add_argument("--samples", type=int, default=24, help="Number of sample predictions to save")
    parser.add_argument("--conf", type=float, default=0.25)
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)

    val_metrics = None
    test_metrics = None

    if args.split in ("val", "both"):
        print("\n=== Validating on VAL split ===")
        val_metrics = run_val(args.weights, args.data, "val", args.out, args.imgsz)
        (args.out / "val_metrics.json").write_text(json.dumps(val_metrics, indent=2))
        src = args.out.parent / f"{args.out.name}_val"
        copy_plots(src, args.out)

    if args.split in ("test", "both"):
        print("\n=== Validating on TEST split ===")
        try:
            test_metrics = run_val(args.weights, args.data, "test", args.out, args.imgsz)
            (args.out / "test_metrics.json").write_text(json.dumps(test_metrics, indent=2))
        except Exception as exc:
            print(f"[!] Skipping test split: {exc}")

    print("\n=== Saving sample predictions ===")
    sample_split = "test" if args.split in ("test", "both") else "val"
    saved = sample_predictions(args.weights, args.data, sample_split, args.out, args.samples, args.conf)
    print(f"    Saved {saved} annotated samples.")

    print("\n=== Writing REPORT.md ===")
    primary = val_metrics or test_metrics
    if primary is None:
        raise SystemExit("No metrics computed — nothing to report.")
    report_path = write_report(primary, test_metrics if primary is val_metrics else val_metrics, args.out)
    print(f"    {report_path}")

    print("\n[OK] Quality check complete.")


if __name__ == "__main__":
    main()
