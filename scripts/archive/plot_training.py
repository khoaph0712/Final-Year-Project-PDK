"""Plot training curves (loss + mAP) from a YOLOv8 `results.csv`.

Usage:
    python scripts/plot_training.py                                   # uses v3 run under runs/dl/
    python scripts/plot_training.py --csv runs/dl/trash_yolov8n_v3/results.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CSV = ROOT / "runs" / "dl" / "trash_yolov8n_v3" / "results.csv"


def plot(csv_path: Path, out_dir: Path) -> None:
    df = pd.read_csv(csv_path)
    df.columns = [c.strip() for c in df.columns]
    out_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for col, label in [
        ("train/box_loss", "train box"),
        ("train/cls_loss", "train cls"),
        ("train/dfl_loss", "train dfl"),
        ("val/box_loss", "val box"),
        ("val/cls_loss", "val cls"),
        ("val/dfl_loss", "val dfl"),
    ]:
        if col in df.columns:
            axes[0].plot(df["epoch"], df[col], label=label)
    axes[0].set_title("Losses")
    axes[0].set_xlabel("epoch")
    axes[0].set_ylabel("loss")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    for col, label in [
        ("metrics/precision(B)", "precision"),
        ("metrics/recall(B)", "recall"),
        ("metrics/mAP50(B)", "mAP@0.5"),
        ("metrics/mAP50-95(B)", "mAP@0.5:0.95"),
    ]:
        if col in df.columns:
            axes[1].plot(df["epoch"], df[col], label=label)
    axes[1].set_title("Validation metrics")
    axes[1].set_xlabel("epoch")
    axes[1].set_ylabel("value")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    fig.suptitle(f"Training curves — {csv_path.parent.name}")
    fig.tight_layout()
    out = out_dir / "training_curves.png"
    fig.savefig(out, dpi=140)
    print(f"[OK] Saved {out}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    out_dir = args.out or args.csv.parent / "quality_check"
    plot(args.csv, out_dir)


if __name__ == "__main__":
    main()
