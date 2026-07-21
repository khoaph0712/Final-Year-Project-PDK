#!/usr/bin/env python
"""Replot the retrained comparison-model curves, copy them into the web assets, and
rebuild the side-by-side confusion-matrix grid (Fig 6.8).

Replotting reads training_history.csv, so figure styling can be changed without retraining.
The EfficientNetB0 panel is reused from the original TensorFlow run: that model was not
retrained, and it was evaluated on the same balanced test split (same loader, same seed).
"""

import csv
import sys
from pathlib import Path

from PIL import Image

sys.path.append(str(Path(__file__).resolve().parent))
from train_comparison_models_torch import plot_history  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "runs" / "dl" / "comparison_models_torch"
WEB = ROOT / "web" / "assets" / "figures"

COPIES = {
    SRC / "resnet50" / "training_plots.png": WEB / "train_resnet50.png",
    SRC / "mobilenetv2" / "training_plots.png": WEB / "train_mobilenetv2.png",
}
GRID = [
    SRC / "mobilenetv2" / "confusion_matrix.png",
    SRC / "resnet50" / "confusion_matrix.png",
    ROOT / "runs" / "dl" / "comparison_models" / "confusion_efficientnetb0_ours.png",
]


def replot(arch):
    rows = list(csv.DictReader((SRC / arch / "training_history.csv").open(encoding="utf-8")))
    hist = {k: [float(r[c]) for r in rows]
            for k, c in (("loss", "loss"), ("acc", "accuracy"),
                         ("val_loss", "val_loss"), ("val_acc", "val_accuracy"))}
    plot_history(arch, hist, SRC / arch / "training_plots.png")


def main():
    for arch in ("resnet50", "mobilenetv2"):
        replot(arch)

    for src, dst in COPIES.items():
        dst.write_bytes(src.read_bytes())
        print(f"[copy] {src.name} -> {dst.relative_to(ROOT)}")

    panels = [Image.open(p).convert("RGB") for p in GRID]
    h = min(p.height for p in panels)
    panels = [p.resize((round(p.width * h / p.height), h), Image.LANCZOS) for p in panels]
    grid = Image.new("RGB", (sum(p.width for p in panels), h), "white")
    x = 0
    for p in panels:
        grid.paste(p, (x, 0))
        x += p.width
    grid.save(WEB / "cm_grid_comparison.png")
    print(f"[grid] {grid.size} -> {(WEB / 'cm_grid_comparison.png').relative_to(ROOT)}")


if __name__ == "__main__":
    main()
