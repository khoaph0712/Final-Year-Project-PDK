#!/usr/bin/env python
"""Re-render the ResNet50 / MobileNetV2 confusion matrices in the same style as the
original EfficientNetB0 panel (counts, Greens colormap, "Confusion Matrix: X" title)
so the three panels of Fig 6.8 read as one figure.

Runs inference from the saved best.pt checkpoints on the test split only.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
from torch.utils.data import DataLoader

SCRIPTS_DIR = Path(__file__).resolve().parent
sys.path += [str(SCRIPTS_DIR), str(SCRIPTS_DIR / "archive")]
from ml_balanced_training import load_crops_and_balance  # noqa: E402
from train_comparison_models_torch import (  # noqa: E402
    CLASSES, DATA_YAML, DEV, DISPLAY, OUT_DIR, CropDataset, build,
)


def main():
    crops, labels = load_crops_and_balance(DATA_YAML, CLASSES, max_per_class=300,
                                           is_train=False, seed=42)
    loader = DataLoader(CropDataset(crops, labels, False), batch_size=64, shuffle=False)

    for arch in ("resnet50", "mobilenetv2"):
        model = build(arch)
        model.load_state_dict(torch.load(OUT_DIR / arch / "best.pt", map_location=DEV))
        model.eval()
        preds = []
        with torch.no_grad():
            for x, _ in loader:
                preds.append(model(x.to(DEV)).argmax(1).cpu().numpy())
        preds = np.concatenate(preds)

        cm = confusion_matrix(labels, preds, labels=range(len(CLASSES)))
        fig, ax = plt.subplots(figsize=(7, 6))
        ConfusionMatrixDisplay(cm, display_labels=CLASSES).plot(
            ax=ax, cmap="Greens", values_format="d", colorbar=False, xticks_rotation=45)
        ax.set_title(f"Confusion Matrix: {DISPLAY[arch]}", fontweight="bold")
        fig.tight_layout()
        fig.savefig(OUT_DIR / arch / "confusion_matrix.png", dpi=110)
        plt.close(fig)
        print(f"[cm] {arch}: acc={(preds == np.array(labels)).mean():.4f}")


if __name__ == "__main__":
    main()
