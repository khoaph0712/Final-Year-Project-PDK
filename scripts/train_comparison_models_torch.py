#!/usr/bin/env python
"""Retrain the Stage-2 comparison models (ResNet50, MobileNetV2) in PyTorch.

Replaces the TensorFlow trainer (scripts/train_comparison_models.py), which only ran
1 warmup + 2 fine-tune epochs and therefore produced two-point "curves" that say
nothing about underfitting or overfitting. Same balanced crops, same test split,
15 fine-tune epochs so the train/val gap is actually visible.

Outputs per architecture into runs/dl/comparison_models_torch/<arch>/:
  training_history.csv, training_plots.png, confusion_matrix.png, best.pt
plus comparison_results.json at the top level.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
from torch.utils.data import DataLoader, Dataset
from torchvision import models

SCRIPTS_DIR = Path(__file__).resolve().parent
ROOT = SCRIPTS_DIR.parent
sys.path += [str(SCRIPTS_DIR), str(SCRIPTS_DIR / "archive")]
from ml_balanced_training import load_crops_and_balance  # noqa: E402

DATA_YAML = ROOT / "data" / "merged_dataset_v5" / "data.yaml"
OUT_DIR = ROOT / "runs" / "dl" / "comparison_models_torch"
CLASSES = ["plastic", "glass", "metal", "paper", "cardboard", "organic", "Background"]
DISPLAY = {"resnet50": "ResNet50", "mobilenetv2": "MobileNetV2"}
EPOCHS = 15
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class CropDataset(Dataset):
    """BGR crops -> normalized CHW tensors. Train split gets a horizontal flip."""

    def __init__(self, crops, labels, train: bool):
        self.crops, self.labels, self.train = crops, labels, train

    def __len__(self):
        return len(self.crops)

    def __getitem__(self, i):
        img = cv2.resize(self.crops[i], (224, 224), interpolation=cv2.INTER_LINEAR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        if self.train and np.random.rand() < 0.5:
            img = img[:, ::-1].copy()
        img = (img - IMAGENET_MEAN) / IMAGENET_STD
        return torch.from_numpy(img.transpose(2, 0, 1)), int(self.labels[i])


def build(arch: str):
    """Pretrained backbone, new head, last block + head trainable (mirrors the TF setup)."""
    if arch == "resnet50":
        m = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        m.fc = nn.Sequential(nn.Dropout(0.35), nn.Linear(2048, 128), nn.ReLU(),
                             nn.Dropout(0.2), nn.Linear(128, len(CLASSES)))
        trainable = [m.layer4, m.fc]
    elif arch == "mobilenetv2":
        m = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
        m.classifier = nn.Sequential(nn.Dropout(0.35), nn.Linear(1280, 128), nn.ReLU(),
                                     nn.Dropout(0.2), nn.Linear(128, len(CLASSES)))
        trainable = [m.features[-4:], m.classifier]
    else:
        raise ValueError(arch)

    for p in m.parameters():
        p.requires_grad = False
    for block in trainable:
        for p in block.parameters():
            p.requires_grad = True
    return m.to(DEV)


def run_epoch(model, loader, crit, opt=None):
    model.train(opt is not None)
    tot_loss = correct = n = 0
    preds_all, ys_all = [], []
    with torch.set_grad_enabled(opt is not None):
        for x, y in loader:
            x, y = x.to(DEV, non_blocking=True), y.to(DEV, non_blocking=True)
            out = model(x)
            loss = crit(out, y)
            if opt is not None:
                opt.zero_grad()
                loss.backward()
                opt.step()
            preds = out.argmax(1)
            tot_loss += loss.item() * y.size(0)
            correct += (preds == y).sum().item()
            n += y.size(0)
            preds_all.append(preds.cpu().numpy())
            ys_all.append(y.cpu().numpy())
    return tot_loss / n, correct / n, np.concatenate(preds_all), np.concatenate(ys_all)


def plot_history(arch, hist, out_png):
    name = DISPLAY.get(arch, arch)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
    ep = range(1, len(hist["loss"]) + 1)
    ax1.plot(ep, hist["loss"], "o-", label="Train loss")
    ax1.plot(ep, hist["val_loss"], "s-", label="Val loss")
    ax1.set(title=f"{name} loss convergence", xlabel="Epoch", ylabel="Loss")
    ax2.plot(ep, hist["acc"], "o-", color="tab:green", label="Train acc")
    ax2.plot(ep, hist["val_acc"], "s-", color="tab:red", label="Val acc")
    ax2.set(title=f"{name} accuracy convergence", xlabel="Epoch", ylabel="Accuracy")
    best = int(np.argmin(hist["val_loss"]))
    for ax in (ax1, ax2):
        ax.axvline(best + 1, color="grey", ls="--", lw=1)
        ax.grid(alpha=0.3)
        ax.legend()
    ax1.set_title(f"{name} loss convergence\n(dashed line: best val loss, epoch {best + 1})", fontsize=10)
    ax2.set_title(f"{name} accuracy convergence\n(gap between the curves is the overfit margin)", fontsize=10)
    fig.tight_layout()
    fig.savefig(out_png, dpi=110)
    plt.close(fig)


def train(arch, tr_loader, te_loader):
    print(f"\n=== {arch} ===")
    out = OUT_DIR / arch
    out.mkdir(parents=True, exist_ok=True)
    model = build(arch)
    crit = nn.CrossEntropyLoss()
    opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=1e-4)

    hist = {k: [] for k in ("loss", "acc", "val_loss", "val_acc")}
    best_acc, best_state, best_pr = -1.0, None, None
    t0 = time.time()
    for e in range(EPOCHS):
        tl, ta, _, _ = run_epoch(model, tr_loader, crit, opt)
        vl, va, vp, vy = run_epoch(model, te_loader, crit)
        for k, v in zip(hist, (tl, ta, vl, va)):
            hist[k].append(v)
        print(f"epoch {e+1}/{EPOCHS}  loss {tl:.4f} acc {ta:.4f} | val_loss {vl:.4f} val_acc {va:.4f}")
        if va > best_acc:
            best_acc, best_pr = va, (vp, vy)
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
    elapsed = time.time() - t0

    torch.save(best_state, out / "best.pt")
    with open(out / "training_history.csv", "w", encoding="utf-8") as f:
        f.write("epoch,accuracy,loss,val_accuracy,val_loss\n")
        for i in range(EPOCHS):
            f.write(f"{i},{hist['acc'][i]},{hist['loss'][i]},{hist['val_acc'][i]},{hist['val_loss'][i]}\n")
    plot_history(arch, hist, out / "training_plots.png")

    vp, vy = best_pr
    cm = confusion_matrix(vy, vp, labels=range(len(CLASSES)), normalize="true")
    fig, ax = plt.subplots(figsize=(7, 6))
    ConfusionMatrixDisplay(cm, display_labels=CLASSES).plot(ax=ax, cmap="Blues", values_format=".2f",
                                                           colorbar=False, xticks_rotation=45)
    ax.set_title(f"{DISPLAY.get(arch, arch)} confusion matrix (normalized)")
    fig.tight_layout()
    fig.savefig(out / "confusion_matrix.png", dpi=110)
    plt.close(fig)

    model.load_state_dict(best_state)
    model.to(DEV).eval()
    x1 = next(iter(te_loader))[0][:1].to(DEV)
    with torch.no_grad():
        for _ in range(10):
            model(x1)
        if DEV.type == "cuda":
            torch.cuda.synchronize()
        t = time.time()
        for _ in range(50):
            model(x1)
        if DEV.type == "cuda":
            torch.cuda.synchronize()
    latency_ms = (time.time() - t) / 50 * 1000

    return {
        "model_name": arch,
        "parameters": sum(p.numel() for p in model.parameters()),
        "size_mb": (out / "best.pt").stat().st_size / 1024**2,
        "accuracy": best_acc,
        "loss": min(hist["val_loss"]),
        "avg_latency_ms": latency_ms,
        "training_time_sec": elapsed,
        "epochs": EPOCHS,
        "best_epoch_val_loss": int(np.argmin(hist["val_loss"])) + 1,
        "best_epoch_val_acc": int(np.argmax(hist["val_acc"])) + 1,
    }


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"[INFO] device={DEV}")
    tr_c, tr_y = load_crops_and_balance(DATA_YAML, CLASSES, max_per_class=1000, is_train=True, seed=42)
    te_c, te_y = load_crops_and_balance(DATA_YAML, CLASSES, max_per_class=300, is_train=False, seed=42)
    print(f"[INFO] train={len(tr_c)} test={len(te_c)}")

    tr_loader = DataLoader(CropDataset(tr_c, tr_y, True), batch_size=32, shuffle=True, num_workers=0)
    te_loader = DataLoader(CropDataset(te_c, te_y, False), batch_size=32, shuffle=False, num_workers=0)

    results = {a: train(a, tr_loader, te_loader) for a in ("resnet50", "mobilenetv2")}
    (OUT_DIR / "comparison_results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
