"""Lightweight CNN baseline on object crops for ML-vs-DL comparison."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, confusion_matrix, f1_score
from torch.utils.data import DataLoader, TensorDataset
import yaml


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATA = ROOT / "merged_dataset_v3" / "data.yaml"
DEFAULT_OUT = ROOT / "runs" / "dl_baseline"


def resolve_split_images(ds_root: Path, split_rel: str) -> Path:
    p = Path(split_rel)
    if p.is_absolute():
        return p
    return ds_root / p


def image_to_label_path(image_path: Path) -> Path:
    return Path(str(image_path).replace("\\images\\", "\\labels\\")).with_suffix(".txt")


def clamp_box(x1: int, y1: int, x2: int, y2: int, w: int, h: int) -> tuple[int, int, int, int]:
    x1 = max(0, min(x1, w - 1))
    y1 = max(0, min(y1, h - 1))
    x2 = max(x1 + 1, min(x2, w))
    y2 = max(y1 + 1, min(y2, h))
    return x1, y1, x2, y2


def collect_crops(image_dir: Path, class_count: int) -> list[tuple[np.ndarray, int]]:
    pairs: list[tuple[np.ndarray, int]] = []
    image_paths = [p for p in image_dir.rglob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}]
    image_paths.sort()

    for img_path in image_paths:
        label_path = image_to_label_path(img_path)
        if not label_path.exists():
            continue
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        h, w = img.shape[:2]
        for line in label_path.read_text(encoding="utf-8").splitlines():
            parts = line.strip().split()
            if len(parts) != 5:
                continue
            cls = int(float(parts[0]))
            if cls < 0 or cls >= class_count:
                continue
            cx, cy, bw, bh = [float(x) for x in parts[1:]]
            x1 = int((cx - bw / 2.0) * w)
            y1 = int((cy - bh / 2.0) * h)
            x2 = int((cx + bw / 2.0) * w)
            y2 = int((cy + bh / 2.0) * h)
            x1, y1, x2, y2 = clamp_box(x1, y1, x2, y2, w, h)
            if (x2 - x1) < 10 or (y2 - y1) < 10:
                continue
            crop = img[y1:y2, x1:x2]
            crop = cv2.cvtColor(cv2.resize(crop, (64, 64), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB)
            pairs.append((crop, cls))
    return pairs


def stratified_limit(samples: list[tuple[np.ndarray, int]], max_items: int, num_classes: int, seed: int = 42) -> list[tuple[np.ndarray, int]]:
    if len(samples) <= max_items:
        return samples
    by_class: dict[int, list[tuple[np.ndarray, int]]] = {i: [] for i in range(num_classes)}
    for item in samples:
        by_class[item[1]].append(item)
    rng = random.Random(seed)
    for arr in by_class.values():
        rng.shuffle(arr)
    per_class = max(1, max_items // max(1, num_classes))
    out: list[tuple[np.ndarray, int]] = []
    extras: list[tuple[np.ndarray, int]] = []
    for cid in range(num_classes):
        arr = by_class[cid]
        out.extend(arr[:per_class])
        extras.extend(arr[per_class:])
    if len(out) < max_items:
        rng.shuffle(extras)
        out.extend(extras[: max_items - len(out)])
    if len(out) > max_items:
        rng.shuffle(out)
        out = out[:max_items]
    return out


class TinyCNN(nn.Module):
    def __init__(self, num_classes: int) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.classifier = nn.Linear(64, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = x.flatten(1)
        return self.classifier(x)


def to_loader(samples: list[tuple[np.ndarray, int]], batch_size: int, shuffle: bool) -> DataLoader:
    xs = np.stack([s[0] for s in samples]).astype(np.float32) / 255.0
    ys = np.array([s[1] for s in samples], dtype=np.int64)
    x = torch.from_numpy(xs).permute(0, 3, 1, 2)
    y = torch.from_numpy(ys)
    return DataLoader(TensorDataset(x, y), batch_size=batch_size, shuffle=shuffle)


def save_confusion(y_true: np.ndarray, y_pred: np.ndarray, class_names: list[str], out_file: Path) -> None:
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))))
    fig, ax = plt.subplots(figsize=(8, 7))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(ax=ax, cmap="Purples", xticks_rotation=35, colorbar=False)
    ax.set_title("Confusion matrix - tiny_cnn")
    fig.tight_layout()
    fig.savefig(out_file, dpi=140)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--max-train-objects", type=int, default=8000)
    parser.add_argument("--max-test-objects", type=int, default=3000)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args()

    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)

    cfg = yaml.safe_load(args.data.read_text(encoding="utf-8"))
    class_names = list(cfg["names"])
    ds_root = Path(cfg["path"])
    train_dir = resolve_split_images(ds_root, cfg["train"])
    test_dir = resolve_split_images(ds_root, cfg.get("test", cfg["val"]))

    args.out.mkdir(parents=True, exist_ok=True)

    print("[1/4] Loading crops...")
    train = stratified_limit(collect_crops(train_dir, len(class_names)), args.max_train_objects, len(class_names))
    test = stratified_limit(collect_crops(test_dir, len(class_names)), args.max_test_objects, len(class_names))
    if not train or not test:
        raise SystemExit("No train/test crops were collected.")

    train_loader = to_loader(train, args.batch_size, shuffle=True)
    test_loader = to_loader(test, args.batch_size, shuffle=False)

    model = TinyCNN(num_classes=len(class_names))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    print("[2/4] Training tiny CNN...")
    losses: list[float] = []
    for _ in range(args.epochs):
        model.train()
        running = 0.0
        count = 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            running += float(loss.item()) * yb.size(0)
            count += yb.size(0)
        losses.append(running / max(1, count))

    print("[3/4] Evaluating...")
    model.eval()
    y_true: list[int] = []
    y_pred: list[int] = []
    with torch.no_grad():
        for xb, yb in test_loader:
            xb = xb.to(device)
            logits = model(xb)
            pred = torch.argmax(logits, dim=1).cpu().numpy()
            y_pred.extend(pred.tolist())
            y_true.extend(yb.numpy().tolist())

    y_true_np = np.array(y_true, dtype=np.int64)
    y_pred_np = np.array(y_pred, dtype=np.int64)
    acc = float(accuracy_score(y_true_np, y_pred_np))
    f1m = float(f1_score(y_true_np, y_pred_np, average="macro"))

    support = {
        "train": {class_names[i]: int(v) for i, v in enumerate(np.bincount(np.array([s[1] for s in train]), minlength=len(class_names)))},
        "test": {class_names[i]: int(v) for i, v in enumerate(np.bincount(y_true_np, minlength=len(class_names)))},
    }
    (args.out / "class_support.json").write_text(json.dumps(support, indent=2), encoding="utf-8")

    metrics = {
        "model": "tiny_cnn",
        "accuracy": acc,
        "f1_macro": f1m,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "train_objects": len(train),
        "test_objects": len(test),
        "device": str(device),
    }
    (args.out / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    print("[4/4] Saving plots and weights...")
    save_confusion(y_true_np, y_pred_np, class_names, args.out / "confusion_tiny_cnn.png")
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(np.arange(1, len(losses) + 1), losses, marker="o")
    ax.set_title("Tiny CNN training loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(args.out / "training_loss.png", dpi=140)
    plt.close(fig)

    torch.save(model.state_dict(), args.out / "tiny_cnn.pt")
    print(f"[OK] Done. See {args.out}")


if __name__ == "__main__":
    main()
