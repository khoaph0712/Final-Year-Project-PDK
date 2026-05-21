"""ANN/CNN baselines on object crops for lecturer-required DL comparison.

This script is intentionally separate from YOLO. It turns YOLO bounding-box
annotations into cropped classification samples, then trains image classifiers:
ANN on flattened pixels and CNN on spatial image tensors.
"""

from __future__ import annotations

import argparse
import csv
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
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from torch.utils.data import DataLoader, TensorDataset
import yaml


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATA = ROOT / "merged_dataset_v3" / "data.yaml"
DEFAULT_OUT = ROOT / "runs" / "dl" / "ann_cnn_baseline"


def resolve_split_images(ds_root: Path, split_rel: str) -> Path:
    p = Path(split_rel)
    if p.is_absolute():
        return p
    direct = ds_root / p
    if direct.exists():
        return direct
    # Roboflow exports sometimes write "../test/images" even when data.yaml
    # sits in the dataset root. Fall back to root/test/images.
    if split_rel.startswith("../"):
        fallback = ds_root / split_rel.replace("../", "", 1)
        if fallback.exists():
            return fallback
    return direct


def image_to_label_path(image_path: Path) -> Path:
    return Path(str(image_path).replace("\\images\\", "\\labels\\")).with_suffix(".txt")


def clamp_box(x1: int, y1: int, x2: int, y2: int, w: int, h: int) -> tuple[int, int, int, int]:
    x1 = max(0, min(x1, w - 1))
    y1 = max(0, min(y1, h - 1))
    x2 = max(x1 + 1, min(x2, w))
    y2 = max(y1 + 1, min(y2, h))
    return x1, y1, x2, y2


def infer_source_name(image_path: Path) -> str:
    name = image_path.name
    if "__" in name:
        return name.split("__", 1)[0]
    return "unknown"


def collect_crops(
    image_dir: Path,
    class_count: int,
    image_size: int,
    source_filter: str | None = None,
) -> list[tuple[np.ndarray, int]]:
    pairs: list[tuple[np.ndarray, int]] = []
    image_paths = [
        p
        for p in image_dir.rglob("*")
        if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    ]
    image_paths.sort()
    if source_filter:
        wanted = {item.strip() for item in source_filter.split(",") if item.strip()}
        image_paths = [p for p in image_paths if infer_source_name(p) in wanted]

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
            crop = cv2.cvtColor(
                cv2.resize(crop, (image_size, image_size), interpolation=cv2.INTER_AREA),
                cv2.COLOR_BGR2RGB,
            )
            pairs.append((crop, cls))
    return pairs


def stratified_limit(
    samples: list[tuple[np.ndarray, int]],
    max_items: int,
    num_classes: int,
    seed: int = 42,
) -> list[tuple[np.ndarray, int]]:
    if max_items <= 0 or len(samples) <= max_items:
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


def cap_per_class(
    samples: list[tuple[np.ndarray, int]],
    num_classes: int,
    max_per_class: int,
    seed: int = 42,
) -> list[tuple[np.ndarray, int]]:
    if max_per_class <= 0:
        return samples
    by_class: dict[int, list[tuple[np.ndarray, int]]] = {i: [] for i in range(num_classes)}
    for item in samples:
        by_class[item[1]].append(item)
    rng = random.Random(seed)
    out: list[tuple[np.ndarray, int]] = []
    for cid in range(num_classes):
        bucket = by_class[cid]
        rng.shuffle(bucket)
        out.extend(bucket[:max_per_class])
    rng.shuffle(out)
    return out


def compact_supported_splits(
    train: list[tuple[np.ndarray, int]],
    val: list[tuple[np.ndarray, int]],
    test: list[tuple[np.ndarray, int]],
    class_names: list[str],
    min_train: int,
    min_val: int,
    min_test: int,
) -> tuple[list[tuple[np.ndarray, int]], list[tuple[np.ndarray, int]], list[tuple[np.ndarray, int]], list[str], dict]:
    def counts(samples: list[tuple[np.ndarray, int]]) -> np.ndarray:
        return np.bincount(np.array([s[1] for s in samples], dtype=np.int64), minlength=len(class_names))

    train_counts = counts(train)
    val_counts = counts(val)
    test_counts = counts(test)
    keep = [
        cid
        for cid in range(len(class_names))
        if train_counts[cid] >= min_train and val_counts[cid] >= min_val and test_counts[cid] >= min_test
    ]
    dropped = [
        {
            "class_id": cid,
            "class_name": class_names[cid],
            "train_count": int(train_counts[cid]),
            "val_count": int(val_counts[cid]),
            "test_count": int(test_counts[cid]),
        }
        for cid in range(len(class_names))
        if cid not in keep
    ]
    if len(keep) < 2:
        raise SystemExit("Fewer than two classes have enough train/val/test samples for ANN/CNN.")
    old_to_new = {old: new for new, old in enumerate(keep)}

    def remap(samples: list[tuple[np.ndarray, int]]) -> list[tuple[np.ndarray, int]]:
        return [(img, old_to_new[label]) for img, label in samples if label in old_to_new]

    meta = {
        "kept_classes": [class_names[cid] for cid in keep],
        "dropped_unsupported_classes": dropped,
        "note": "Classes below min train/val/test support were removed and labels remapped to 0..K-1 for balanced ANN/CNN.",
    }
    return remap(train), remap(val), remap(test), [class_names[cid] for cid in keep], meta


def to_loader(samples: list[tuple[np.ndarray, int]], batch_size: int, shuffle: bool) -> DataLoader:
    xs = np.stack([s[0] for s in samples]).astype(np.float32) / 255.0
    ys = np.array([s[1] for s in samples], dtype=np.int64)
    x = torch.from_numpy(xs).permute(0, 3, 1, 2)
    y = torch.from_numpy(ys)
    return DataLoader(TensorDataset(x, y), batch_size=batch_size, shuffle=shuffle)


class SimpleANN(nn.Module):
    def __init__(self, num_classes: int, image_size: int) -> None:
        super().__init__()
        n_in = 3 * image_size * image_size
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(n_in, 512),
            nn.ReLU(),
            nn.Dropout(0.35),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


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


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
) -> tuple[float, float]:
    is_train = optimizer is not None
    model.train(is_train)
    total_loss = 0.0
    total = 0
    correct = 0
    with torch.set_grad_enabled(is_train):
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            if optimizer is not None:
                optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            if optimizer is not None:
                loss.backward()
                optimizer.step()
            total_loss += float(loss.item()) * yb.size(0)
            pred = torch.argmax(logits, dim=1)
            correct += int((pred == yb).sum().item())
            total += int(yb.size(0))
    return total_loss / max(1, total), correct / max(1, total)


def predict_all(model: nn.Module, loader: DataLoader, device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    y_true: list[int] = []
    y_pred: list[int] = []
    with torch.no_grad():
        for xb, yb in loader:
            logits = model(xb.to(device))
            pred = torch.argmax(logits, dim=1).cpu().numpy()
            y_pred.extend(pred.tolist())
            y_true.extend(yb.numpy().tolist())
    return np.array(y_true, dtype=np.int64), np.array(y_pred, dtype=np.int64)


def save_confusion(y_true: np.ndarray, y_pred: np.ndarray, class_names: list[str], out_file: Path, title: str) -> None:
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))))
    fig, ax = plt.subplots(figsize=(8, 7))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(ax=ax, cmap="Purples", xticks_rotation=35, colorbar=False)
    ax.set_title(title)
    ax.set_xlabel("Predicted class")
    ax.set_ylabel("True class")
    fig.tight_layout()
    fig.savefig(out_file, dpi=140)
    plt.close(fig)


def save_curves(history: list[dict], out_file: Path, model_name: str) -> None:
    epochs = [row["epoch"] for row in history]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(epochs, [row["train_loss"] for row in history], marker="o", label="train")
    axes[0].plot(epochs, [row["val_loss"] for row in history], marker="o", label="val")
    axes[0].set_title(f"{model_name} loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()
    axes[1].plot(epochs, [row["train_acc"] for row in history], marker="o", label="train")
    axes[1].plot(epochs, [row["val_acc"] for row in history], marker="o", label="val")
    axes[1].set_title(f"{model_name} accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].set_ylim(0.0, 1.0)
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(out_file, dpi=140)
    plt.close(fig)


def write_history_csv(history: list[dict], out_file: Path) -> None:
    with out_file.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["epoch", "train_loss", "val_loss", "train_acc", "val_acc"])
        writer.writeheader()
        writer.writerows(history)


def train_one_model(
    model_name: str,
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader,
    class_names: list[str],
    image_size: int,
    args: argparse.Namespace,
    device: torch.device,
) -> dict:
    out_dir = args.out / model_name
    out_dir.mkdir(parents=True, exist_ok=True)

    if model_name == "ann":
        model: nn.Module = SimpleANN(len(class_names), image_size)
    elif model_name == "cnn":
        model = TinyCNN(len(class_names))
    else:
        raise ValueError(f"Unknown model: {model_name}")

    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    history: list[dict] = []
    best_val_acc = -1.0
    best_state = None
    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = run_epoch(model, train_loader, criterion, device, optimizer)
        val_loss, val_acc = run_epoch(model, val_loader, criterion, device)
        row = {
            "epoch": epoch,
            "train_loss": round(train_loss, 6),
            "val_loss": round(val_loss, 6),
            "train_acc": round(train_acc, 6),
            "val_acc": round(val_acc, 6),
        }
        history.append(row)
        print(
            f"  {model_name} epoch {epoch}/{args.epochs}: "
            f"train_loss={train_loss:.4f} val_loss={val_loss:.4f} "
            f"train_acc={train_acc:.4f} val_acc={val_acc:.4f}"
        )
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)

    y_true, y_pred = predict_all(model, test_loader, device)
    acc = float(accuracy_score(y_true, y_pred))
    f1m = float(f1_score(y_true, y_pred, average="macro"))
    report = classification_report(
        y_true,
        y_pred,
        labels=list(range(len(class_names))),
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )
    metrics = {
        "model": model_name,
        "accuracy": acc,
        "f1_macro": f1m,
        "best_val_accuracy": best_val_acc,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "image_size": image_size,
        "device": str(device),
    }

    torch.save(model.state_dict(), out_dir / f"{model_name}.pt")
    torch.save(model, out_dir / f"{model_name}_full.pt")
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    (out_dir / "classification_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (out_dir / "training_log.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    write_history_csv(history, out_dir / "training_log.csv")
    save_confusion(y_true, y_pred, class_names, out_dir / f"confusion_{model_name}.png", f"Confusion matrix - {model_name.upper()}")
    save_curves(history, out_dir / f"training_curves_{model_name}.png", model_name.upper())
    return metrics


def write_report(out_dir: Path, data_yaml: Path, class_names: list[str], support: dict, results: list[dict]) -> None:
    lines = [
        "# ANN/CNN Baseline Report",
        "",
        "## Purpose",
        "- YOLO is paused for this experiment.",
        "- This run trains object-crop classifiers only: ANN and/or CNN.",
        "- Each model saves weights, logs, curves, confusion matrix, and classification report for end-user/report inspection.",
        "",
        "## Dataset",
        f"- Data YAML: `{data_yaml}`",
        "- Source filter: see `run_config.json`.",
        f"- Classes: {', '.join(class_names)}",
        f"- Train objects: {sum(support['train'].values())}",
        f"- Validation objects: {sum(support['val'].values())}",
        f"- Test objects: {sum(support['test'].values())}",
        "",
        "## Models",
        "- **ANN:** flattened RGB crop pixels; baseline that does not preserve local spatial structure.",
        "- **CNN:** convolutional image classifier; better suited for image texture/shape because it preserves spatial locality.",
        "",
        "## Outputs",
        "- `training_log.csv/json`: epoch-wise train/val loss and accuracy.",
        "- `training_curves_*.png`: train/val loss and accuracy figure.",
        "- `*.pt` and `*_full.pt`: saved model weights/full model.",
        "- `confusion_*.png`: classification confusion matrix. No `background` class is used here because this is crop classification, not object detection.",
        "- `classification_report.json`: precision, recall, F1-score, and support by class.",
        "",
        "## Results",
        "| Model | Test Accuracy | Test F1-macro | Best Val Accuracy |",
        "|---|---:|---:|---:|",
    ]
    for item in results:
        lines.append(
            f"| {item['model']} | {item['accuracy']:.4f} | {item['f1_macro']:.4f} | {item['best_val_accuracy']:.4f} |"
        )
    lines.append("")
    lines.append("## Confusion Matrix Background Note")
    lines.append(
        "- For ANN/CNN crop classification, every sample already has a real class, so the matrix only contains dataset classes."
    )
    lines.append(
        "- In YOLO detection confusion matrices, `background` usually means unmatched predictions or missed ground-truth boxes; "
        "it is not an eighth trash class."
    )
    (out_dir / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--model", choices=["ann", "cnn", "both"], default="both")
    parser.add_argument("--max-train-objects", type=int, default=8000)
    parser.add_argument("--max-val-objects", type=int, default=2000)
    parser.add_argument("--max-test-objects", type=int, default=3000)
    parser.add_argument("--max-per-class-train", type=int, default=0)
    parser.add_argument("--max-per-class-val", type=int, default=0)
    parser.add_argument("--max-per-class-test", type=int, default=0)
    parser.add_argument("--min-per-class-train", type=int, default=1)
    parser.add_argument("--min-per-class-val", type=int, default=1)
    parser.add_argument("--min-per-class-test", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--image-size", type=int, default=64)
    parser.add_argument(
        "--source-filter",
        type=str,
        default="",
        help="Optional comma-separated filename source prefixes, e.g. rf_taco_trash.",
    )
    args = parser.parse_args()

    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)

    cfg = yaml.safe_load(args.data.read_text(encoding="utf-8"))
    class_names = list(cfg["names"])
    ds_root = Path(cfg.get("path", args.data.parent))
    train_dir = resolve_split_images(ds_root, cfg["train"])
    val_dir = resolve_split_images(ds_root, cfg.get("val", cfg.get("valid", cfg["train"])))
    test_dir = resolve_split_images(ds_root, cfg.get("test", cfg.get("val", cfg["train"])))

    args.out.mkdir(parents=True, exist_ok=True)
    source_filter = args.source_filter.strip() or None
    (args.out / "run_config.json").write_text(
        json.dumps(
            {
                "data": str(args.data),
                "out": str(args.out),
                "model": args.model,
                "source_filter": source_filter,
                "max_train_objects": args.max_train_objects,
                "max_val_objects": args.max_val_objects,
                "max_test_objects": args.max_test_objects,
                "max_per_class_train": args.max_per_class_train,
                "max_per_class_val": args.max_per_class_val,
                "max_per_class_test": args.max_per_class_test,
                "min_per_class_train": args.min_per_class_train,
                "min_per_class_val": args.min_per_class_val,
                "min_per_class_test": args.min_per_class_test,
                "epochs": args.epochs,
                "batch_size": args.batch_size,
                "lr": args.lr,
                "image_size": args.image_size,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print("[1/4] Loading crops...")
    train = stratified_limit(
        cap_per_class(
            collect_crops(train_dir, len(class_names), args.image_size, source_filter),
            len(class_names),
            args.max_per_class_train,
        ),
        args.max_train_objects,
        len(class_names),
    )
    val = stratified_limit(
        cap_per_class(
            collect_crops(val_dir, len(class_names), args.image_size, source_filter),
            len(class_names),
            args.max_per_class_val,
        ),
        args.max_val_objects,
        len(class_names),
    )
    test = stratified_limit(
        cap_per_class(
            collect_crops(test_dir, len(class_names), args.image_size, source_filter),
            len(class_names),
            args.max_per_class_test,
        ),
        args.max_test_objects,
        len(class_names),
    )
    if not train or not val or not test:
        raise SystemExit("No train/val/test crops were collected.")
    train, val, test, class_names, balance_meta = compact_supported_splits(
        train,
        val,
        test,
        class_names,
        args.min_per_class_train,
        args.min_per_class_val,
        args.min_per_class_test,
    )

    support = {
        "train": {class_names[i]: int(v) for i, v in enumerate(np.bincount(np.array([s[1] for s in train]), minlength=len(class_names)))},
        "val": {class_names[i]: int(v) for i, v in enumerate(np.bincount(np.array([s[1] for s in val]), minlength=len(class_names)))},
        "test": {class_names[i]: int(v) for i, v in enumerate(np.bincount(np.array([s[1] for s in test]), minlength=len(class_names)))},
    }
    support.update(balance_meta)
    (args.out / "class_support.json").write_text(json.dumps(support, indent=2), encoding="utf-8")

    train_loader = to_loader(train, args.batch_size, shuffle=True)
    val_loader = to_loader(val, args.batch_size, shuffle=False)
    test_loader = to_loader(test, args.batch_size, shuffle=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    models = ["ann", "cnn"] if args.model == "both" else [args.model]
    print(f"[2/4] Training {', '.join(models)} on {device}...")
    results = [
        train_one_model(name, train_loader, val_loader, test_loader, class_names, args.image_size, args, device)
        for name in models
    ]

    print("[3/4] Saving summary...")
    (args.out / "metrics_summary.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    write_report(args.out, args.data.resolve(), class_names, support, results)

    print(f"[4/4] Done. See {args.out}")


if __name__ == "__main__":
    main()
