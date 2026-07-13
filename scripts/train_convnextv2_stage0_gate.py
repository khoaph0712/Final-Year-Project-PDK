"""Train a ConvNeXtV2 crop classifier for Stage 0 trash-validity gating."""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import timm
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from PIL import Image
from sklearn.metrics import accuracy_score, classification_report, f1_score
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA = ROOT / "data" / "stage0_trashify_v1" / "data.yaml"
DEFAULT_OUT = ROOT / "runs" / "dl" / "convnextv2_stage0_trash_gate"


def read_data_yaml(path: Path) -> dict[str, str]:
    cfg: dict[str, str] = {}
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("- ") or ":" not in line:
            continue
        key, value = line.split(":", 1)
        cfg[key.strip()] = value.strip()
    return cfg


def read_class_names(path: Path, fallback: list[str]) -> list[str]:
    names: list[str] = []
    in_names = False
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if line == "names:":
            in_names = True
            continue
        if in_names:
            if line.startswith("- "):
                names.append(line[2:].strip())
                continue
            if line and ":" in line:
                break
    return names or fallback


def image_files(path: Path) -> list[Path]:
    return [item for item in path.iterdir() if item.is_file() and item.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}]


def load_split(root: Path, split: str, classes: list[str]) -> list[tuple[Path, int]]:
    samples = []
    for label, class_name in enumerate(classes):
        class_dir = root / split / class_name
        if class_dir.exists():
            samples.extend((path, label) for path in image_files(class_dir))
    return samples


class ImageFolderList(Dataset):
    def __init__(self, samples: list[tuple[Path, int]], transform: transforms.Compose) -> None:
        self.samples = samples
        self.transform = transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        path, label = self.samples[idx]
        image = Image.open(path).convert("RGB")
        return self.transform(image), torch.tensor(label, dtype=torch.long)


def make_sampler(samples: list[tuple[Path, int]]) -> WeightedRandomSampler:
    counts = np.bincount([label for _, label in samples], minlength=max(label for _, label in samples) + 1)
    weights = [1.0 / max(1, counts[label]) for _, label in samples]
    return WeightedRandomSampler(weights, num_samples=len(samples), replacement=True)


def run_epoch(model: nn.Module, loader: DataLoader, criterion: nn.Module, optimizer: optim.Optimizer | None, device: torch.device) -> float:
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    total = 0
    with torch.set_grad_enabled(training):
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)
            logits = model(images)
            loss = criterion(logits, labels)
            if training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            total_loss += float(loss.item()) * int(labels.numel())
            total += int(labels.numel())
    return total_loss / max(1, total)


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> tuple[float, float, list[int], list[int]]:
    model.eval()
    trues: list[int] = []
    preds: list[int] = []
    with torch.no_grad():
        for images, labels in loader:
            logits = model(images.to(device))
            pred = logits.argmax(dim=1).cpu().tolist()
            preds.extend(pred)
            trues.extend(labels.tolist())
    return (
        accuracy_score(trues, preds),
        f1_score(trues, preds, average="macro", zero_division=0),
        trues,
        preds,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--model", default="convnextv2_tiny.fcmae_ft_in22k_in1k")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=24)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--artifact-name", default="best_convnextv2_classifier.pth")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    cfg = read_data_yaml(args.data)
    classes = read_class_names(args.data, ["trash", "not_trash", "hand", "bin"])
    data_root = Path(cfg.get("path", args.data.parent))
    if not data_root.is_absolute():
        data_root = args.data.parent / data_root

    train_samples = load_split(data_root, "train", classes)
    val_samples = load_split(data_root, "val", classes)
    test_samples = load_split(data_root, "test", classes)
    if not train_samples or not val_samples:
        raise SystemExit(f"missing train/val samples under {data_root}")

    transform_train = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomResizedCrop(224, scale=(0.72, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.12),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    transform_eval = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    train_loader = DataLoader(
        ImageFolderList(train_samples, transform_train),
        batch_size=args.batch_size,
        sampler=make_sampler(train_samples),
        num_workers=args.num_workers,
    )
    val_loader = DataLoader(ImageFolderList(val_samples, transform_eval), batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    test_loader = DataLoader(ImageFolderList(test_samples, transform_eval), batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = timm.create_model(args.model, pretrained=True, num_classes=len(classes)).to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.04)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.04)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_f1 = -1.0
    history = []
    started = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        train_loss = run_epoch(model, train_loader, criterion, optimizer, device)
        val_loss = run_epoch(model, val_loader, criterion, None, device)
        val_acc, val_f1, _, _ = evaluate(model, val_loader, device)
        scheduler.step()
        record = {"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss, "val_accuracy": val_acc, "val_macro_f1": val_f1}
        history.append(record)
        print(json.dumps(record))
        if val_f1 > best_f1:
            best_f1 = val_f1
            torch.save(model.state_dict(), args.out / args.artifact_name)

    model.load_state_dict(torch.load(args.out / args.artifact_name, map_location=device, weights_only=True))
    splits = {}
    for name, loader in (("val", val_loader), ("test", test_loader)):
        acc, macro_f1, trues, preds = evaluate(model, loader, device)
        splits[name] = {
            "accuracy": acc,
            "macro_f1": macro_f1,
            "classification_report": classification_report(trues, preds, target_names=classes, output_dict=True, zero_division=0),
        }

    result = {
        "data": str(args.data),
        "model": args.model,
        "classes": classes,
        "train_samples": len(train_samples),
        "val_samples": len(val_samples),
        "test_samples": len(test_samples),
        "best_val_macro_f1": best_f1,
        "elapsed_seconds": round(time.perf_counter() - started, 2),
        "history": history,
        "splits": splits,
    }
    (args.out / "RESULT.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    (args.out / "RESULT.md").write_text(
        "# ConvNeXtV2 Stage 0 Trash Gate\n\n"
        f"- Model: `{args.model}`\n"
        f"- Dataset: `{args.data}`\n"
        f"- Train / val / test: {len(train_samples)} / {len(val_samples)} / {len(test_samples)}\n"
        f"- Best val macro F1: {best_f1:.4f}\n"
        f"- Test accuracy: {splits['test']['accuracy']:.2%}\n"
        f"- Test macro F1: {splits['test']['macro_f1']:.4f}\n",
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
