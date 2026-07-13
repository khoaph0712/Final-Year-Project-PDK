"""Retrain the ConvNeXt + 637-feature classifier on hard-case data."""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT / "scripts"
sys.path.append(str(SCRIPTS_DIR / "archive"))
from custom_feature_extractor import extract_637_features  # noqa: E402
from ml_balanced_training import augment_crop  # noqa: E402


CLASSES = ["plastic", "glass", "metal", "paper", "cardboard", "organic", "Background"]
DEFAULT_DATA = ROOT / "data" / "hard_case_classifier_v1" / "data.yaml"
DEFAULT_OUT = ROOT / "runs" / "dl" / "convnext_hardcase_tuned"


@dataclass(frozen=True)
class Sample:
    path: Path
    label: int
    augment_seed: int | None = None


class ConvNeXtFeatureExtractor(nn.Module):
    def __init__(self, unfreeze_final: bool = False) -> None:
        super().__init__()
        self.backbone = models.convnext_tiny(weights=models.ConvNeXt_Tiny_Weights.DEFAULT)
        self.backbone.classifier = nn.Identity()
        for param in self.backbone.parameters():
            param.requires_grad = False
        if unfreeze_final:
            print("[INFO] Unfreezing ConvNeXt final stage features[7].")
            for param in self.backbone.features[7].parameters():
                param.requires_grad = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)
        return torch.flatten(features, 1)


class Stage3EnsembleClassifier(nn.Module):
    def __init__(self, num_classes: int = 7, dropout_rate: float = 0.3, unfreeze_final: bool = False) -> None:
        super().__init__()
        self.convnext_extractor = ConvNeXtFeatureExtractor(unfreeze_final=unfreeze_final)
        self.classifier = nn.Sequential(
            nn.Linear(768 + 637, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate),
            nn.Linear(256, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(64, num_classes),
        )

    def forward(self, image_tensor: torch.Tensor, handcrafted_features_tensor: torch.Tensor) -> torch.Tensor:
        deep_features = self.convnext_extractor(image_tensor)
        return self.classifier(torch.cat((deep_features, handcrafted_features_tensor), dim=1))


def read_data_yaml(path: Path) -> dict[str, str]:
    cfg: dict[str, str] = {}
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("- ") or ":" not in line:
            continue
        key, value = line.split(":", 1)
        cfg[key.strip()] = value.strip()
    return cfg


def image_paths(folder: Path) -> list[Path]:
    return [
        item
        for item in folder.iterdir()
        if item.is_file() and item.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    ]


def load_split(root: Path, split: str) -> list[Sample]:
    samples: list[Sample] = []
    for label, class_name in enumerate(CLASSES):
        class_dir = root / split / class_name
        if not class_dir.exists():
            continue
        for path in image_paths(class_dir):
            samples.append(Sample(path=path, label=label))
    return samples


def balance_train(samples: list[Sample], max_per_class: int, seed: int) -> list[Sample]:
    rng = random.Random(seed)
    by_label: dict[int, list[Sample]] = {idx: [] for idx in range(len(CLASSES))}
    for sample in samples:
        by_label[sample.label].append(sample)

    balanced: list[Sample] = []
    for label, items in by_label.items():
        rng.shuffle(items)
        if len(items) >= max_per_class:
            selected = items[:max_per_class]
            print(f"  * {CLASSES[label]:<12}: raw {len(items):>5} -> {len(selected):>5}")
        else:
            selected = list(items)
            needed = max_per_class - len(items)
            for index in range(needed):
                base = rng.choice(items)
                selected.append(Sample(path=base.path, label=base.label, augment_seed=seed * 100000 + label * 1000 + index))
            print(f"  * {CLASSES[label]:<12}: raw {len(items):>5} -> {len(selected):>5} augmented")
        balanced.extend(selected)
    rng.shuffle(balanced)
    return balanced


def load_bgr(sample: Sample) -> np.ndarray:
    image = cv2.imread(str(sample.path))
    if image is None:
        raise ValueError(f"failed to read image: {sample.path}")
    if sample.augment_seed is not None:
        image = augment_crop(image, random.Random(sample.augment_seed))
    return image


def load_or_extract_features(samples: list[Sample], cache_path: Path, split_name: str) -> np.ndarray:
    if cache_path.exists():
        cached = np.load(cache_path)
        if cached.shape[0] == len(samples):
            print(f"[INFO] Loading cached {split_name} 637-feature vectors: {cache_path}")
            return cached
        print(f"[WARN] Ignoring stale {split_name} feature cache with shape {cached.shape}.")

    features = []
    for index, sample in enumerate(samples):
        if index % 500 == 0:
            print(f"  - {split_name} features: {index}/{len(samples)}")
        features.append(extract_637_features(load_bgr(sample)))
    arr = np.array(features, dtype=np.float32)
    np.save(cache_path, arr)
    return arr


class HybridPathDataset(Dataset):
    def __init__(self, samples: list[Sample], handcrafted: np.ndarray, transform: transforms.Compose) -> None:
        self.samples = samples
        self.handcrafted = handcrafted.astype(np.float32)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        sample = self.samples[idx]
        image = load_bgr(sample)
        image = cv2.resize(image, (224, 224), interpolation=cv2.INTER_CUBIC)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_tensor = self.transform(Image.fromarray(image))
        handcrafted_tensor = torch.tensor(self.handcrafted[idx], dtype=torch.float32)
        label_tensor = torch.tensor(sample.label, dtype=torch.long)
        return image_tensor, handcrafted_tensor, label_tensor


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> tuple[float, float, list[int], list[int]]:
    model.eval()
    preds: list[int] = []
    trues: list[int] = []
    with torch.no_grad():
        for images, handcrafted, labels in loader:
            images = images.to(device)
            handcrafted = handcrafted.to(device)
            labels = labels.to(device)
            logits = model(images, handcrafted)
            pred = logits.argmax(dim=1)
            preds.extend(pred.cpu().tolist())
            trues.extend(labels.cpu().tolist())
    acc = accuracy_score(trues, preds)
    macro_f1 = f1_score(trues, preds, average="macro", zero_division=0)
    return acc, macro_f1, trues, preds


def write_report(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=True), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--train-max-per-class", type=int, default=1800)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--warmup-epochs", type=int, default=3)
    parser.add_argument("--finetune-epochs", type=int, default=7)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=2)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    args.out.mkdir(parents=True, exist_ok=True)

    cfg = read_data_yaml(args.data)
    data_root = Path(cfg.get("path", args.data.parent))
    if not data_root.is_absolute():
        data_root = args.data.parent / data_root

    print("====================================================")
    print("Hard-case ConvNeXt + 637-feature retraining")
    print("====================================================")
    print(f"[INFO] Dataset: {data_root}")
    print(f"[INFO] Output:  {args.out}")

    train_raw = load_split(data_root, "train")
    val_samples = load_split(data_root, "val")
    test_samples = load_split(data_root, "test")
    print(f"[INFO] Raw samples train={len(train_raw)} val={len(val_samples)} test={len(test_samples)}")
    print("[INFO] Balanced train split:")
    train_samples = balance_train(train_raw, args.train_max_per_class, args.seed)

    x_train = load_or_extract_features(train_samples, args.out / "train_handcrafted_637.npy", "train")
    x_val = load_or_extract_features(val_samples, args.out / "val_handcrafted_637.npy", "val")
    x_test = load_or_extract_features(test_samples, args.out / "test_handcrafted_637.npy", "test")

    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train).astype(np.float32)
    x_val = scaler.transform(x_val).astype(np.float32)
    x_test = scaler.transform(x_test).astype(np.float32)
    np.savez(args.out / "handcrafted_scaler.npz", mean=scaler.mean_.astype(np.float32), scale=scaler.scale_.astype(np.float32))

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    train_loader = DataLoader(
        HybridPathDataset(train_samples, x_train, transform),
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=args.num_workers,
    )
    val_loader = DataLoader(HybridPathDataset(val_samples, x_val, transform), batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    test_loader = DataLoader(HybridPathDataset(test_samples, x_test, transform), batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Device: {device}")
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    model = Stage3EnsembleClassifier(num_classes=len(CLASSES), unfreeze_final=False).to(device)
    optimizer = optim.AdamW(model.classifier.parameters(), lr=1e-3, weight_decay=0.01)

    history: list[dict[str, float]] = []
    started = time.time()

    print(f"\n--- WARMUP ({args.warmup_epochs} epochs) ---")
    for epoch in range(1, args.warmup_epochs + 1):
        model.train()
        total_loss = 0.0
        total = 0
        correct = 0
        for images, handcrafted, labels in train_loader:
            images = images.to(device)
            handcrafted = handcrafted.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            logits = model(images, handcrafted)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * images.size(0)
            total += labels.size(0)
            correct += (logits.argmax(dim=1) == labels).sum().item()
        val_acc, val_f1, _, _ = evaluate(model, val_loader, device)
        row = {
            "phase": "warmup",
            "epoch": epoch,
            "train_loss": total_loss / total,
            "train_acc": correct / total,
            "val_acc": val_acc,
            "val_macro_f1": val_f1,
        }
        history.append(row)
        print(f"  * Warmup {epoch:02d} | loss={row['train_loss']:.4f} train_acc={row['train_acc']*100:.2f}% val_acc={val_acc*100:.2f}% val_f1={val_f1:.4f}")

    torch.save(model.state_dict(), args.out / "warmed_up_model.pth")

    print(f"\n--- FINE-TUNE ({args.finetune_epochs} epochs) ---")
    model_tuned = Stage3EnsembleClassifier(num_classes=len(CLASSES), unfreeze_final=True)
    model_tuned.load_state_dict(torch.load(args.out / "warmed_up_model.pth", weights_only=True))
    model_tuned.to(device)
    optimizer_tuned = optim.AdamW(
        [
            {"params": model_tuned.convnext_extractor.backbone.features[7].parameters(), "lr": 1e-5},
            {"params": model_tuned.classifier.parameters(), "lr": 1e-4},
        ],
        weight_decay=0.05,
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer_tuned, T_max=max(1, args.finetune_epochs))

    best_val_f1 = -1.0
    for epoch in range(1, args.finetune_epochs + 1):
        model_tuned.train()
        total_loss = 0.0
        total = 0
        correct = 0
        for images, handcrafted, labels in train_loader:
            images = images.to(device)
            handcrafted = handcrafted.to(device)
            labels = labels.to(device)
            optimizer_tuned.zero_grad()
            logits = model_tuned(images, handcrafted)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer_tuned.step()
            total_loss += loss.item() * images.size(0)
            total += labels.size(0)
            correct += (logits.argmax(dim=1) == labels).sum().item()
        scheduler.step()

        val_acc, val_f1, _, _ = evaluate(model_tuned, val_loader, device)
        row = {
            "phase": "finetune",
            "epoch": epoch,
            "train_loss": total_loss / total,
            "train_acc": correct / total,
            "val_acc": val_acc,
            "val_macro_f1": val_f1,
        }
        history.append(row)
        print(f"  * Fine {epoch:02d} | loss={row['train_loss']:.4f} train_acc={row['train_acc']*100:.2f}% val_acc={val_acc*100:.2f}% val_f1={val_f1:.4f}")
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            torch.save(model_tuned.state_dict(), args.out / "best_convnext_ensemble_tuned.pth")

    model_tuned.load_state_dict(torch.load(args.out / "best_convnext_ensemble_tuned.pth", map_location=device, weights_only=True))
    val_acc, val_f1, val_true, val_pred = evaluate(model_tuned, val_loader, device)
    test_acc, test_f1, test_true, test_pred = evaluate(model_tuned, test_loader, device)

    report = {
        "data": str(args.data),
        "output": str(args.out),
        "classes": CLASSES,
        "train_samples_balanced": len(train_samples),
        "val_samples": len(val_samples),
        "test_samples": len(test_samples),
        "warmup_epochs": args.warmup_epochs,
        "finetune_epochs": args.finetune_epochs,
        "best_val_macro_f1": best_val_f1,
        "final_val_accuracy": val_acc,
        "final_val_macro_f1": val_f1,
        "final_test_accuracy": test_acc,
        "final_test_macro_f1": test_f1,
        "elapsed_seconds": round(time.time() - started, 2),
        "history": history,
        "val_classification_report": classification_report(val_true, val_pred, target_names=CLASSES, zero_division=0, output_dict=True),
        "test_classification_report": classification_report(test_true, test_pred, target_names=CLASSES, zero_division=0, output_dict=True),
    }
    write_report(args.out / "RESULT.json", report)
    (args.out / "RESULT.md").write_text(
        "# Hard-case ConvNeXt Classifier Retraining\n\n"
        f"- Dataset: `{args.data}`\n"
        f"- Balanced train samples: {len(train_samples)}\n"
        f"- Validation samples: {len(val_samples)}\n"
        f"- Test samples: {len(test_samples)}\n"
        f"- Best validation macro F1: {best_val_f1:.4f}\n"
        f"- Final validation accuracy: {val_acc*100:.2f}%\n"
        f"- Final validation macro F1: {val_f1:.4f}\n"
        f"- Final test accuracy: {test_acc*100:.2f}%\n"
        f"- Final test macro F1: {test_f1:.4f}\n"
        f"- Elapsed seconds: {report['elapsed_seconds']}\n",
        encoding="utf-8",
    )
    print("\n====================================================")
    print("[SUCCESS] Hard-case classifier retraining completed")
    print(f"  - Best val macro F1: {best_val_f1:.4f}")
    print(f"  - Test accuracy: {test_acc*100:.2f}%")
    print(f"  - Test macro F1: {test_f1:.4f}")
    print(f"  - Model: {args.out / 'best_convnext_ensemble_tuned.pth'}")
    print("====================================================")


if __name__ == "__main__":
    main()
