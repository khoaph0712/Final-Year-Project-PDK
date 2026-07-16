"""Evaluate a ConvNeXt + 637-feature classifier checkpoint."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
from sklearn.metrics import accuracy_score, classification_report, f1_score
from torch.utils.data import DataLoader, Dataset


ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "scripts"))
from custom_feature_extractor import extract_637_features  # noqa: E402
from stage2_model import Stage3EnsembleClassifier  # noqa: E402


CLASSES = ["plastic", "glass", "metal", "paper", "cardboard", "organic", "Background"]


class EvalDataset(Dataset):
    def __init__(self, samples: list[tuple[Path, int]], features: np.ndarray) -> None:
        self.samples = samples
        self.features = features.astype(np.float32)
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        path, label = self.samples[index]
        image = cv2.imread(str(path))
        if image is None:
            raise ValueError(f"failed to read image: {path}")
        image = cv2.resize(image, (224, 224), interpolation=cv2.INTER_CUBIC)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return (
            self.transform(Image.fromarray(image)),
            torch.tensor(self.features[index], dtype=torch.float32),
            torch.tensor(label, dtype=torch.long),
        )


def read_data_yaml(path: Path) -> dict[str, str]:
    cfg: dict[str, str] = {}
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("- ") or ":" not in line:
            continue
        key, value = line.split(":", 1)
        cfg[key.strip()] = value.strip()
    return cfg


def load_split(root: Path, split: str) -> list[tuple[Path, int]]:
    samples: list[tuple[Path, int]] = []
    for label, class_name in enumerate(CLASSES):
        class_dir = root / split / class_name
        if not class_dir.exists():
            continue
        for path in sorted(class_dir.iterdir()):
            if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}:
                samples.append((path, label))
    return samples


def load_or_extract_features(samples: list[tuple[Path, int]], cache_path: Path) -> np.ndarray:
    if cache_path.exists():
        cached = np.load(cache_path)
        if cached.shape[0] == len(samples):
            return cached
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    features = []
    for index, (path, _) in enumerate(samples):
        if index % 500 == 0:
            print(f"  - features {index}/{len(samples)}")
        image = cv2.imread(str(path))
        if image is None:
            raise ValueError(f"failed to read image: {path}")
        features.append(extract_637_features(image))
    arr = np.array(features, dtype=np.float32)
    np.save(cache_path, arr)
    return arr


def evaluate_split(model: nn.Module, loader: DataLoader, device: torch.device) -> tuple[dict, list[int], list[int]]:
    model.eval()
    trues: list[int] = []
    preds: list[int] = []
    with torch.no_grad():
        for images, features, labels in loader:
            images = images.to(device)
            features = features.to(device)
            labels = labels.to(device)
            logits = model(images, features)
            pred = logits.argmax(dim=1)
            trues.extend(labels.cpu().tolist())
            preds.extend(pred.cpu().tolist())
    metrics = {
        "accuracy": accuracy_score(trues, preds),
        "macro_f1": f1_score(trues, preds, average="macro", zero_division=0),
        "weighted_f1": f1_score(trues, preds, average="weighted", zero_division=0),
        "classification_report": classification_report(trues, preds, target_names=CLASSES, zero_division=0, output_dict=True),
    }
    return metrics, trues, preds


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=ROOT / "data" / "hard_case_classifier_v1" / "data.yaml")
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--scaler", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = read_data_yaml(args.data)
    data_root = Path(cfg.get("path", args.data.parent))
    if not data_root.is_absolute():
        data_root = args.data.parent / data_root

    scaler = np.load(args.scaler)
    mean = scaler["mean"].astype(np.float32)
    scale = scaler["scale"].astype(np.float32)
    scale[scale == 0] = 1.0

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Stage3EnsembleClassifier(num_classes=len(CLASSES))
    model.load_state_dict(torch.load(args.weights, map_location=device, weights_only=True))
    model.to(device)

    result = {
        "data": str(args.data),
        "weights": str(args.weights),
        "scaler": str(args.scaler),
        "splits": {},
    }
    for split in ("val", "test"):
        samples = load_split(data_root, split)
        raw_features = load_or_extract_features(samples, args.out.parent / f"{args.out.stem}_{split}_features.npy")
        features = ((raw_features - mean) / scale).astype(np.float32)
        loader = DataLoader(EvalDataset(samples, features), batch_size=args.batch_size, shuffle=False)
        metrics, _, _ = evaluate_split(model, loader, device)
        metrics["samples"] = len(samples)
        result["splits"][split] = metrics
        print(f"{split}: accuracy={metrics['accuracy']*100:.2f}% macro_f1={metrics['macro_f1']:.4f} samples={len(samples)}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, ensure_ascii=True), encoding="utf-8")


if __name__ == "__main__":
    main()
