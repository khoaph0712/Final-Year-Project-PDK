#!/usr/bin/env python
"""Cross-domain (studio vs field) eval of the ACTUALLY DEPLOYED ConvNeXt classifier,
on the leak-clean test set (data/merged_dataset_v5_clean_test/test).

Reuses the domain marker from cross_dataset_validation.py and the model class from
evaluate_convnext_classifier.py, but points at the real deployed checkpoint
(web/server.py TUNED_CLASSIFIER_PATH) instead of training a fresh proxy MLP.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "scripts" / "archive"))
from custom_feature_extractor import extract_637_features  # noqa: E402

CLASSES = ["plastic", "glass", "metal", "paper", "cardboard", "organic", "Background"]
DATA_ROOT = ROOT / "data" / "merged_dataset_v5_clean_test" / "test"
WEIGHTS = ROOT / "runs" / "dl" / "convnext_ensemble_tuned" / "best_convnext_ensemble_tuned.pth"
SCALER = ROOT / "runs" / "dl" / "convnext_ensemble_tuned" / "handcrafted_scaler.npz"
OUT = ROOT / "runs" / "audits" / "classifier_domain_gap.json"
FIELD_RE = re.compile(r"(_train_|_test_|_val_|_valid_|rf_)", re.IGNORECASE)


class ConvNeXtFeatureExtractor(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.backbone = models.convnext_tiny(weights=None)
        self.backbone.classifier = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.flatten(self.backbone(x), 1)


class Stage3EnsembleClassifier(nn.Module):
    def __init__(self, num_classes: int = 7) -> None:
        super().__init__()
        self.convnext_extractor = ConvNeXtFeatureExtractor()
        self.classifier = nn.Sequential(
            nn.Linear(768 + 637, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(256, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(64, num_classes),
        )

    def forward(self, image_tensor: torch.Tensor, handcrafted_features_tensor: torch.Tensor) -> torch.Tensor:
        deep_features = self.convnext_extractor(image_tensor)
        return self.classifier(torch.cat((deep_features, handcrafted_features_tensor), dim=1))


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

    def __getitem__(self, index: int):
        path, label = self.samples[index]
        image = cv2.imread(str(path))
        image = cv2.resize(image, (224, 224), interpolation=cv2.INTER_CUBIC)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return (
            self.transform(Image.fromarray(image)),
            torch.tensor(self.features[index], dtype=torch.float32),
            torch.tensor(label, dtype=torch.long),
        )


def domain_of(filename: str) -> str:
    return "field" if FIELD_RE.search(filename) else "studio"


def collect_domain_samples() -> dict[str, list[tuple[Path, int]]]:
    buckets: dict[str, list[tuple[Path, int]]] = {"studio": [], "field": []}
    for label, cname in enumerate(CLASSES):
        cdir = DATA_ROOT / cname
        if not cdir.exists():
            continue
        for p in cdir.iterdir():
            if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}:
                buckets[domain_of(p.name)].append((p, label))
    return buckets


def extract_features(samples: list[tuple[Path, int]], cache_path: Path) -> np.ndarray:
    if cache_path.exists():
        cached = np.load(cache_path)
        if cached.shape[0] == len(samples):
            return cached
    feats = []
    for i, (p, _) in enumerate(samples):
        if i % 500 == 0:
            print(f"  - features {i}/{len(samples)}")
        img = cv2.imread(str(p))
        feats.append(extract_637_features(img))
    arr = np.asarray(feats, dtype=np.float32)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(cache_path, arr)
    return arr


def evaluate(model, loader, device) -> dict:
    model.eval()
    trues, preds = [], []
    with torch.no_grad():
        for images, features, labels in loader:
            logits = model(images.to(device), features.to(device))
            preds.extend(logits.argmax(dim=1).cpu().tolist())
            trues.extend(labels.tolist())
    cm = confusion_matrix(trues, preds, labels=list(range(len(CLASSES))))
    return {
        "accuracy": accuracy_score(trues, preds),
        "macro_f1": f1_score(trues, preds, average="macro", zero_division=0),
        "classification_report": classification_report(trues, preds, labels=list(range(len(CLASSES))), target_names=CLASSES, zero_division=0, output_dict=True),
        "confusion_matrix": cm.tolist(),
    }


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] device={device}")

    scaler = np.load(SCALER)
    mean = scaler["mean"].astype(np.float32)
    scale = scaler["scale"].astype(np.float32)
    scale[scale == 0] = 1.0

    model = Stage3EnsembleClassifier(num_classes=len(CLASSES))
    model.load_state_dict(torch.load(WEIGHTS, map_location=device, weights_only=True))
    model.to(device)

    buckets = collect_domain_samples()
    result = {"weights": str(WEIGHTS), "data_root": str(DATA_ROOT), "domains": {}}
    for domain, samples in buckets.items():
        print(f"\n[INFO] domain={domain} n={len(samples)}")
        if not samples:
            continue
        raw_feats = extract_features(samples, OUT.parent / f"domain_gap_{domain}_features.npy")
        feats = (raw_feats - mean) / scale
        loader = DataLoader(EvalDataset(samples, feats), batch_size=32, shuffle=False)
        metrics = evaluate(model, loader, device)
        metrics["samples"] = len(samples)
        result["domains"][domain] = metrics
        print(f"  accuracy={metrics['accuracy']*100:.2f}% macro_f1={metrics['macro_f1']:.4f}")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"\n[SUCCESS] wrote {OUT}")


if __name__ == "__main__":
    main()
