"""Benchmark a ConvNeXtV2 Stage 1 material-classifier candidate.

This is a fast, evidence-first benchmark:
- same hard-case train/val/test split as current Stage 1
- frozen ConvNeXtV2-Tiny image embeddings
- existing 637 handcrafted features
- sklearn MLP head
- CPU latency comparison against the deployed ConvNeXt+637 model

It does not promote anything. It writes numbers only.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import cv2
import joblib
import numpy as np
import timm
import torch
import torch.nn as nn
import torchvision.models as tv_models
import torchvision.transforms as transforms
from PIL import Image
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT / "scripts"
sys.path.append(str(SCRIPTS_DIR))
sys.path.append(str(SCRIPTS_DIR / "archive"))
from train_convnext_hardcase_tuned import (  # noqa: E402
    CLASSES,
    Sample,
    balance_train,
    load_or_extract_features,
    load_split,
    read_data_yaml,
)

DEFAULT_DATA = ROOT / "data" / "hard_case_classifier_v1" / "data.yaml"
DEFAULT_OUT = ROOT / "runs" / "dl" / "convnextv2_material_stage1_benchmark"


def load_bgr(path: Path) -> np.ndarray:
    image = cv2.imread(str(path))
    if image is None:
        raise ValueError(f"failed to read image: {path}")
    return image


def extract_convnextv2_features(samples: list[Sample], cache_path: Path, batch_size: int, device: torch.device) -> np.ndarray:
    if cache_path.exists():
        cached = np.load(cache_path)
        if cached.shape[0] == len(samples):
            print(f"[INFO] Loading cached ConvNeXtV2 features: {cache_path}")
            return cached.astype(np.float32)
        print(f"[WARN] Ignoring stale feature cache {cache_path}: {cached.shape} vs {len(samples)}")

    model = timm.create_model("convnextv2_tiny.fcmae_ft_in22k_in1k", pretrained=True, num_classes=0)
    model.to(device)
    model.eval()

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    features: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(samples), batch_size):
            batch_samples = samples[start : start + batch_size]
            tensors = []
            for sample in batch_samples:
                image = load_bgr(sample.path)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                tensors.append(transform(Image.fromarray(image)))
            batch = torch.stack(tensors).to(device)
            output = model(batch)
            features.append(output.detach().cpu().numpy().astype(np.float32))
            if start % (batch_size * 20) == 0:
                print(f"[INFO] ConvNeXtV2 features {start}/{len(samples)}")

    arr = np.concatenate(features, axis=0)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(cache_path, arr)
    return arr


def labels(samples: list[Sample]) -> np.ndarray:
    return np.array([sample.label for sample in samples], dtype=np.int64)


def benchmark_latency(model, deployed_predict_fn, samples: list[Sample], limit: int) -> dict[str, float]:
    selected = samples[:limit]

    candidate_times = []
    for sample in selected:
        image = load_bgr(sample.path)
        started = time.perf_counter()
        deployed_predict_fn(model, image)
        candidate_times.append((time.perf_counter() - started) * 1000)

    return {
        "images": len(selected),
        "mean_ms": float(np.mean(candidate_times)),
        "median_ms": float(np.median(candidate_times)),
        "p95_ms": float(np.percentile(candidate_times, 95)),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--train-max-per-class", type=int, default=1800)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--latency-images", type=int, default=120)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    cfg = read_data_yaml(args.data)
    data_root = Path(cfg.get("path", args.data.parent))
    if not data_root.is_absolute():
        data_root = args.data.parent / data_root

    train_raw = load_split(data_root, "train")
    val_samples = load_split(data_root, "val")
    test_samples = load_split(data_root, "test")
    train_samples = balance_train(train_raw, args.train_max_per_class, args.seed)

    print(f"[INFO] train/val/test: {len(train_samples)}/{len(val_samples)}/{len(test_samples)}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] device: {device}")

    train_deep = extract_convnextv2_features(train_samples, args.out / "train_convnextv2.npy", args.batch_size, device)
    val_deep = extract_convnextv2_features(val_samples, args.out / "val_convnextv2.npy", args.batch_size, device)
    test_deep = extract_convnextv2_features(test_samples, args.out / "test_convnextv2.npy", args.batch_size, device)

    train_hand = load_or_extract_features(train_samples, args.out / "train_handcrafted_637.npy", "train")
    val_hand = load_or_extract_features(val_samples, args.out / "val_handcrafted_637.npy", "val")
    test_hand = load_or_extract_features(test_samples, args.out / "test_handcrafted_637.npy", "test")

    x_train = np.concatenate([train_deep, train_hand], axis=1).astype(np.float32)
    x_val = np.concatenate([val_deep, val_hand], axis=1).astype(np.float32)
    x_test = np.concatenate([test_deep, test_hand], axis=1).astype(np.float32)
    y_train = labels(train_samples)
    y_val = labels(val_samples)
    y_test = labels(test_samples)

    head = make_pipeline(
        StandardScaler(),
        MLPClassifier(
            hidden_layer_sizes=(512, 128),
            activation="relu",
            alpha=1e-4,
            batch_size=256,
            learning_rate_init=1e-3,
            max_iter=80,
            early_stopping=True,
            validation_fraction=0.12,
            n_iter_no_change=10,
            random_state=args.seed,
            verbose=True,
        ),
    )
    started = time.perf_counter()
    head.fit(x_train, y_train)
    train_seconds = time.perf_counter() - started
    joblib.dump(head, args.out / "convnextv2_637_mlp_head.joblib")

    results = {
        "data": str(args.data),
        "model": "convnextv2_tiny.fcmae_ft_in22k_in1k frozen + 637 features + MLP head",
        "train_samples": len(train_samples),
        "val_samples": len(val_samples),
        "test_samples": len(test_samples),
        "feature_dim": int(x_train.shape[1]),
        "train_seconds": round(train_seconds, 2),
        "splits": {},
    }
    for split_name, x, y in (("val", x_val, y_val), ("test", x_test, y_test)):
        pred = head.predict(x)
        results["splits"][split_name] = {
            "accuracy": float(accuracy_score(y, pred)),
            "macro_f1": float(f1_score(y, pred, average="macro", zero_division=0)),
            "weighted_f1": float(f1_score(y, pred, average="weighted", zero_division=0)),
            "classification_report": classification_report(y, pred, target_names=CLASSES, output_dict=True, zero_division=0),
        }

    def current_model_cpu() -> nn.Module:
        class ConvNeXtFeatureExtractor(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.backbone = tv_models.convnext_tiny(weights=None)
                self.backbone.classifier = nn.Identity()

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return torch.flatten(self.backbone(x), 1)

        class Stage3EnsembleClassifier(nn.Module):
            def __init__(self) -> None:
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
                    nn.Linear(64, len(CLASSES)),
                )

            def forward(self, image_tensor: torch.Tensor, handcrafted_tensor: torch.Tensor) -> torch.Tensor:
                deep = self.convnext_extractor(image_tensor)
                return self.classifier(torch.cat((deep, handcrafted_tensor), dim=1))

        model = Stage3EnsembleClassifier()
        weights = ROOT / "runs" / "dl" / "convnext_ensemble_tuned" / "best_convnext_ensemble_tuned.pth"
        model.load_state_dict(torch.load(weights, map_location="cpu", weights_only=True))
        model.eval()
        return model

    def latency_current(samples: list[Sample]) -> dict[str, float]:
        from train_convnext_hardcase_tuned import load_bgr as load_sample_bgr

        scaler = np.load(ROOT / "runs" / "dl" / "convnext_ensemble_tuned" / "handcrafted_scaler.npz")
        scale = scaler["scale"].astype("float32")
        scale[scale == 0] = 1.0
        model = current_model_cpu()
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        times = []
        with torch.no_grad():
            for sample in samples:
                image = load_sample_bgr(sample)
                started = time.perf_counter()
                resized = cv2.resize(image, (224, 224), interpolation=cv2.INTER_CUBIC)
                rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
                image_tensor = transform(Image.fromarray(rgb)).unsqueeze(0)
                hand = load_or_extract_features([sample], args.out / "_latency_one.npy", "_latency")[0]
                hand = ((hand - scaler["mean"].astype("float32")) / scale).astype(np.float32)
                hand_tensor = torch.from_numpy(hand).unsqueeze(0)
                _ = model(image_tensor, hand_tensor).argmax(dim=1)
                times.append((time.perf_counter() - started) * 1000)
                if (args.out / "_latency_one.npy").exists():
                    (args.out / "_latency_one.npy").unlink()
        return {
            "images": len(samples),
            "mean_ms": float(np.mean(times)),
            "median_ms": float(np.median(times)),
            "p95_ms": float(np.percentile(times, 95)),
        }

    def latency_candidate(samples: list[Sample]) -> dict[str, float]:
        from train_convnext_hardcase_tuned import load_bgr as load_sample_bgr
        from custom_feature_extractor import extract_637_features

        deep_model = timm.create_model("convnextv2_tiny.fcmae_ft_in22k_in1k", pretrained=True, num_classes=0)
        deep_model.eval()
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        times = []
        with torch.no_grad():
            for sample in samples:
                image = load_sample_bgr(sample)
                started = time.perf_counter()
                rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                image_tensor = transform(Image.fromarray(rgb)).unsqueeze(0)
                deep = deep_model(image_tensor).detach().numpy().astype(np.float32)
                hand = extract_637_features(image).reshape(1, -1).astype(np.float32)
                _ = head.predict(np.concatenate([deep, hand], axis=1))
                times.append((time.perf_counter() - started) * 1000)
        return {
            "images": len(samples),
            "mean_ms": float(np.mean(times)),
            "median_ms": float(np.median(times)),
            "p95_ms": float(np.percentile(times, 95)),
        }

    latency_samples = test_samples[: args.latency_images]
    results["latency_cpu"] = {
        "current_convnext_637": latency_current(latency_samples),
        "candidate_convnextv2_637_mlp": latency_candidate(latency_samples),
    }

    (args.out / "RESULT.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    (args.out / "RESULT.md").write_text(
        "# ConvNeXtV2 Stage 1 Candidate Benchmark\n\n"
        f"- Model: {results['model']}\n"
        f"- Train / val / test: {len(train_samples)} / {len(val_samples)} / {len(test_samples)}\n"
        f"- Feature dim: {results['feature_dim']}\n"
        f"- Val accuracy: {results['splits']['val']['accuracy']:.2%}\n"
        f"- Val macro F1: {results['splits']['val']['macro_f1']:.4f}\n"
        f"- Test accuracy: {results['splits']['test']['accuracy']:.2%}\n"
        f"- Test macro F1: {results['splits']['test']['macro_f1']:.4f}\n",
        encoding="utf-8",
    )
    print(json.dumps(results, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
