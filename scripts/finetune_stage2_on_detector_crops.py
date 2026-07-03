#!/usr/bin/env python
"""Fine-tune the deployed ConvNeXt+637 classifier on the DETECTOR's own crops.

Fixes train/serve skew: the deployed stage-2 model trained on ground-truth crops
but serves on YOLO crops. Starts FROM the deployed weights, keeps the deployed
feature scaler (drop-in swap for the web server), trains at low LR on
data/detector_crops_v1, and promotes only if:
  (a) detector-crop val accuracy improves, AND
  (b) clean GT-crop test (hard_case_classifier_v1_clean) drops < 1pp (no forgetting).

Run: .venv311\\Scripts\\python.exe scripts/finetune_stage2_on_detector_crops.py
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "scripts" / "archive"))
from custom_feature_extractor import extract_637_features  # noqa: E402

CLASSES = ["plastic", "glass", "metal", "paper", "cardboard", "organic", "Background"]
CROPS = ROOT / "data" / "detector_crops_v1"
CLEAN_GT = ROOT / "data" / "hard_case_classifier_v1_clean"
DEPLOYED_WEIGHTS = ROOT / "runs" / "dl" / "convnext_ensemble_tuned" / "best_convnext_ensemble_tuned.pth"
DEPLOYED_SCALER = ROOT / "runs" / "dl" / "convnext_ensemble_tuned" / "handcrafted_scaler.npz"
OUT = ROOT / "runs" / "dl" / "convnext_detector_crops_ft"
IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


class ConvNeXtFeatureExtractor(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.backbone = models.convnext_tiny(weights=None)
        self.backbone.classifier = nn.Identity()

    def forward(self, x):
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

    def forward(self, image_tensor, handcrafted):
        deep = self.convnext_extractor(image_tensor)
        return self.classifier(torch.cat((deep, handcrafted), dim=1))


class CropDataset(Dataset):
    def __init__(self, samples: list[tuple[Path, int]], features: np.ndarray, train: bool) -> None:
        self.samples = samples
        self.features = features.astype(np.float32)
        aug = [transforms.RandomHorizontalFlip(), transforms.ColorJitter(0.2, 0.2, 0.1)] if train else []
        self.transform = transforms.Compose(
            aug + [transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]
        )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        path, label = self.samples[i]
        img = cv2.imread(str(path))
        img = cv2.cvtColor(cv2.resize(img, (224, 224), interpolation=cv2.INTER_CUBIC), cv2.COLOR_BGR2RGB)
        return (
            self.transform(Image.fromarray(img)),
            torch.tensor(self.features[i], dtype=torch.float32),
            torch.tensor(label, dtype=torch.long),
        )


def load_split(root: Path, split: str, cap: int | None, seed: int) -> list[tuple[Path, int]]:
    rng = random.Random(seed)
    out: list[tuple[Path, int]] = []
    for label, cname in enumerate(CLASSES):
        cdir = root / split / cname
        if not cdir.exists():
            continue
        paths = [p for p in cdir.iterdir() if p.suffix.lower() in IMG_EXT]
        rng.shuffle(paths)
        if cap:
            paths = paths[:cap]
        out.extend((p, label) for p in paths)
    rng.shuffle(out)
    return out


def features_for(samples: list[tuple[Path, int]], cache: Path, mean, scale) -> np.ndarray:
    if cache.exists():
        arr = np.load(cache)
        if arr.shape[0] == len(samples):
            return ((arr - mean) / scale).astype(np.float32)
    feats = []
    for i, (p, _) in enumerate(samples):
        if i % 2000 == 0:
            print(f"  features {i}/{len(samples)}")
        feats.append(extract_637_features(cv2.imread(str(p))))
    arr = np.array(feats, dtype=np.float32)
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.save(cache, arr)
    return ((arr - mean) / scale).astype(np.float32)


def evaluate(model, loader, device) -> tuple[float, float]:
    model.eval()
    trues, preds = [], []
    with torch.no_grad():
        for img, feat, y in loader:
            logits = model(img.to(device), feat.to(device))
            preds.extend(logits.argmax(1).cpu().tolist())
            trues.extend(y.tolist())
    return accuracy_score(trues, preds), f1_score(trues, preds, average="macro", zero_division=0)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--train-cap", type=int, default=3000)
    ap.add_argument("--mix-gt-per-class", type=int, default=0,
                    help="also mix in this many GT crops per class from hard_case_classifier_v1/train (anti-forgetting)")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    assert torch.cuda.is_available(), "CUDA required - use .venv311 python"
    device = torch.device("cuda")
    torch.manual_seed(args.seed)
    OUT.mkdir(parents=True, exist_ok=True)

    sc = np.load(DEPLOYED_SCALER)
    mean, scale = sc["mean"].astype(np.float32), sc["scale"].astype(np.float32)
    scale[scale == 0] = 1.0

    train_s = load_split(CROPS, "train", args.train_cap, args.seed)
    if args.mix_gt_per_class > 0:
        gt_train = load_split(ROOT / "data" / "hard_case_classifier_v1", "train", args.mix_gt_per_class, args.seed)
        train_s = train_s + gt_train
        random.Random(args.seed).shuffle(train_s)
        print(f"[INFO] mixed in {len(gt_train)} GT crops")
    val_s = load_split(CROPS, "val", None, args.seed)
    gt_s = load_split(CLEAN_GT, "test", None, args.seed)
    print(f"[INFO] train {len(train_s)} | detector-crop val {len(val_s)} | clean GT test {len(gt_s)}")

    x_train = features_for(train_s, OUT / "train_feats.npy", mean, scale)
    x_val = features_for(val_s, OUT / "val_feats.npy", mean, scale)
    x_gt = features_for(gt_s, OUT / "gt_feats.npy", mean, scale)

    dl = lambda s, x, train: DataLoader(
        CropDataset(s, x, train), batch_size=args.batch_size, shuffle=train, num_workers=2
    )
    train_loader = dl(train_s, x_train, True)
    val_loader = dl(val_s, x_val, False)
    gt_loader = dl(gt_s, x_gt, False)

    model = Stage3EnsembleClassifier(len(CLASSES))
    model.load_state_dict(torch.load(DEPLOYED_WEIGHTS, map_location=device, weights_only=True))
    model.to(device)

    base_val_acc, base_val_f1 = evaluate(model, val_loader, device)
    base_gt_acc, _ = evaluate(model, gt_loader, device)
    print(f"[BASELINE] detector-crop val acc {base_val_acc*100:.2f}% (F1 {base_val_f1:.4f}) | clean GT test {base_gt_acc*100:.2f}%")

    # freeze backbone except final stage; train head + final stage at low LR
    for p in model.convnext_extractor.backbone.parameters():
        p.requires_grad = False
    for p in model.convnext_extractor.backbone.features[7].parameters():
        p.requires_grad = True
    params = [p for p in model.parameters() if p.requires_grad]
    opt = optim.AdamW(params, lr=args.lr, weight_decay=0.01)
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    crit = nn.CrossEntropyLoss(label_smoothing=0.05)

    best_val, best_state = base_val_acc, None
    for epoch in range(1, args.epochs + 1):
        model.train()
        running = 0.0
        for img, feat, y in train_loader:
            opt.zero_grad()
            loss = crit(model(img.to(device), feat.to(device)), y.to(device))
            loss.backward()
            opt.step()
            running += loss.item() * img.size(0)
        sched.step()
        val_acc, val_f1 = evaluate(model, val_loader, device)
        print(f"  epoch {epoch}/{args.epochs} loss {running/len(train_s):.4f} | val acc {val_acc*100:.2f}% F1 {val_f1:.4f}")
        if val_acc > best_val:
            best_val = val_acc
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            torch.save(best_state, OUT / "best_detector_crops_ft.pth")

    result = {
        "baseline": {"detector_crop_val_acc": base_val_acc, "clean_gt_test_acc": base_gt_acc},
        "finetuned": None,
        "promote": False,
    }
    if best_state is not None:
        model.load_state_dict(best_state)
        ft_val_acc, ft_val_f1 = evaluate(model, val_loader, device)
        ft_gt_acc, _ = evaluate(model, gt_loader, device)
        result["finetuned"] = {
            "detector_crop_val_acc": ft_val_acc,
            "detector_crop_val_f1": ft_val_f1,
            "clean_gt_test_acc": ft_gt_acc,
        }
        result["promote"] = ft_val_acc > base_val_acc and ft_gt_acc >= base_gt_acc - 0.01
        print(
            f"[FINETUNED] val {ft_val_acc*100:.2f}% (base {base_val_acc*100:.2f}%) | "
            f"clean GT {ft_gt_acc*100:.2f}% (base {base_gt_acc*100:.2f}%) | promote={result['promote']}"
        )
    else:
        print("[RESULT] no epoch beat the baseline on detector-crop val; keep deployed weights")

    (OUT / "finetune_result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"[OK] {OUT / 'finetune_result.json'}")


if __name__ == "__main__":
    main()
