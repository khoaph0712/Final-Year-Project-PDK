#!/usr/bin/env python
"""Cross-dataset (domain-shift) generalization validation.

The merged crop dataset mixes two acquisition domains under the same class
folders:

* STUDIO  - curated/lab images (TrashNet + Kaggle Garbage Classification).
            Filenames look like ``<material><index>.jpg`` (e.g. ``plastic575.jpg``,
            ``white-glass555.jpg``, ``biological331.jpg``, ``shoes1688.jpg``).
* FIELD   - real-world detection exports (Roboflow / TACO). Filenames carry a
            split/source marker: ``_train_`` / ``_test_`` / ``_val_`` / ``rf_``.

The previous version used a hand-coded ``is_trashnet_file`` heuristic that only
recognised a subset of material names, so plastic/metal/paper/cardboard were all
routed into one domain and the FIELD test set ended up with just 3 of 7 classes
(reported 17.83% "accuracy" on a broken split).

This script splits by the acquisition-domain marker above, keeps all 7 classes
in both domains, and measures the in-domain vs cross-domain gap in both
directions using a repeatable MLP over the cached 637-D handcrafted features.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

SCRIPTS_DIR = Path(__file__).resolve().parent
ROOT = SCRIPTS_DIR.parent
import sys

sys.path.append(str(SCRIPTS_DIR / "archive"))
from custom_feature_extractor import extract_637_features  # noqa: E402

DEFAULT_DATA_ROOT = ROOT / "data" / "merged_dataset_v5"
OUT_DIR = ROOT / "runs" / "dl" / "cross_dataset_validation"
TARGET_CLASSES = ["plastic", "glass", "metal", "paper", "cardboard", "organic", "Background"]
IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# Roboflow/TACO field-export markers. Anything without one is treated as STUDIO.
FIELD_RE = re.compile(r"(_train_|_test_|_val_|_valid_|rf_)", re.IGNORECASE)


class WasteMLP(nn.Module):
    def __init__(self, input_dim: int = 637, num_classes: int = 7):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, num_classes),
        )

    def forward(self, x):
        return self.net(x)


def domain_of(filename: str) -> str:
    return "field" if FIELD_RE.search(filename) else "studio"


def collect_domain_features(
    data_root: Path, cap_per_class: int, seed: int
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Walk train/ and test/ crop folders, split by acquisition domain, extract
    637-D features. Returns {domain: (X, y)}."""
    rng = np.random.default_rng(seed)
    buckets: dict[str, dict[int, list[Path]]] = {
        "studio": {i: [] for i in range(len(TARGET_CLASSES))},
        "field": {i: [] for i in range(len(TARGET_CLASSES))},
    }

    for split in ("train", "test"):
        for cls_idx, cls_name in enumerate(TARGET_CLASSES):
            cls_dir = data_root / split / cls_name
            if not cls_dir.exists():
                continue
            for f in cls_dir.iterdir():
                if f.suffix.lower() in IMG_EXT:
                    buckets[domain_of(f.name)][cls_idx].append(f)

    result: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for domain, per_class in buckets.items():
        feats: list[np.ndarray] = []
        labels: list[int] = []
        print(f"\n[INFO] Domain {domain.upper()} - extracting features:")
        for cls_idx, cls_name in enumerate(TARGET_CLASSES):
            paths = per_class[cls_idx]
            rng.shuffle(paths)
            paths = paths[:cap_per_class]
            count = 0
            for p in paths:
                img = cv2.imread(str(p))
                if img is None:
                    continue
                feats.append(extract_637_features(img))
                labels.append(cls_idx)
                count += 1
            print(f"  * {cls_name:<10}: {count} crops")
        result[domain] = (
            np.asarray(feats, dtype=np.float32),
            np.asarray(labels, dtype=np.int64),
        )
    return result


def train_mlp(x_train, y_train, device, epochs, seed) -> WasteMLP:
    torch.manual_seed(seed)
    model = WasteMLP(input_dim=x_train.shape[1], num_classes=len(TARGET_CLASSES)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
    loader = DataLoader(
        TensorDataset(torch.tensor(x_train), torch.tensor(y_train)),
        batch_size=64,
        shuffle=True,
    )
    model.train()
    for epoch in range(1, epochs + 1):
        running = 0.0
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
            running += loss.item() * xb.size(0)
        if epoch % 5 == 0 or epoch == epochs:
            print(f"    epoch {epoch:02d}/{epochs} | loss {running / len(x_train):.4f}")
    return model


def predict(model, x, device) -> np.ndarray:
    model.eval()
    preds: list[int] = []
    loader = DataLoader(TensorDataset(torch.tensor(x)), batch_size=128, shuffle=False)
    with torch.no_grad():
        for (xb,) in loader:
            out = model(xb.to(device))
            preds.extend(out.argmax(1).cpu().numpy())
    return np.asarray(preds)


def run_direction(
    train_domain: str,
    test_domain: str,
    data: dict[str, tuple[np.ndarray, np.ndarray]],
    device,
    epochs: int,
    seed: int,
) -> dict:
    """Train on train_domain (80% holdout for in-domain check), test on the held-out
    train_domain split (in-domain) and on the whole test_domain (cross-domain)."""
    x_all, y_all = data[train_domain]
    x_tr, x_hold, y_tr, y_hold = train_test_split(
        x_all, y_all, test_size=0.2, random_state=seed, stratify=y_all
    )
    x_cross, y_cross = data[test_domain]

    scaler = StandardScaler().fit(x_tr)
    x_tr_s = scaler.transform(x_tr).astype(np.float32)
    x_hold_s = scaler.transform(x_hold).astype(np.float32)
    x_cross_s = scaler.transform(x_cross).astype(np.float32)

    print(f"\n[INFO] Training MLP on {train_domain.upper()} ({len(x_tr)} samples)...")
    model = train_mlp(x_tr_s, y_tr, device, epochs, seed)

    in_pred = predict(model, x_hold_s, device)
    cross_pred = predict(model, x_cross_s, device)
    in_acc = accuracy_score(y_hold, in_pred)
    cross_acc = accuracy_score(y_cross, cross_pred)

    print(f"[RESULT] {train_domain}->{train_domain} (in-domain):    {in_acc * 100:.2f}%")
    print(f"[RESULT] {train_domain}->{test_domain} (cross-domain): {cross_acc * 100:.2f}%")
    print(f"[RESULT] generalization gap: {(in_acc - cross_acc) * 100:.2f} pp")

    return {
        "train_domain": train_domain,
        "test_domain": test_domain,
        "in_domain_accuracy": float(in_acc),
        "cross_domain_accuracy": float(cross_acc),
        "generalization_gap_pp": float((in_acc - cross_acc) * 100),
        "in_domain_macro_f1": float(f1_score(y_hold, in_pred, average="macro")),
        "cross_domain_macro_f1": float(f1_score(y_cross, cross_pred, average="macro")),
        "cross_domain_report": classification_report(
            y_cross, cross_pred, target_names=TARGET_CLASSES, zero_division=0
        ),
        "n_train": int(len(x_tr)),
        "n_in_domain_test": int(len(x_hold)),
        "n_cross_domain_test": int(len(x_cross)),
    }


def write_report(out_dir: Path, results: list[dict], counts: dict[str, int]) -> None:
    lines = [
        "# Cross-Dataset Generalizability Validation Report",
        "",
        "Domain-shift evaluation across two acquisition domains that share the same "
        "7 material classes:",
        "",
        "* **STUDIO** - curated/lab images (TrashNet + Kaggle Garbage Classification).",
        "* **FIELD**  - real-world detection exports (Roboflow / TACO).",
        "",
        "Domains are separated by the Roboflow/TACO filename markers "
        "(`_train_`/`_test_`/`_val_`/`rf_`); everything else is STUDIO. Both domains "
        "keep all 7 classes, so the earlier broken 3-class / 17.83% result is fixed.",
        "",
        f"* STUDIO samples used: **{counts['studio']}**",
        f"* FIELD samples used:  **{counts['field']}**",
        "",
        "## Results",
        "",
        "| Direction | In-domain acc | Cross-domain acc | Gap (pp) | Cross macro-F1 |",
        "|---|---:|---:|---:|---:|",
    ]
    for r in results:
        lines.append(
            f"| {r['train_domain']} -> {r['test_domain']} | "
            f"{r['in_domain_accuracy'] * 100:.2f}% | "
            f"{r['cross_domain_accuracy'] * 100:.2f}% | "
            f"{r['generalization_gap_pp']:.2f} | "
            f"{r['cross_domain_macro_f1']:.4f} |"
        )
    lines.append("")
    lines.append(
        "The gap is the honest cost of domain shift: how much accuracy drops when the "
        "model meets an acquisition domain it was not trained on."
    )
    for r in results:
        lines += [
            "",
            f"### {r['train_domain'].upper()} -> {r['test_domain'].upper()} cross-domain report",
            "",
            "```text",
            r["cross_domain_report"].rstrip(),
            "```",
        ]
    (out_dir / "Cross_Dataset_Report.md").write_text("\n".join(lines), encoding="utf-8")
    json_results = [{k: v for k, v in r.items() if k != "cross_domain_report"} for r in results]
    (out_dir / "cross_dataset_results.json").write_text(
        json.dumps(json_results, indent=2), encoding="utf-8"
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    p.add_argument("--out", type=Path, default=OUT_DIR)
    p.add_argument("--cap-per-class", type=int, default=700)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 60)
    print("Cross-Dataset Domain-Shift Validation (STUDIO vs FIELD)")
    print("=" * 60)
    print(f"[INFO] Device: {device}")

    data = collect_domain_features(args.data_root, args.cap_per_class, args.seed)
    counts = {d: int(len(y)) for d, (x, y) in data.items()}
    for d in ("studio", "field"):
        if counts.get(d, 0) == 0:
            print(f"[ERROR] Domain {d} is empty. Check dataset path/markers.")
            return

    results = [
        run_direction("studio", "field", data, device, args.epochs, args.seed),
        run_direction("field", "studio", data, device, args.epochs, args.seed),
    ]

    write_report(args.out, results, counts)
    print(f"\n[SUCCESS] Report written to {args.out / 'Cross_Dataset_Report.md'}")


if __name__ == "__main__":
    main()
