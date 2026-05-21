"""Prepare GINI as binary full-image classification in YOLO-style folders.

Class 0: background / non-garbage image.
Class 1: garbage image.

Each image receives one full-image pseudo box so the existing feature ML and
ANN/CNN crop-classification scripts can run. This is not object detection
evidence; it is a binary dataset-level classification baseline.
"""

from __future__ import annotations

import argparse
import csv
import random
import shutil
from pathlib import Path

import cv2
import yaml


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def find_image(root: Path, rel_or_name: str) -> Path | None:
    p = root / rel_or_name
    if p.exists():
        return p
    name = Path(rel_or_name).name
    hits = list(root.rglob(name))
    return hits[0] if hits else None


def split_name(i: int, n: int, train_ratio: float, valid_ratio: float) -> str:
    if i < int(round(n * train_ratio)):
        return "train"
    if i < int(round(n * (train_ratio + valid_ratio))):
        return "valid"
    return "test"


def read_csv_images(path: Path, require_box: bool) -> list[str]:
    out: list[str] = []
    with path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            image = row.get("image", "").strip()
            if not image:
                continue
            if require_box and not all(row.get(k, "").strip() for k in ("startX", "startY", "endX", "endY")):
                continue
            out.append(image)
    return out


def copy_items(items: list[tuple[Path, int]], out: Path, seed: int, train_ratio: float, valid_ratio: float) -> dict:
    rng = random.Random(seed)
    rng.shuffle(items)
    counts = {"train": [0, 0], "valid": [0, 0], "test": [0, 0]}
    for i, (src, cid) in enumerate(items):
        img = cv2.imread(str(src))
        if img is None:
            continue
        split = split_name(i, len(items), train_ratio, valid_ratio)
        out_name = f"{'garbage' if cid == 1 else 'background'}__{i:05d}{src.suffix.lower()}"
        out_img = out / split / "images" / out_name
        out_lbl = out / split / "labels" / Path(out_name).with_suffix(".txt").name
        out_img.parent.mkdir(parents=True, exist_ok=True)
        out_lbl.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, out_img)
        out_lbl.write_text(f"{cid} 0.5 0.5 1.0 1.0\n", encoding="utf-8")
        counts[split][cid] += 1
    return counts


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--max-per-class", type=int, default=1200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--valid-ratio", type=float, default=0.1)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if args.out.exists() and args.overwrite:
        shutil.rmtree(args.out)
    args.out.mkdir(parents=True, exist_ok=True)

    root = args.source / "spotgarbage"
    positives = read_csv_images(root / "garbage-queried-images.csv", require_box=True)
    negatives = read_csv_images(root / "non-garbage-queried-images.csv", require_box=False)
    rng = random.Random(args.seed)
    rng.shuffle(positives)
    rng.shuffle(negatives)
    positives = positives[: args.max_per_class]
    negatives = negatives[: args.max_per_class]

    items: list[tuple[Path, int]] = []
    for image in positives:
        p = find_image(root / "garbage-queried-images", image)
        if p is not None:
            items.append((p, 1))
    for image in negatives:
        p = find_image(root / "non-garbage-queried-images", image)
        if p is not None:
            items.append((p, 0))

    counts = copy_items(items, args.out, args.seed, args.train_ratio, args.valid_ratio)
    data = {
        "path": str(args.out.resolve()),
        "train": "train/images",
        "val": "valid/images",
        "test": "test/images",
        "nc": 2,
        "names": ["background", "garbage"],
    }
    (args.out / "data.yaml").write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    lines = [
        "# Prepared GINI Binary Classification Dataset",
        "",
        f"- Source: `{args.source.resolve()}`",
        f"- Output: `{args.out.resolve()}`",
        "- Conversion: full-image pseudo boxes for binary classification.",
        "- This is not material sorting; it is garbage-vs-background.",
        "",
        "| Split | Background | Garbage |",
        "|---|---:|---:|",
    ]
    for split in ("train", "valid", "test"):
        lines.append(f"| {split} | {counts[split][0]} | {counts[split][1]} |")
    (args.out / "PREPARE_REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"[OK] Prepared {args.out}")


if __name__ == "__main__":
    main()
