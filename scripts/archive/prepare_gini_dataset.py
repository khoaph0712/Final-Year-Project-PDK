"""Prepare GINI as a YOLO-style binary/context dataset.

Annotated garbage-query rows become class 0 (`garbage`) boxes.
Non-garbage query images are copied with empty labels for EDA/context only.
Classical crop ML will use only labeled garbage boxes; the empty-label negatives
are retained to document that GINI is not a material-sorting dataset.
"""

from __future__ import annotations

import argparse
import csv
import random
import shutil
from collections import Counter
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--valid-ratio", type=float, default=0.1)
    parser.add_argument("--max-negative-images", type=int, default=500)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if args.out.exists() and args.overwrite:
        shutil.rmtree(args.out)
    args.out.mkdir(parents=True, exist_ok=True)

    root = args.source / "spotgarbage"
    garbage_csv = root / "garbage-queried-images.csv"
    non_csv = root / "non-garbage-queried-images.csv"

    labeled: list[dict] = []
    with garbage_csv.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if all(row.get(k, "").strip() for k in ("startX", "startY", "endX", "endY")):
                labeled.append(row)

    rng = random.Random(args.seed)
    rng.shuffle(labeled)
    counts = Counter()
    missing = 0
    for i, row in enumerate(labeled):
        src = find_image(root / "garbage-queried-images", row["image"])
        if src is None:
            missing += 1
            continue
        img = cv2.imread(str(src))
        if img is None:
            missing += 1
            continue
        h, w = img.shape[:2]
        x1 = float(row["startX"])
        y1 = float(row["startY"])
        x2 = float(row["endX"])
        y2 = float(row["endY"])
        bw = max(1.0, x2 - x1)
        bh = max(1.0, y2 - y1)
        cx = (x1 + bw / 2.0) / w
        cy = (y1 + bh / 2.0) / h
        line = f"0 {cx:.6f} {cy:.6f} {bw / w:.6f} {bh / h:.6f}"
        split = split_name(i, len(labeled), args.train_ratio, args.valid_ratio)
        out_name = f"garbage__{i:05d}{src.suffix.lower()}"
        out_img = args.out / split / "images" / out_name
        out_lbl = args.out / split / "labels" / Path(out_name).with_suffix(".txt").name
        out_img.parent.mkdir(parents=True, exist_ok=True)
        out_lbl.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, out_img)
        out_lbl.write_text(line + "\n", encoding="utf-8")
        counts[f"{split}_positive"] += 1

    negatives: list[str] = []
    with non_csv.open(newline="", encoding="utf-8") as f:
        negatives = [row["image"] for row in csv.DictReader(f) if row.get("image")]
    rng.shuffle(negatives)
    negatives = negatives[: args.max_negative_images]
    for i, img_name in enumerate(negatives):
        src = find_image(root / "non-garbage-queried-images", img_name)
        if src is None:
            continue
        split = split_name(i, len(negatives), args.train_ratio, args.valid_ratio)
        out_name = f"background__{i:05d}{src.suffix.lower()}"
        out_img = args.out / split / "images" / out_name
        out_lbl = args.out / split / "labels" / Path(out_name).with_suffix(".txt").name
        out_img.parent.mkdir(parents=True, exist_ok=True)
        out_lbl.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, out_img)
        out_lbl.write_text("", encoding="utf-8")
        counts[f"{split}_negative"] += 1

    data = {
        "path": str(args.out.resolve()),
        "train": "train/images",
        "val": "valid/images",
        "test": "test/images",
        "nc": 1,
        "names": ["garbage"],
    }
    (args.out / "data.yaml").write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    (args.out / "PREPARE_REPORT.md").write_text(
        "\n".join(
            [
                "# Prepared GINI Dataset",
                "",
                f"- Source: `{args.source.resolve()}`",
                "- Positive annotated rows converted to garbage boxes.",
                "- Non-garbage rows copied with empty labels for context/EDA.",
                "- Not suitable for standalone multi-class sorting.",
                f"- Missing/unreadable positive images: {missing}",
                "",
                "## Counts",
                *[f"- {k}: {v}" for k, v in sorted(counts.items())],
            ]
        ),
        encoding="utf-8",
    )
    print(f"[OK] Prepared {args.out}")


if __name__ == "__main__":
    main()
