"""Convert class-folder image datasets into project YOLO-style crops.

For classification-only datasets such as TrashNet, each image is treated as one
object crop covering the full image. This is not detection evidence; it is a
classification/feature baseline with clean-background images.
"""

from __future__ import annotations

import argparse
import random
import shutil
from pathlib import Path

import cv2
import yaml


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def find_class_dirs(root: Path) -> list[Path]:
    return sorted([p for p in root.iterdir() if p.is_dir() and any(x.suffix.lower() in IMAGE_EXTS for x in p.rglob("*"))])


def safe_copy(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--valid-ratio", type=float, default=0.1)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if args.out.exists() and args.overwrite:
        shutil.rmtree(args.out)
    args.out.mkdir(parents=True, exist_ok=True)

    class_dirs = find_class_dirs(args.source)
    if not class_dirs:
        raise SystemExit(f"No class folders with images found under {args.source}")
    names = [p.name.lower().replace(" ", "_") for p in class_dirs]

    rng = random.Random(args.seed)
    split_counts = {"train": 0, "valid": 0, "test": 0}
    class_counts = {split: {name: 0 for name in names} for split in split_counts}

    for cid, class_dir in enumerate(class_dirs):
        images = sorted([p for p in class_dir.rglob("*") if p.suffix.lower() in IMAGE_EXTS])
        rng.shuffle(images)
        n = len(images)
        n_train = int(round(n * args.train_ratio))
        n_valid = int(round(n * args.valid_ratio))
        assignments = (
            [("train", p) for p in images[:n_train]]
            + [("valid", p) for p in images[n_train : n_train + n_valid]]
            + [("test", p) for p in images[n_train + n_valid :]]
        )
        for split, img_path in assignments:
            img = cv2.imread(str(img_path))
            if img is None:
                continue
            out_name = f"{names[cid]}__{img_path.stem}{img_path.suffix.lower()}"
            out_img = args.out / split / "images" / out_name
            out_lbl = args.out / split / "labels" / Path(out_name).with_suffix(".txt").name
            safe_copy(img_path, out_img)
            out_lbl.parent.mkdir(parents=True, exist_ok=True)
            out_lbl.write_text(f"{cid} 0.5 0.5 1.0 1.0\n", encoding="utf-8")
            split_counts[split] += 1
            class_counts[split][names[cid]] += 1

    data = {
        "path": str(args.out.resolve()),
        "train": "train/images",
        "val": "valid/images",
        "test": "test/images",
        "nc": len(names),
        "names": names,
    }
    (args.out / "data.yaml").write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")

    report = [
        "# Prepared Classification Dataset",
        "",
        f"- Source: `{args.source.resolve()}`",
        f"- Output: `{args.out.resolve()}`",
        "- Conversion: each classification image becomes one full-image YOLO box.",
        "- Warning: this is classification evidence, not object-detection annotation evidence.",
        "",
        "## Split Counts",
        "",
        "| Split | Images |",
        "|---|---:|",
    ]
    for split, count in split_counts.items():
        report.append(f"| {split} | {count} |")
    report.extend(["", "## Class Counts"])
    for split, counts in class_counts.items():
        report.append(f"### {split}")
        for name, count in counts.items():
            report.append(f"- {name}: {count}")
    (args.out / "PREPARE_REPORT.md").write_text("\n".join(report), encoding="utf-8")
    print(f"[OK] Prepared {args.out}")


if __name__ == "__main__":
    main()
