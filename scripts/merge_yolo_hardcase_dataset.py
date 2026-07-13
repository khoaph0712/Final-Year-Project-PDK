"""Merge the existing YOLO dataset with TACO hard cases for YOLO26 tests."""

from __future__ import annotations

import argparse
import csv
import hashlib
import os
import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BASE = ROOT / "external_datasets" / "super_yolo_dataset"
DEFAULT_TACO = ROOT / "external_datasets" / "hard_case_yolo_taco_v1"
DEFAULT_OUT = ROOT / "external_datasets" / "yolo26_hardcase_dataset_v1"
CLASSES = ["plastic", "glass", "metal", "paper", "cardboard", "organic"]


def reset_output(path: Path) -> None:
    external_root = ROOT / "external_datasets"
    if external_root not in path.parents:
        raise ValueError(f"refusing to clear output outside external_datasets/: {path}")
    if path.exists():
        for child in path.iterdir():
            shutil.rmtree(child) if child.is_dir() else child.unlink()
    for split in ("train", "val", "test"):
        (path / split / "images").mkdir(parents=True, exist_ok=True)
        (path / split / "labels").mkdir(parents=True, exist_ok=True)


def link_or_copy(source: Path, target: Path) -> str:
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        return "exists"
    try:
        os.link(source, target)
        return "hardlink"
    except OSError:
        shutil.copy2(source, target)
        return "copy"


def stable_name(prefix: str, path: Path) -> str:
    digest = hashlib.md5(str(path).encode("utf-8")).hexdigest()[:10]
    return f"{prefix}_{digest}{path.suffix.lower() or '.jpg'}"


def image_files(path: Path) -> list[Path]:
    return [
        item
        for item in path.iterdir()
        if item.is_file() and item.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    ]


def add_dataset(out_dir: Path, source_dir: Path, source_id: str) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for split in ("train", "val", "test"):
        image_dir = source_dir / split / "images"
        label_dir = source_dir / split / "labels"
        if not image_dir.exists():
            continue
        for image in image_files(image_dir):
            stem = Path(stable_name(source_id, image)).stem
            target_image = out_dir / split / "images" / f"{stem}{image.suffix.lower()}"
            target_label = out_dir / split / "labels" / f"{stem}.txt"
            label = label_dir / f"{image.stem}.txt"
            image_method = link_or_copy(image, target_image)
            if label.exists():
                label_method = link_or_copy(label, target_label)
            else:
                target_label.write_text("", encoding="utf-8")
                label_method = "empty"
            rows.append(
                {
                    "source_id": source_id,
                    "split": split,
                    "source_image": str(image.relative_to(ROOT)),
                    "source_label": str(label.relative_to(ROOT)) if label.exists() else "",
                    "target_image": str(target_image.relative_to(ROOT)),
                    "target_label": str(target_label.relative_to(ROOT)),
                    "image_method": image_method,
                    "label_method": label_method,
                }
            )
    return rows


def write_data_yaml(out_dir: Path) -> None:
    names = "\n".join(f"- {name}" for name in CLASSES)
    (out_dir / "data.yaml").write_text(
        f"path: {out_dir}\n"
        "train: train/images\n"
        "val: val/images\n"
        "test: test/images\n"
        f"nc: {len(CLASSES)}\n"
        "names:\n"
        f"{names}\n",
        encoding="utf-8",
    )


def write_manifest(out_dir: Path, rows: list[dict[str, str]]) -> None:
    fieldnames = sorted({key for row in rows for key in row})
    with (out_dir / "source_manifest.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def print_counts(out_dir: Path) -> None:
    for split in ("train", "val", "test"):
        image_count = len(image_files(out_dir / split / "images"))
        label_count = len(list((out_dir / split / "labels").glob("*.txt")))
        print(f"{split}: images={image_count} labels={label_count}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, default=DEFAULT_BASE)
    parser.add_argument("--taco", type=Path, default=DEFAULT_TACO)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = args.out_dir
    if out_dir.exists() and not args.overwrite:
        raise SystemExit(f"output exists; pass --overwrite to rebuild: {out_dir}")
    reset_output(out_dir)
    rows = []
    rows.extend(add_dataset(out_dir, args.base, "super_yolo"))
    rows.extend(add_dataset(out_dir, args.taco, "taco_hardcase"))
    write_data_yaml(out_dir)
    write_manifest(out_dir, rows)
    print(f"merged records: {len(rows)}")
    print(f"output: {out_dir}")
    print_counts(out_dir)


if __name__ == "__main__":
    main()
