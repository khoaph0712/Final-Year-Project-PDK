"""Export Trashify boxes into a crop classifier dataset for Stage 0.

The source dataset labels object-state concepts, not material classes:
bin, hand, not_bin, not_hand, not_trash, trash, trash_arm.

For WasteWise Stage 0 we keep only clear gate classes:
  trash      <- trash, trash_arm
  not_trash  <- not_trash
  hand       <- hand
  bin        <- bin

Ambiguous `not_bin` and `not_hand` boxes are skipped by default because they
are negative annotations for a detector, not stable material/gate classes.
"""

from __future__ import annotations

import argparse
import csv
import random
import shutil
from collections import Counter
from pathlib import Path
from typing import Any

from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = ROOT / "data" / "stage0_trashify_v1"
DATASET_ID = "mrdbourke/trashify_manual_labelled_images"
STAGE0_CLASSES = ["trash", "not_trash", "hand", "bin"]
LABEL_MAP = {
    "trash": "trash",
    "trash_arm": "trash",
    "not_trash": "not_trash",
    "hand": "hand",
    "bin": "bin",
}


def reset_output(path: Path) -> None:
    resolved = path.resolve()
    data_root = (ROOT / "data").resolve()
    if data_root not in resolved.parents:
        raise ValueError(f"refusing to clear output outside data/: {resolved}")
    if resolved.exists():
        shutil.rmtree(resolved)
    for split in ("train", "val", "test"):
        for class_name in STAGE0_CLASSES:
            (resolved / split / class_name).mkdir(parents=True, exist_ok=True)


def safe_crop(image: Image.Image, bbox: list[float], pad_ratio: float) -> Image.Image | None:
    width, height = image.size
    x, y, w, h = [float(value) for value in bbox]
    if w < 8 or h < 8:
        return None
    pad = max(w, h) * pad_ratio
    x1 = max(0, int(round(x - pad)))
    y1 = max(0, int(round(y - pad)))
    x2 = min(width, int(round(x + w + pad)))
    y2 = min(height, int(round(y + h + pad)))
    if x2 - x1 < 12 or y2 - y1 < 12:
        return None
    return image.crop((x1, y1, x2, y2)).convert("RGB")


def split_for_image(image_id: int, seed: int) -> str:
    rng = random.Random(f"{seed}:{image_id}")
    value = rng.random()
    if value < 0.80:
        return "train"
    if value < 0.90:
        return "val"
    return "test"


def load_categories(dataset: Any) -> list[str]:
    annotations = dataset["train"].features["annotations"]
    if isinstance(annotations, dict):
        return annotations["category_id"].feature.names
    return annotations.feature["category_id"].names


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--max-rows", type=int, default=0, help="0 means all rows.")
    parser.add_argument("--max-per-class", type=int, default=1200)
    parser.add_argument("--pad-ratio", type=float, default=0.06)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.out_dir.exists() and not args.overwrite:
        raise SystemExit(f"output exists; pass --overwrite to rebuild: {args.out_dir}")
    reset_output(args.out_dir)

    from datasets import load_dataset

    dataset = load_dataset(DATASET_ID)
    categories = load_categories(dataset)
    per_class = Counter()
    per_split = Counter()
    skipped = Counter()
    rows: list[dict[str, str]] = []

    for row_index, row in enumerate(dataset["train"]):
        if args.max_rows and row_index >= args.max_rows:
            break
        image = row["image"].convert("RGB")
        image_id = int(row["image_id"])
        split = split_for_image(image_id, args.seed)
        annotations = row["annotations"]
        for ann_index, category_id in enumerate(annotations["category_id"]):
            source_label = categories[int(category_id)]
            class_name = LABEL_MAP.get(source_label)
            if class_name is None:
                skipped[source_label] += 1
                continue
            if per_class[class_name] >= args.max_per_class:
                skipped[f"{class_name}:cap"] += 1
                continue

            crop = safe_crop(image, annotations["bbox"][ann_index], args.pad_ratio)
            if crop is None:
                skipped[f"{source_label}:bad_crop"] += 1
                continue

            file_name = f"trashify_{image_id:06d}_{ann_index:02d}_{source_label}.jpg"
            target = args.out_dir / split / class_name / file_name
            crop.save(target, quality=92)
            per_class[class_name] += 1
            per_split[(split, class_name)] += 1
            rows.append(
                {
                    "split": split,
                    "class": class_name,
                    "source_label": source_label,
                    "image_id": str(image_id),
                    "annotation_index": str(ann_index),
                    "target_path": str(target.relative_to(ROOT)),
                }
            )

    manifest = args.out_dir / "source_manifest.csv"
    with manifest.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["split", "class", "source_label", "image_id", "annotation_index", "target_path"])
        writer.writeheader()
        writer.writerows(rows)

    names = "\n".join(f"- {name}" for name in STAGE0_CLASSES)
    (args.out_dir / "data.yaml").write_text(
        f"path: {args.out_dir}\n"
        "type: classification\n"
        "task: stage0_trash_gate\n"
        "train: train\n"
        "val: val\n"
        "test: test\n"
        f"nc: {len(STAGE0_CLASSES)}\n"
        "names:\n"
        f"{names}\n",
        encoding="utf-8",
    )

    report = {
        "dataset": DATASET_ID,
        "out_dir": str(args.out_dir),
        "rows_exported": len(rows),
        "per_class": dict(per_class),
        "per_split": {f"{split}/{class_name}": count for (split, class_name), count in per_split.items()},
        "skipped": dict(skipped),
        "source_categories": categories,
    }
    (args.out_dir / "EXPORT_REPORT.json").write_text(__import__("json").dumps(report, indent=2), encoding="utf-8")
    print(__import__("json").dumps(report, indent=2))


if __name__ == "__main__":
    main()
