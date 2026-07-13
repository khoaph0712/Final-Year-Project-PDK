"""Export source-aware TACO hard cases to YOLO detection format."""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TACO = ROOT / "external_datasets" / "hard_case_full" / "taco_official"
DEFAULT_SPLIT = ROOT / "external_datasets" / "hard_case_full" / "source_aware_split_manifest.csv"
DEFAULT_OUT = ROOT / "external_datasets" / "hard_case_yolo_taco_v1"

YOLO_CLASSES = ["plastic", "glass", "metal", "paper", "cardboard", "organic"]
SPLIT_MAP = {
    "yolo_train_candidate": "train",
    "yolo_hard_val": "val",
    "yolo_hard_test": "test",
}
TACO_MATERIAL_KEYWORDS = {
    "plastic": [
        "plastic",
        "bottle cap",
        "straw",
        "lid",
        "wrapper",
        "sachet",
        "rope",
        "styrofoam",
    ],
    "glass": ["glass"],
    "metal": ["aluminium", "aluminum", "metal", "can", "aerosol"],
    "paper": ["paper", "magazine", "newspaper", "receipt", "paper bag"],
    "cardboard": ["carton", "cardboard", "corrugated"],
    "organic": ["food waste"],
}


def taco_class_to_material(name: str) -> str | None:
    lower = name.lower()
    for material, keywords in TACO_MATERIAL_KEYWORDS.items():
        if any(keyword in lower for keyword in keywords):
            return material
    return None


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


def load_split_lookup(path: Path) -> dict[str, tuple[str, str]]:
    lookup: dict[str, tuple[str, str]] = {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if row.get("source_id") != "taco_official":
                continue
            split = SPLIT_MAP.get(row.get("recommended_split", ""))
            if not split:
                continue
            lookup[row["source_file"].replace("\\", "/")] = (split, row.get("image_path", ""))
    return lookup


def yolo_line(bbox: list[float], width: int, height: int, class_id: int) -> str | None:
    x, y, w, h = bbox
    if w <= 1 or h <= 1 or width <= 0 or height <= 0:
        return None
    cx = (x + w / 2.0) / width
    cy = (y + h / 2.0) / height
    nw = w / width
    nh = h / height
    values = [max(0.0, min(1.0, value)) for value in (cx, cy, nw, nh)]
    if values[2] <= 0 or values[3] <= 0:
        return None
    return f"{class_id} {values[0]:.6f} {values[1]:.6f} {values[2]:.6f} {values[3]:.6f}"


def write_data_yaml(out_dir: Path) -> None:
    names = "\n".join(f"- {name}" for name in YOLO_CLASSES)
    (out_dir / "data.yaml").write_text(
        f"path: {out_dir}\n"
        "train: train/images\n"
        "val: val/images\n"
        "test: test/images\n"
        f"nc: {len(YOLO_CLASSES)}\n"
        "names:\n"
        f"{names}\n",
        encoding="utf-8",
    )


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--taco-dir", type=Path, default=DEFAULT_TACO)
    parser.add_argument("--split-manifest", type=Path, default=DEFAULT_SPLIT)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = args.out_dir
    if out_dir.exists() and not args.overwrite:
        raise SystemExit(f"output exists; pass --overwrite to rebuild: {out_dir}")
    reset_output(out_dir)

    annotations = json.loads((args.taco_dir / "data" / "annotations.json").read_text(encoding="utf-8"))
    categories = {item["id"]: item["name"] for item in annotations["categories"]}
    images = {item["id"]: item for item in annotations["images"]}
    anns_by_image: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for ann in annotations["annotations"]:
        anns_by_image[ann["image_id"]].append(ann)

    split_lookup = load_split_lookup(args.split_manifest)
    exported_rows: list[dict[str, Any]] = []
    class_counts: Counter[str] = Counter()
    split_counts: Counter[str] = Counter()
    skipped_no_mapped = 0
    skipped_missing_image = 0

    for image_id, image in images.items():
        source_file = image["file_name"].replace("\\", "/")
        split_info = split_lookup.get(source_file)
        if not split_info:
            continue
        split, image_rel = split_info
        image_path = ROOT / image_rel
        if not image_path.exists():
            skipped_missing_image += 1
            continue

        lines: list[str] = []
        mapped_labels: list[str] = []
        ignored_labels: list[str] = []
        for ann in anns_by_image.get(image_id, []):
            category = categories[ann["category_id"]]
            material = taco_class_to_material(category)
            if not material:
                ignored_labels.append(category)
                continue
            class_id = YOLO_CLASSES.index(material)
            line = yolo_line(ann["bbox"], int(image["width"]), int(image["height"]), class_id)
            if line:
                lines.append(line)
                mapped_labels.append(material)
                class_counts[material] += 1

        if not lines:
            skipped_no_mapped += 1
            continue

        target_name = f"taco_{image_id:05d}{image_path.suffix.lower() or '.jpg'}"
        target_image = out_dir / split / "images" / target_name
        target_label = out_dir / split / "labels" / Path(target_name).with_suffix(".txt").name
        method = link_or_copy(image_path, target_image)
        target_label.write_text("\n".join(lines) + "\n", encoding="utf-8")
        split_counts[split] += 1
        exported_rows.append(
            {
                "source_file": source_file,
                "source_image_path": image_rel,
                "target_image": str(target_image.relative_to(ROOT)),
                "target_label": str(target_label.relative_to(ROOT)),
                "split": split,
                "annotation_count": len(lines),
                "mapped_labels": "|".join(sorted(set(mapped_labels))),
                "ignored_labels": "|".join(sorted(set(ignored_labels))),
                "link_method": method,
            }
        )

    write_data_yaml(out_dir)
    write_csv(out_dir / "source_manifest.csv", exported_rows)
    summary = {
        "images_exported": len(exported_rows),
        "split_counts": dict(split_counts),
        "box_counts_by_class": dict(class_counts),
        "skipped_no_mapped_labels": skipped_no_mapped,
        "skipped_missing_image": skipped_missing_image,
        "classes": YOLO_CLASSES,
        "data_yaml": str((out_dir / "data.yaml").relative_to(ROOT)),
    }
    (out_dir / "SUMMARY.json").write_text(json.dumps(summary, indent=2, ensure_ascii=True), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
