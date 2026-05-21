"""Build tuned_dataset_v2_realworld by adding a real-world Roboflow dataset.

This keeps tuned_dataset_v1 unchanged and creates a new YOLO dataset:
- copies/hardlinks all tuned_dataset_v1 images and labels;
- appends rf_realworld_trash_detection_ujrn0 with class remapping;
- writes data.yaml and class_support.json for evidence.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import shutil
from collections import Counter, defaultdict
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_BASE = ROOT / "tuned_dataset_v1"
DEFAULT_EXTRA = ROOT / "rf_realworld_trash_detection_ujrn0"
DEFAULT_OUT = ROOT / "tuned_dataset_v2_realworld"
TARGET_NAMES = ["plastic", "glass", "metal", "paper", "cardboard", "organic", "other"]
SPLITS = ("train", "valid", "test")
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def safe_link_or_copy(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        dst.unlink()
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def image_to_label_path(image_path: Path) -> Path:
    return Path(str(image_path).replace("\\images\\", "\\labels\\")).with_suffix(".txt")


def short_output_name(prefix: str, image_path: Path) -> str:
    digest = hashlib.md5(str(image_path).encode("utf-8")).hexdigest()[:10]
    stem = image_path.stem[:50].rstrip(" ._-")
    return f"{prefix}_{stem}_{digest}{image_path.suffix.lower()}"


def copy_yolo_split(
    src_root: Path,
    dst_root: Path,
    split: str,
    prefix: str,
    class_map: dict[int, int] | None,
    max_train_images_per_class: int | None = None,
    seed: int = 42,
) -> Counter[int]:
    counts: Counter[int] = Counter()
    img_dir = src_root / split / "images"
    if split == "valid" and not img_dir.exists():
        img_dir = src_root / "val" / "images"
    if not img_dir.exists():
        return counts

    candidates: list[tuple[Path, str, list[str], Counter[int], int]] = []
    for image_path in sorted(p for p in img_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS):
        label_path = image_to_label_path(image_path)
        if not label_path.exists():
            continue
        out_lines: list[str] = []
        image_counts: Counter[int] = Counter()
        for raw in label_path.read_text(encoding="utf-8").splitlines():
            parts = raw.strip().split()
            if len(parts) != 5:
                continue
            try:
                old_id = int(float(parts[0]))
            except ValueError:
                continue
            new_id = class_map.get(old_id) if class_map is not None else old_id
            if new_id is None or new_id < 0 or new_id >= len(TARGET_NAMES):
                continue
            out_lines.append(" ".join([str(new_id), *parts[1:]]))
            image_counts[new_id] += 1
        if not out_lines:
            continue
        out_name = short_output_name(prefix, image_path)
        dominant_class = image_counts.most_common(1)[0][0]
        candidates.append((image_path, out_name, out_lines, image_counts, dominant_class))

    if split == "train" and max_train_images_per_class is not None:
        grouped: dict[int, list[tuple[Path, str, list[str], Counter[int], int]]] = defaultdict(list)
        for candidate in candidates:
            grouped[candidate[4]].append(candidate)
        rng = random.Random(seed)
        selected = []
        for class_id, class_candidates in grouped.items():
            rng.shuffle(class_candidates)
            selected.extend(class_candidates[:max_train_images_per_class])
        candidates = sorted(selected, key=lambda item: item[1])

    for image_path, out_name, out_lines, image_counts, _dominant_class in candidates:
        counts.update(image_counts)
        safe_link_or_copy(image_path, dst_root / split / "images" / out_name)
        (dst_root / split / "labels" / Path(out_name).with_suffix(".txt").name).write_text(
            "\n".join(out_lines) + "\n",
            encoding="utf-8",
        )
    return counts


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path, default=DEFAULT_BASE)
    parser.add_argument("--extra", type=Path, default=DEFAULT_EXTRA)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--max-base-train-images-per-class", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.out.exists():
        if not args.overwrite:
            raise SystemExit(f"Output exists: {args.out}. Use --overwrite to rebuild.")
        shutil.rmtree(args.out)

    for split in SPLITS:
        (args.out / split / "images").mkdir(parents=True, exist_ok=True)
        (args.out / split / "labels").mkdir(parents=True, exist_ok=True)

    extra_yaml = yaml.safe_load((args.extra / "data.yaml").read_text(encoding="utf-8"))
    extra_names = list(extra_yaml["names"])
    name_to_target = {name: TARGET_NAMES.index(name) for name in TARGET_NAMES}
    extra_map = {i: name_to_target[name] for i, name in enumerate(extra_names) if name in name_to_target}

    support: dict[str, dict[str, int]] = defaultdict(dict)
    total_counts: Counter[int] = Counter()
    for split in SPLITS:
        base_counts = copy_yolo_split(
            args.base,
            args.out,
            split,
            "base",
            None,
            max_train_images_per_class=args.max_base_train_images_per_class,
            seed=args.seed,
        )
        extra_counts = copy_yolo_split(args.extra, args.out, split, "realworld", extra_map, seed=args.seed)
        split_counts = base_counts + extra_counts
        total_counts.update(split_counts)
        for cid, name in enumerate(TARGET_NAMES):
            support[split][name] = int(split_counts[cid])

    yaml_text = {
        "path": str(args.out.resolve()),
        "train": "train/images",
        "val": "valid/images",
        "test": "test/images",
        "nc": len(TARGET_NAMES),
        "names": TARGET_NAMES,
    }
    (args.out / "data.yaml").write_text(yaml.safe_dump(yaml_text, sort_keys=False), encoding="utf-8")
    (args.out / "class_support.json").write_text(
        json.dumps(
            {
                "base_dataset": str(args.base),
                "extra_dataset": str(args.extra),
                "extra_source": extra_yaml.get("roboflow", {}),
                "max_base_train_images_per_class": args.max_base_train_images_per_class,
                "class_support": support,
                "total_boxes": {TARGET_NAMES[cid]: int(total_counts[cid]) for cid in range(len(TARGET_NAMES))},
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"[OK] Built {args.out}")
    print(f"[OK] Extra class map: {extra_map}")


if __name__ == "__main__":
    main()
