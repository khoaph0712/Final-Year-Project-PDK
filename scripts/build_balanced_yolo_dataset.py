"""Build a more balanced YOLO material dataset from an existing YOLO folder.

The current hard-case YOLO merge is heavily skewed by box counts. This script
selects images greedily so each split keeps roughly capped per-class boxes.
Images can still contain multiple classes, so exact balance is not guaranteed,
but dominant classes stop flooding training/evaluation.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import os
import random
import shutil
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SRC = ROOT / "external_datasets" / "yolo26_hardcase_dataset_v1"
DEFAULT_OUT = ROOT / "external_datasets" / "yolo26_balanced_realworld_v1"
CLASSES = ["plastic", "glass", "metal", "paper", "cardboard", "organic"]
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def reset_output(path: Path) -> None:
    resolved = path.resolve()
    external_root = (ROOT / "external_datasets").resolve()
    if external_root not in resolved.parents:
        raise ValueError(f"refusing to clear output outside external_datasets/: {resolved}")
    if resolved.exists():
        shutil.rmtree(resolved)
    for split in ("train", "val", "test"):
        (resolved / split / "images").mkdir(parents=True, exist_ok=True)
        (resolved / split / "labels").mkdir(parents=True, exist_ok=True)


def image_files(path: Path) -> list[Path]:
    return [item for item in path.iterdir() if item.is_file() and item.suffix.lower() in IMAGE_EXTS]


def class_counts(label_file: Path) -> Counter[int]:
    counts: Counter[int] = Counter()
    if not label_file.exists():
        return counts
    for line in label_file.read_text(encoding="utf-8").splitlines():
        parts = line.strip().split()
        if parts:
            counts[int(float(parts[0]))] += 1
    return counts


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


def stable_name(split: str, image: Path) -> str:
    digest = hashlib.md5(str(image).encode("utf-8")).hexdigest()[:10]
    return f"{split}_{digest}{image.suffix.lower()}"


def load_items(src: Path, split: str) -> list[dict]:
    items = []
    for image in image_files(src / split / "images"):
        label = src / split / "labels" / f"{image.stem}.txt"
        counts = class_counts(label)
        if not counts:
            continue
        items.append({"image": image, "label": label, "counts": counts, "boxes": sum(counts.values())})
    return items


def select_balanced(items: list[dict], caps: dict[int, int], max_images: int, seed: int) -> list[dict]:
    rng = random.Random(seed)
    global_counts: Counter[int] = Counter()
    for item in items:
        global_counts.update(item["counts"])
    shuffled = items[:]
    rng.shuffle(shuffled)
    # Prefer rare-class and low-box images first; giant dominant-class images enter only if useful.
    def item_key(item: dict) -> tuple[float, int, float]:
        rare_score = sum(item["counts"][class_id] / max(1, global_counts[class_id]) for class_id in item["counts"])
        return (-rare_score, item["boxes"], rng.random())

    shuffled.sort(key=item_key)
    selected = []
    totals: Counter[int] = Counter()
    for item in shuffled:
        if len(selected) >= max_images:
            break
        useful = False
        overflow = False
        for class_id, count in item["counts"].items():
            if totals[class_id] < caps[class_id]:
                useful = True
            if totals[class_id] + count > int(caps[class_id] * 1.12):
                overflow = True
        if useful and not overflow:
            selected.append(item)
            totals.update(item["counts"])
    return selected


def remove_selected(items: list[dict], selected: list[dict]) -> list[dict]:
    selected_paths = {item["image"] for item in selected}
    return [item for item in items if item["image"] not in selected_paths]


def export_split(out: Path, split: str, items: list[dict]) -> list[dict[str, str]]:
    rows = []
    for item in items:
        name = stable_name(split, item["image"])
        target_image = out / split / "images" / name
        target_label = out / split / "labels" / f"{Path(name).stem}.txt"
        image_method = link_or_copy(item["image"], target_image)
        label_method = link_or_copy(item["label"], target_label)
        rows.append(
            {
                "split": split,
                "source_image": str(item["image"].relative_to(ROOT)),
                "source_label": str(item["label"].relative_to(ROOT)),
                "target_image": str(target_image.relative_to(ROOT)),
                "target_label": str(target_label.relative_to(ROOT)),
                "image_method": image_method,
                "label_method": label_method,
                "counts": ";".join(f"{CLASSES[k]}={v}" for k, v in sorted(item["counts"].items())),
            }
        )
    return rows


def write_yaml(out: Path) -> None:
    names = "\n".join(f"- {name}" for name in CLASSES)
    (out / "data.yaml").write_text(
        f"path: {out}\n"
        "train: train/images\n"
        "val: val/images\n"
        "test: test/images\n"
        f"nc: {len(CLASSES)}\n"
        "names:\n"
        f"{names}\n",
        encoding="utf-8",
    )


def summarize(out: Path) -> dict[str, dict[str, int]]:
    summary: dict[str, dict[str, int]] = {}
    for split in ("train", "val", "test"):
        counts: Counter[int] = Counter()
        for label in (out / split / "labels").glob("*.txt"):
            counts.update(class_counts(label))
        summary[split] = {CLASSES[idx]: counts[idx] for idx in range(len(CLASSES))}
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--src", type=Path, default=DEFAULT_SRC)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--train-cap", type=int, default=6500)
    parser.add_argument("--val-cap", type=int, default=900)
    parser.add_argument("--test-cap", type=int, default=900)
    parser.add_argument("--max-train-images", type=int, default=9000)
    parser.add_argument("--max-val-images", type=int, default=1200)
    parser.add_argument("--max-test-images", type=int, default=1200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--global-resplit", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out = args.out_dir.resolve()
    if out.exists() and not args.overwrite:
        raise SystemExit(f"output exists; pass --overwrite to rebuild: {out}")
    reset_output(out)

    rows = []
    if args.global_resplit:
        pool = []
        for old_split in ("train", "val", "test"):
            pool.extend(load_items(args.src.resolve(), old_split))
        random.Random(args.seed).shuffle(pool)
        test = select_balanced(pool, {idx: args.test_cap for idx in range(len(CLASSES))}, args.max_test_images, args.seed + 101)
        pool = remove_selected(pool, test)
        val = select_balanced(pool, {idx: args.val_cap for idx in range(len(CLASSES))}, args.max_val_images, args.seed + 202)
        pool = remove_selected(pool, val)
        train = select_balanced(pool, {idx: args.train_cap for idx in range(len(CLASSES))}, args.max_train_images, args.seed + 303)
        for split, selected in (("train", train), ("val", val), ("test", test)):
            rows.extend(export_split(out, split, selected))
            print(f"{split}: selected {len(selected)} images")
    else:
        split_cfg = {
            "train": (args.train_cap, args.max_train_images),
            "val": (args.val_cap, args.max_val_images),
            "test": (args.test_cap, args.max_test_images),
        }
        for split, (cap, max_images) in split_cfg.items():
            items = load_items(args.src.resolve(), split)
            selected = select_balanced(items, {idx: cap for idx in range(len(CLASSES))}, max_images, args.seed + len(split))
            rows.extend(export_split(out, split, selected))
            print(f"{split}: selected {len(selected)} / {len(items)} images")

    write_yaml(out)
    with (out / "source_manifest.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=sorted({key for row in rows for key in row}))
        writer.writeheader()
        writer.writerows(rows)
    summary = summarize(out)
    (out / "BALANCE_REPORT.json").write_text(__import__("json").dumps(summary, indent=2), encoding="utf-8")
    print(__import__("json").dumps(summary, indent=2))


if __name__ == "__main__":
    main()
