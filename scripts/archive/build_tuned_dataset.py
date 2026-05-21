"""Build a non-destructive tuned YOLO dataset copy.

The builder:
- keeps the original dataset untouched;
- drops unreadable images, missing/empty labels, invalid labels, tiny boxes, and exact duplicate images;
- rebuilds train/valid/test splits by dominant class;
- hardlinks images where possible and writes cleaned label files.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import random
import shutil
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

import cv2
import yaml


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATA = ROOT / "merged_dataset_v3" / "data.yaml"
DEFAULT_OUT = ROOT / "tuned_dataset_v1"
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


@dataclass
class Candidate:
    image_path: Path
    label_lines: list[str]
    dominant_class: int
    class_counts: Counter[int]
    image_hash: str


def resolve_split(root: Path, rel: str) -> Path:
    p = Path(rel)
    return p if p.is_absolute() else root / p


def image_to_label_path(image_path: Path) -> Path:
    return Path(str(image_path).replace("\\images\\", "\\labels\\")).with_suffix(".txt")


def md5_file(path: Path) -> str:
    h = hashlib.md5()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def clean_label_lines(label_path: Path, w_px: int, h_px: int, nc: int, min_box_px: int, min_area: float) -> tuple[list[str], Counter[int], Counter[str]]:
    cleaned: list[str] = []
    counts: Counter[int] = Counter()
    dropped: Counter[str] = Counter()
    for line in label_path.read_text(encoding="utf-8").splitlines():
        parts = line.strip().split()
        if len(parts) != 5:
            dropped["wrong_column_count"] += 1
            continue
        try:
            cid = int(float(parts[0]))
            cx, cy, bw, bh = [float(x) for x in parts[1:]]
        except ValueError:
            dropped["parse_error"] += 1
            continue
        if cid < 0 or cid >= nc:
            dropped["class_out_of_range"] += 1
            continue
        if not (0 <= cx <= 1 and 0 <= cy <= 1 and 0 < bw <= 1 and 0 < bh <= 1):
            dropped["box_out_of_range"] += 1
            continue
        if bw * w_px < min_box_px or bh * h_px < min_box_px or bw * bh < min_area:
            dropped["tiny_box"] += 1
            continue
        cleaned.append(f"{cid} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
        counts[cid] += 1
    return cleaned, counts, dropped


def safe_link_or_copy(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        dst.unlink()
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def unique_name(src: Path, seen: set[str]) -> str:
    name = src.name
    if name not in seen:
        seen.add(name)
        return name
    stem = src.stem
    suffix = src.suffix
    i = 2
    while True:
        candidate = f"{stem}_{i}{suffix}"
        if candidate not in seen:
            seen.add(candidate)
            return candidate
        i += 1


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--valid-ratio", type=float, default=0.1)
    parser.add_argument("--test-ratio", type=float, default=0.1)
    parser.add_argument("--max-train-images-per-class", type=int, default=5000)
    parser.add_argument("--max-valid-images-per-class", type=int, default=900)
    parser.add_argument("--max-test-images-per-class", type=int, default=900)
    parser.add_argument("--max-train-boxes-per-class", type=int, default=9000)
    parser.add_argument("--max-valid-boxes-per-class", type=int, default=1200)
    parser.add_argument("--max-test-boxes-per-class", type=int, default=1200)
    parser.add_argument("--min-box-px", type=int, default=10)
    parser.add_argument("--min-area", type=float, default=0.0002)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if args.out.exists() and not args.overwrite:
        raise SystemExit(f"{args.out} already exists. Pass --overwrite to rebuild it.")
    if args.out.exists() and args.overwrite:
        shutil.rmtree(args.out)

    cfg = yaml.safe_load(args.data.read_text(encoding="utf-8"))
    ds_root = Path(cfg["path"])
    names = list(cfg["names"])
    nc = len(names)
    split_rels = [cfg["train"], cfg.get("val", cfg.get("valid", "valid/images")), cfg["test"]]

    rng = random.Random(args.seed)
    candidates_by_hash: dict[str, Candidate] = {}
    dropped: Counter[str] = Counter()
    label_drop_reasons: Counter[str] = Counter()

    for rel in split_rels:
        img_dir = resolve_split(ds_root, rel)
        for img_path in sorted(p for p in img_dir.rglob("*") if p.suffix.lower() in IMAGE_EXTS):
            label_path = image_to_label_path(img_path)
            if not label_path.exists():
                dropped["missing_label"] += 1
                continue
            img = cv2.imread(str(img_path))
            if img is None:
                dropped["unreadable_image"] += 1
                continue
            h_px, w_px = img.shape[:2]
            cleaned, class_counts, reasons = clean_label_lines(label_path, w_px, h_px, nc, args.min_box_px, args.min_area)
            label_drop_reasons.update(reasons)
            if not cleaned:
                dropped["empty_after_cleaning"] += 1
                continue
            image_hash = md5_file(img_path)
            if image_hash in candidates_by_hash:
                dropped["duplicate_image"] += 1
                continue
            dominant = class_counts.most_common(1)[0][0]
            candidates_by_hash[image_hash] = Candidate(img_path, cleaned, dominant, class_counts, image_hash)

    by_class: dict[int, list[Candidate]] = defaultdict(list)
    for c in candidates_by_hash.values():
        by_class[c.dominant_class].append(c)
    for bucket in by_class.values():
        rng.shuffle(bucket)

    image_caps = {
        "train": args.max_train_images_per_class,
        "valid": args.max_valid_images_per_class,
        "test": args.max_test_images_per_class,
    }
    box_caps = {
        "train": args.max_train_boxes_per_class,
        "valid": args.max_valid_boxes_per_class,
        "test": args.max_test_boxes_per_class,
    }
    assigned: dict[str, list[Candidate]] = {"train": [], "valid": [], "test": []}
    assigned_box_counts: dict[str, Counter[int]] = {s: Counter() for s in assigned}

    def can_add(split: str, item: Candidate) -> bool:
        cap = box_caps[split]
        if cap <= 0:
            return True
        for cid, count in item.class_counts.items():
            if assigned_box_counts[split][cid] >= cap:
                return False
            # Allow a small overshoot only when this is the first useful example for that class.
            if assigned_box_counts[split][cid] + count > cap and assigned_box_counts[split][cid] > cap * 0.98:
                return False
        return True

    def add_items(split: str, items: list[Candidate], image_cap: int) -> None:
        added_for_dominant = 0
        for item in items:
            if image_cap > 0 and added_for_dominant >= image_cap:
                break
            if not can_add(split, item):
                continue
            assigned[split].append(item)
            assigned_box_counts[split].update(item.class_counts)
            added_for_dominant += 1

    for cid in range(nc):
        bucket = by_class.get(cid, [])
        n = len(bucket)
        n_train = min(image_caps["train"], int(round(n * args.train_ratio)))
        n_valid = min(image_caps["valid"], int(round(n * args.valid_ratio)))
        n_test = min(image_caps["test"], n - n_train - n_valid)
        if n_train + n_valid + n_test > n:
            n_test = max(0, n - n_train - n_valid)
        add_items("train", bucket[:n_train], image_caps["train"])
        add_items("valid", bucket[n_train : n_train + n_valid], image_caps["valid"])
        add_items("test", bucket[n_train + n_valid : n_train + n_valid + n_test], image_caps["test"])

    seen_names: set[str] = set()
    output_counts: dict[str, Counter[int]] = {s: Counter() for s in assigned}
    output_images: Counter[str] = Counter()
    for split, items in assigned.items():
        rng.shuffle(items)
        for item in items:
            out_name = unique_name(item.image_path, seen_names)
            out_img = args.out / split / "images" / out_name
            out_lbl = args.out / split / "labels" / Path(out_name).with_suffix(".txt").name
            safe_link_or_copy(item.image_path, out_img)
            out_lbl.parent.mkdir(parents=True, exist_ok=True)
            out_lbl.write_text("\n".join(item.label_lines) + "\n", encoding="utf-8")
            output_images[split] += 1
            output_counts[split].update(item.class_counts)

    tuned_yaml = {
        "path": str(args.out.resolve()),
        "train": "train/images",
        "val": "valid/images",
        "test": "test/images",
        "nc": nc,
        "names": names,
    }
    (args.out / "data.yaml").write_text(yaml.safe_dump(tuned_yaml, sort_keys=False), encoding="utf-8")

    lines = [
        "# Tuned Dataset v1",
        "",
        f"- Source: `{args.data}`",
        f"- Output: `{args.out.resolve()}`",
        f"- Classes: {', '.join(names)}",
        f"- Cleaning: invalid labels removed, boxes smaller than {args.min_box_px}px or area < {args.min_area} removed, exact duplicate images removed.",
        "- Split strategy: rebuild train/valid/test by dominant class with per-class image and box caps.",
        "",
        "## Dropped Inputs",
        "",
    ]
    for key, value in sorted(dropped.items()):
        lines.append(f"- {key}: {value}")
    for key, value in sorted(label_drop_reasons.items()):
        lines.append(f"- label_{key}: {value}")
    lines.extend(["", "## Output Counts", "", "| Split | Images | Boxes |", "|---|---:|---:|"])
    for split in ("train", "valid", "test"):
        lines.append(f"| {split} | {output_images[split]} | {sum(output_counts[split].values())} |")
    lines.extend(["", "## Boxes By Class", "", "| Class | Train | Valid | Test | Total |", "|---|---:|---:|---:|---:|"])
    for cid, name in enumerate(names):
        row = [output_counts[s].get(cid, 0) for s in ("train", "valid", "test")]
        lines.append(f"| {name} | {row[0]} | {row[1]} | {row[2]} | {sum(row)} |")
    (args.out / "TUNING_REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"[OK] Tuned dataset written to {args.out}")
    print(f"[OK] Use data config: {args.out / 'data.yaml'}")


if __name__ == "__main__":
    main()
