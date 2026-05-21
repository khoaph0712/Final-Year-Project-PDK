"""Convert COCO detection/segmentation annotations to YOLO labels.

This is used for official TACO-style datasets after images are downloaded.
It preserves the original class names unless a later controlled-merge mapping
is applied.
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
from collections import Counter
from pathlib import Path

import cv2
import yaml


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def find_image(root: Path, file_name: str) -> Path | None:
    direct = root / file_name
    if direct.exists():
        return direct
    matches = list(root.rglob(Path(file_name).name))
    return matches[0] if matches else None


def yolo_line(cid: int, bbox: list[float], w: int, h: int) -> str | None:
    x, y, bw, bh = bbox
    if bw <= 0 or bh <= 0 or w <= 0 or h <= 0:
        return None
    cx = (x + bw / 2.0) / w
    cy = (y + bh / 2.0) / h
    nw = bw / w
    nh = bh / h
    vals = [max(0.0, min(1.0, v)) for v in (cx, cy, nw, nh)]
    if vals[2] <= 0 or vals[3] <= 0:
        return None
    return f"{cid} {vals[0]:.6f} {vals[1]:.6f} {vals[2]:.6f} {vals[3]:.6f}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--images-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--valid-ratio", type=float, default=0.1)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if args.out.exists() and args.overwrite:
        shutil.rmtree(args.out)
    args.out.mkdir(parents=True, exist_ok=True)

    coco = json.loads(args.annotations.read_text(encoding="utf-8"))
    categories = sorted(coco["categories"], key=lambda c: int(c["id"]))
    cat_to_new = {int(cat["id"]): i for i, cat in enumerate(categories)}
    names = [cat["name"] for cat in categories]
    images = {int(img["id"]): img for img in coco["images"]}
    anns_by_img: dict[int, list[dict]] = {iid: [] for iid in images}
    for ann in coco.get("annotations", []):
        if int(ann.get("category_id", -1)) in cat_to_new:
            anns_by_img.setdefault(int(ann["image_id"]), []).append(ann)

    ids = [iid for iid, anns in anns_by_img.items() if anns]
    rng = random.Random(args.seed)
    rng.shuffle(ids)
    n = len(ids)
    n_train = int(round(n * args.train_ratio))
    n_valid = int(round(n * args.valid_ratio))
    split_ids = {
        "train": ids[:n_train],
        "valid": ids[n_train : n_train + n_valid],
        "test": ids[n_train + n_valid :],
    }

    class_counts = {split: Counter() for split in split_ids}
    split_counts = Counter()
    missing = 0
    for split, image_ids in split_ids.items():
        for iid in image_ids:
            img_info = images[iid]
            src_img = find_image(args.images_root, img_info["file_name"])
            if src_img is None:
                missing += 1
                continue
            img = cv2.imread(str(src_img))
            if img is None:
                missing += 1
                continue
            h, w = img.shape[:2]
            lines: list[str] = []
            for ann in anns_by_img.get(iid, []):
                cid = cat_to_new[int(ann["category_id"])]
                line = yolo_line(cid, ann["bbox"], w, h)
                if line:
                    lines.append(line)
                    class_counts[split][names[cid]] += 1
            if not lines:
                continue
            safe_name = img_info["file_name"].replace("/", "__").replace("\\", "__")
            out_img = args.out / split / "images" / safe_name
            out_lbl = args.out / split / "labels" / Path(safe_name).with_suffix(".txt").name
            out_img.parent.mkdir(parents=True, exist_ok=True)
            out_lbl.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src_img, out_img)
            out_lbl.write_text("\n".join(lines) + "\n", encoding="utf-8")
            split_counts[split] += 1

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
        "# Prepared COCO Dataset",
        "",
        f"- Source annotations: `{args.annotations.resolve()}`",
        f"- Images root: `{args.images_root.resolve()}`",
        f"- Output: `{args.out.resolve()}`",
        f"- Classes: {len(names)}",
        f"- Missing/unreadable images skipped: {missing}",
        "",
        "## Split Counts",
        "| Split | Images | Objects |",
        "|---|---:|---:|",
    ]
    for split in ("train", "valid", "test"):
        report.append(f"| {split} | {split_counts[split]} | {sum(class_counts[split].values())} |")
    (args.out / "PREPARE_REPORT.md").write_text("\n".join(report), encoding="utf-8")
    print(f"[OK] Prepared {args.out}; missing images: {missing}")


if __name__ == "__main__":
    main()
