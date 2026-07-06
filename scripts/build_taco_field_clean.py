#!/usr/bin/env python
"""Build a clean, source-isolated TACO field YOLO split (F10 fix).

The in-repo TACO data is leak-entangled (F1/F10): the hardcase merge put the same
TACO photo into train AND val/test via different re-encodings, and realworld_v2 is
74% studio leaked. This rebuilds a TACO-only detector dataset straight from the
official COCO annotations, split BY PHOTO so no image can appear in two splits.

- Boxes: TACO 60 categories -> 6 material classes via material_map_draft.json.
  Categories that map to "review"/other are dropped (kept as unlabeled background
  content is fine for a class-agnostic-style detector; they are litter but not one
  of our 6 materials, so we exclude those boxes rather than mislabel them).
- Split: deterministic hash of the TACO image id -> 80/10/10 train/val/test.
- Output YOLO dataset at external_datasets/taco_field_clean_v1/ with data.yaml.

This is FIELD training + a clean FIELD test set. Run:
  .venv311\\Scripts\\python.exe scripts/build_taco_field_clean.py
"""
from __future__ import annotations

import hashlib
import json
import shutil
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "external_datasets" / "hard_case_full" / "taco_official"
OUT = ROOT / "external_datasets" / "taco_field_clean_v1"
CLASSES = ["plastic", "glass", "metal", "paper", "cardboard", "organic"]
CLASS_IDX = {c: i for i, c in enumerate(CLASSES)}


def hardcase_train_taco_md5() -> set[str]:
    """MD5 of every TACO image the deployed detector trained on (hardcase train).

    Test/val must be drawn ONLY from photos NOT in this set, so the baseline
    deployed model is evaluated on genuinely unseen field images (F10)."""
    d = ROOT / "external_datasets" / "yolo26_hardcase_dataset_v1" / "train" / "images"
    seen = set()
    if d.exists():
        for p in d.iterdir():
            if p.name.startswith("taco") and p.suffix.lower() in {".jpg", ".jpeg", ".png"}:
                seen.add(hashlib.md5(p.read_bytes()).hexdigest())
    return seen


def main() -> None:
    coco = json.load(open(SRC / "data" / "annotations.json", encoding="utf-8"))
    mm = json.load(open(SRC / "material_map_draft.json", encoding="utf-8"))
    cat_to_cls = {}
    for cid, rec in mm.items():
        tgt = str(rec.get("target_class", "")).lower()
        if tgt in CLASS_IDX:
            cat_to_cls[int(cid)] = CLASS_IDX[tgt]

    seen_md5 = hardcase_train_taco_md5()
    imgs = {im["id"]: im for im in coco["images"]}
    anns_by_img = defaultdict(list)
    for a in coco["annotations"]:
        anns_by_img[a["image_id"]].append(a)

    def mapped_box_count(iid: int) -> int:
        return sum(1 for a in anns_by_img.get(iid, []) if a["category_id"] in cat_to_cls)

    # Novel photos (unseen by deployed model) WITH >=1 mapped box are reserved for
    # val+test; first 120 by id -> test, next 90 -> val, remainder -> train. Seen
    # photos -> train only, so baseline+finetuned models both get an unseen field test.
    novel_ids = []
    for iid, im in imgs.items():
        src_img = SRC / "images" / im["file_name"]
        if not src_img.exists() or mapped_box_count(iid) == 0:
            continue
        if hashlib.md5(src_img.read_bytes()).hexdigest() not in seen_md5:
            novel_ids.append(iid)
    novel_ids.sort()
    test_ids = set(novel_ids[:120])
    val_ids = set(novel_ids[120:210])

    def split_of(image_id: int) -> str:
        if image_id in test_ids:
            return "test"
        if image_id in val_ids:
            return "val"
        return "train"

    if OUT.exists():
        shutil.rmtree(OUT)
    for sp in ("train", "val", "test"):
        (OUT / sp / "images").mkdir(parents=True, exist_ok=True)
        (OUT / sp / "labels").mkdir(parents=True, exist_ok=True)

    split_imgs = Counter()
    split_boxes = Counter()
    per_split_cls = {sp: Counter() for sp in ("train", "val", "test")}
    skipped_no_box = 0

    for iid, im in imgs.items():
        src_img = SRC / "images" / im["file_name"]
        if not src_img.exists():
            continue
        W, H = im["width"], im["height"]
        lines = []
        for a in anns_by_img.get(iid, []):
            ci = cat_to_cls.get(a["category_id"])
            if ci is None:
                continue
            x, y, bw, bh = a["bbox"]
            cx, cy = (x + bw / 2) / W, (y + bh / 2) / H
            nw, nh = bw / W, bh / H
            if nw <= 0 or nh <= 0:
                continue
            lines.append(f"{ci} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")
            per_split_cls_key = ci
        if not lines:
            skipped_no_box += 1
            continue
        sp = split_of(iid)
        stem = f"taco_official_{iid:05d}"
        shutil.copy2(src_img, OUT / sp / "images" / f"{stem}.jpg")
        (OUT / sp / "labels" / f"{stem}.txt").write_text("\n".join(lines), encoding="utf-8")
        split_imgs[sp] += 1
        split_boxes[sp] += len(lines)
        for ln in lines:
            per_split_cls[sp][CLASSES[int(ln.split()[0])]] += 1

    (OUT / "data.yaml").write_text(
        "path: " + str(OUT) + "\n"
        "train: train/images\nval: val/images\ntest: test/images\n"
        f"nc: {len(CLASSES)}\nnames: {CLASSES}\n",
        encoding="utf-8",
    )
    report = {
        "images": dict(split_imgs),
        "boxes": dict(split_boxes),
        "per_split_class": {k: dict(v) for k, v in per_split_cls.items()},
        "skipped_images_no_mapped_box": skipped_no_box,
        "mapped_categories": len(cat_to_cls),
    }
    json.dump(report, open(OUT / "BUILD_REPORT.json", "w"), indent=2)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
