#!/usr/bin/env python
"""Build a clean YOLO split from PlastOPol + a leakage audit (F11 next lever).

PlastOPol is a ONE-CLASS ("litter") real-world litter-detection dataset (2,418 imgs,
~5,300 boxes, COCO bbox x1,y1,w,h, CC-BY-4.0). This:
  1. auto-detects the extracted layout (COCO json + image dir),
  2. converts to YOLO format as a single class 0 = "litter",
  3. splits BY IMAGE (deterministic hash) 80/10/10,
  4. runs a perceptual dHash near-dup audit of every PlastOPol image against the
     existing field/studio sets (taco_field_clean_v1, hardcase train) and DROPS any
     PlastOPol image that duplicates one, so the split can be merged without re-poisoning
     the clean eval (F1 procedure).

Output: external_datasets/plastopol_clean_v1/ with data.yaml (nc=1) + AUDIT.json.

Run (after the zip is extracted):
  .venv311\\Scripts\\python.exe scripts/build_plastopol_split.py \
      --src external_datasets/downloads/plastopol/extracted
"""
from __future__ import annotations
import argparse, hashlib, json, shutil
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "external_datasets" / "plastopol_clean_v1"
IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
DEDUP_REFS = [
    ROOT / "external_datasets" / "taco_field_clean_v1",
    ROOT / "external_datasets" / "yolo26_hardcase_dataset_v1" / "train",
]


def dhash(path: Path, size: int = 8) -> int | None:
    try:
        im = Image.open(path).convert("L").resize((size + 1, size), Image.LANCZOS)
    except Exception:
        return None
    a = np.asarray(im, dtype=np.int16)
    bits = a[:, 1:] > a[:, :-1]
    v = 0
    for b in bits.flatten():
        v = (v << 1) | int(b)
    return v


def hamming(a: int, b: int) -> int:
    return bin(a ^ b).count("1")


def split_of(key: str) -> str:
    h = int(hashlib.md5(key.encode()).hexdigest(), 16) % 100
    return "train" if h < 80 else ("val" if h < 90 else "test")


def find_coco(src: Path) -> Path | None:
    cands = []
    for j in src.rglob("*.json"):
        try:
            d = json.load(open(j, encoding="utf-8"))
        except Exception:
            continue
        if isinstance(d, dict) and {"images", "annotations"} <= set(d):
            cands.append((len(d["annotations"]), j))
    return max(cands)[1] if cands else None


def build_image_index(src: Path) -> dict[str, Path]:
    idx = {}
    for p in src.rglob("*"):
        if p.suffix.lower() in IMG_EXT:
            idx.setdefault(p.name, p)
    return idx


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", type=Path, required=True, help="extracted PlastOPol dir")
    ap.add_argument("--dedup-hamming", type=int, default=6)
    a = ap.parse_args()

    coco_path = find_coco(a.src)
    if not coco_path:
        raise SystemExit(f"No COCO annotations.json found under {a.src}")
    print("COCO:", coco_path)
    coco = json.load(open(coco_path, encoding="utf-8"))
    imgs = {im["id"]: im for im in coco["images"]}
    anns_by_img = defaultdict(list)
    for an in coco["annotations"]:
        anns_by_img[an["image_id"]].append(an)
    img_index = build_image_index(a.src)

    # reference dHashes for the leakage audit
    ref_hashes = []
    for ref in DEDUP_REFS:
        for sub in ("train", "val", "test", ""):
            d = ref / sub / "images" if sub else ref / "images"
            if d.exists():
                for p in d.iterdir():
                    if p.suffix.lower() in IMG_EXT:
                        h = dhash(p)
                        if h is not None:
                            ref_hashes.append(h)
    print(f"reference images for dedup: {len(ref_hashes)}")

    if OUT.exists():
        shutil.rmtree(OUT)
    for sp in ("train", "val", "test"):
        (OUT / sp / "images").mkdir(parents=True, exist_ok=True)
        (OUT / sp / "labels").mkdir(parents=True, exist_ok=True)

    from collections import Counter
    kept = Counter(); boxes = Counter(); dropped_leak = 0; missing = 0
    for iid, im in imgs.items():
        fname = Path(im["file_name"]).name
        src_img = img_index.get(fname)
        if src_img is None:
            missing += 1
            continue
        W = im.get("width") or 0
        H = im.get("height") or 0
        if not W or not H:
            with Image.open(src_img) as pim:
                W, H = pim.size
        lines = []
        for an in anns_by_img.get(iid, []):
            x, y, bw, bh = an["bbox"]
            if bw <= 0 or bh <= 0:
                continue
            cx, cy = (x + bw / 2) / W, (y + bh / 2) / H
            lines.append(f"0 {cx:.6f} {cy:.6f} {bw / W:.6f} {bh / H:.6f}")
        if not lines:
            continue
        h = dhash(src_img)
        if h is not None and any(hamming(h, rh) <= a.dedup_hamming for rh in ref_hashes):
            dropped_leak += 1
            continue
        sp = split_of(fname)
        stem = f"plastopol_{iid:05d}"
        shutil.copy2(src_img, OUT / sp / "images" / f"{stem}.jpg")
        (OUT / sp / "labels" / f"{stem}.txt").write_text("\n".join(lines), encoding="utf-8")
        kept[sp] += 1; boxes[sp] += len(lines)

    (OUT / "data.yaml").write_text(
        f"path: {OUT}\ntrain: train/images\nval: val/images\ntest: test/images\n"
        "nc: 1\nnames: ['litter']\n", encoding="utf-8")
    report = {
        "coco": str(coco_path), "images_total": len(imgs),
        "kept": dict(kept), "boxes": dict(boxes),
        "dropped_as_leak": dropped_leak, "missing_image_file": missing,
        "dedup_hamming_threshold": a.dedup_hamming,
        "reference_images": len(ref_hashes),
    }
    json.dump(report, open(OUT / "AUDIT.json", "w"), indent=2)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
