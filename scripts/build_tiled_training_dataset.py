#!/usr/bin/env python
"""Build a tile-augmented YOLO training set to attack tiny-object misses.

Motivation (see FAILURES_AND_FIXES.md F4): the detector misses small objects
(organic median box ~1.4% of image). Sliced INFERENCE was tested and hurt
precision; this instead trains ON tiles so small objects appear larger in the
network input during learning.

Construction:
- Every original training image is kept (full-scene context) via hardlink (no
  extra disk).
- Images that contain at least one tiny GT box are additionally cut into 2x2
  overlapping tiles. Tile sources are ranked by tiny-organic content and capped
  (--max-tiled-images) so total size and epoch time stay comparable to baseline.
- Box remap uses Intersection-over-Smaller (IoS): a GT box is kept in a tile
  only if >= --ios-keep of it is inside, then clipped to tile bounds. Tiles with
  no surviving boxes are dropped.
- val/test in the emitted data.yaml point at the ORIGINAL splits (absolute
  paths) so training model-selection matches the baseline and disk stays small.

Run: .venv311\\Scripts\\python.exe scripts/build_tiled_training_dataset.py
"""

from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "external_datasets" / "yolo26_hardcase_dataset_v1"
OUT = ROOT / "external_datasets" / "yolo26_hardcase_tiled_v1"
CLASSES = ["plastic", "glass", "metal", "paper", "cardboard", "organic"]
ORGANIC = CLASSES.index("organic")
IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def load_labels(path: Path) -> list[tuple[int, float, float, float, float]]:
    boxes = []
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            p = line.split()
            if len(p) == 5:
                boxes.append((int(float(p[0])), *(float(v) for v in p[1:5])))
    return boxes


def tile_origins(w: int, h: int, overlap: float) -> list[tuple[int, int, int, int]]:
    tw, th = int(w * (0.5 + overlap / 2)), int(h * (0.5 + overlap / 2))
    xs = [0, w - tw]
    ys = [0, h - th]
    return [(x, y, tw, th) for y in ys for x in xs]


def remap_box(
    box: tuple[int, float, float, float, float], ox: int, oy: int, tw: int, th: int, w: int, h: int, ios_keep: float
) -> tuple[int, float, float, float, float] | None:
    c, cx, cy, bw, bh = box
    x1, y1 = (cx - bw / 2) * w, (cy - bh / 2) * h
    x2, y2 = (cx + bw / 2) * w, (cy + bh / 2) * h
    ix1, iy1 = max(x1, ox), max(y1, oy)
    ix2, iy2 = min(x2, ox + tw), min(y2, oy + th)
    iw, ih = ix2 - ix1, iy2 - iy1
    if iw <= 1 or ih <= 1:
        return None
    box_area = (x2 - x1) * (y2 - y1)
    if (iw * ih) / (box_area + 1e-9) < ios_keep:
        return None
    ncx = ((ix1 + ix2) / 2 - ox) / tw
    ncy = ((iy1 + iy2) / 2 - oy) / th
    nbw, nbh = iw / tw, ih / th
    return (c, ncx, ncy, nbw, nbh)


def write_label(path: Path, boxes: list[tuple[int, float, float, float, float]]) -> None:
    path.write_text(
        "\n".join(f"{c} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}" for c, cx, cy, bw, bh in boxes) + "\n",
        encoding="utf-8",
    )


def link_or_copy(src: Path, dst: Path) -> None:
    if dst.exists():
        return
    try:
        os.link(src, dst)  # hardlink (same NTFS volume, no extra disk)
    except OSError:
        import shutil

        shutil.copy2(src, dst)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tiny-frac", type=float, default=0.01, help="GT box area < this fraction of image = tiny")
    ap.add_argument("--ios-keep", type=float, default=0.5)
    ap.add_argument("--overlap", type=float, default=0.2)
    ap.add_argument("--max-tiled-images", type=int, default=4500)
    ap.add_argument("--min-tile-px", type=int, default=32)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--preview", type=int, default=0, help="write N tile preview images and exit")
    args = ap.parse_args()

    img_dir = SRC / "train" / "images"
    lbl_dir = SRC / "train" / "labels"
    images = sorted(p for p in img_dir.iterdir() if p.suffix.lower() in IMG_EXT)

    # rank images by tiny-organic content for the tiling budget
    ranked = []
    for p in images:
        boxes = load_labels(lbl_dir / (p.stem + ".txt"))
        tiny = [b for b in boxes if b[3] * b[4] < args.tiny_frac]
        tiny_org = sum(1 for b in tiny if b[0] == ORGANIC)
        if tiny:
            ranked.append((tiny_org, len(tiny), p, boxes))
    ranked.sort(key=lambda r: (r[0], r[1]), reverse=True)
    tiled_sources = ranked[: args.max_tiled_images]
    print(f"[INFO] {len(images)} train images; {len(ranked)} contain tiny boxes; tiling top {len(tiled_sources)}")

    out_img = OUT / "train" / "images"
    out_lbl = OUT / "train" / "labels"
    out_img.mkdir(parents=True, exist_ok=True)
    out_lbl.mkdir(parents=True, exist_ok=True)

    if args.preview:
        prev = OUT / "_preview"
        prev.mkdir(parents=True, exist_ok=True)
        for _, _, p, boxes in tiled_sources[: args.preview]:
            img = cv2.imread(str(p))
            h, w = img.shape[:2]
            for ti, (ox, oy, tw, th) in enumerate(tile_origins(w, h, args.overlap)):
                tile = img[oy : oy + th, ox : ox + tw].copy()
                for b in boxes:
                    rb = remap_box(b, ox, oy, tw, th, w, h, args.ios_keep)
                    if rb:
                        c, cx, cy, bw, bh = rb
                        x1 = int((cx - bw / 2) * tw); y1 = int((cy - bh / 2) * th)
                        x2 = int((cx + bw / 2) * tw); y2 = int((cy + bh / 2) * th)
                        cv2.rectangle(tile, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.putText(tile, CLASSES[c], (x1, max(12, y1 - 3)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
                cv2.imwrite(str(prev / f"{p.stem}_tile{ti}.jpg"), tile)
        print(f"[OK] previews at {prev}")
        return

    # 1) keep every full image (hardlink) + its label
    n_full = 0
    for p in images:
        link_or_copy(p, out_img / p.name)
        lbl = lbl_dir / (p.stem + ".txt")
        if lbl.exists():
            link_or_copy(lbl, out_lbl / lbl.name)
        n_full += 1

    # 2) tiles for the budgeted tiny-heavy images
    n_tiles = 0
    tiles_with_organic = 0
    rng = random.Random(args.seed)
    for _, _, p, boxes in tiled_sources:
        img = cv2.imread(str(p))
        if img is None:
            continue
        h, w = img.shape[:2]
        for ti, (ox, oy, tw, th) in enumerate(tile_origins(w, h, args.overlap)):
            if tw < args.min_tile_px or th < args.min_tile_px:
                continue
            kept = [remap_box(b, ox, oy, tw, th, w, h, args.ios_keep) for b in boxes]
            kept = [b for b in kept if b is not None]
            if not kept:
                continue
            tile = img[oy : oy + th, ox : ox + tw]
            name = f"{p.stem}__tile{ti}.jpg"
            cv2.imwrite(str(out_img / name), tile, [cv2.IMWRITE_JPEG_QUALITY, 92])
            write_label(out_lbl / f"{p.stem}__tile{ti}.txt", kept)
            n_tiles += 1
            if any(b[0] == ORGANIC for b in kept):
                tiles_with_organic += 1

    val_dir = SRC / "val" / "images"
    test_dir = SRC / "test" / "images"
    (OUT / "data.yaml").write_text(
        f"path: {OUT}\n"
        f"train: train/images\n"
        f"val: {val_dir}\n"
        f"test: {test_dir}\n"
        f"nc: {len(CLASSES)}\n"
        f"names: {CLASSES}\n",
        encoding="utf-8",
    )
    summary = {
        "full_images": n_full,
        "tiles": n_tiles,
        "tiles_with_organic": tiles_with_organic,
        "total_train": n_full + n_tiles,
        "tiled_sources": len(tiled_sources),
    }
    (OUT / "build_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"[OK] tiled dataset at {OUT}")


if __name__ == "__main__":
    main()
