#!/usr/bin/env python
"""Finalize the Roboflow PlastOPol export into a clean nc=1 YOLO dataset.

- Drops the 21 leaked images flagged by scripts/audit_plastopol_roboflow.py
  (near-dups of taco_field_clean_v1 / hardcase train) so it can be merged without
  contaminating the clean field eval.
- Converts Roboflow segmentation-polygon labels ('0 x1 y1 x2 y2 ...') to axis-aligned
  YOLO bounding boxes ('0 cx cy w h') for detection training.
- Writes external_datasets/plastopol_clean_v1/ (train/val/test) + data.yaml (nc=1).

Run: .venv311\\Scripts\\python.exe scripts/finalize_plastopol_clean.py
"""
from __future__ import annotations
import json, shutil
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "external_datasets" / "PlastOPol.v4-plastopol-ver-4.yolo26"
OUT = ROOT / "external_datasets" / "plastopol_clean_v1"
AUDIT = ROOT / "runs" / "audits" / "plastopol_leakage_audit.json"
IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
SRC_SPLIT = {"train": "train", "valid": "val", "test": "test"}


def poly_to_bbox(tokens: list[str]) -> str | None:
    """tokens = [cls, x1, y1, x2, y2, ...] normalized. Returns '0 cx cy w h'."""
    coords = [float(t) for t in tokens[1:]]
    if len(coords) == 4:  # already cx,cy,w,h
        cx, cy, w, h = coords
        return f"0 {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}" if w > 0 and h > 0 else None
    if len(coords) < 6 or len(coords) % 2:
        return None
    xs, ys = coords[0::2], coords[1::2]
    x0, x1 = min(xs), max(xs)
    y0, y1 = min(ys), max(ys)
    w, h = x1 - x0, y1 - y0
    if w <= 0 or h <= 0:
        return None
    return f"0 {(x0 + x1) / 2:.6f} {(y0 + y1) / 2:.6f} {w:.6f} {h:.6f}"


def main() -> None:
    audit = json.load(open(AUDIT))
    leaked = {d["image"] for split in audit["leak_detail"].values() for d in split}
    print(f"dropping {len(leaked)} leaked images")

    if OUT.exists():
        shutil.rmtree(OUT)
    for sp in ("train", "val", "test"):
        (OUT / sp / "images").mkdir(parents=True, exist_ok=True)
        (OUT / sp / "labels").mkdir(parents=True, exist_ok=True)

    kept = Counter(); boxes = Counter(); dropped_leak = 0; dropped_nobox = 0
    for src_split, dst_split in SRC_SPLIT.items():
        for img in (SRC / src_split / "images").iterdir():
            if img.suffix.lower() not in IMG_EXT:
                continue
            if img.name in leaked:
                dropped_leak += 1
                continue
            lbl = SRC / src_split / "labels" / (img.stem + ".txt")
            lines = []
            if lbl.exists():
                for ln in lbl.read_text().splitlines():
                    t = ln.split()
                    if len(t) >= 5:
                        bb = poly_to_bbox(t)
                        if bb:
                            lines.append(bb)
            if not lines:
                dropped_nobox += 1
                continue
            shutil.copy2(img, OUT / dst_split / "images" / img.name)
            (OUT / dst_split / "labels" / (img.stem + ".txt")).write_text("\n".join(lines), encoding="utf-8")
            kept[dst_split] += 1; boxes[dst_split] += len(lines)

    (OUT / "data.yaml").write_text(
        f"path: {OUT}\ntrain: train/images\nval: val/images\ntest: test/images\n"
        "nc: 1\nnames: ['litter']\n", encoding="utf-8")
    report = {
        "kept_images": dict(kept), "boxes": dict(boxes),
        "dropped_leaked": dropped_leak, "dropped_no_valid_box": dropped_nobox,
    }
    json.dump(report, open(OUT / "BUILD_REPORT.json", "w"), indent=2)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
