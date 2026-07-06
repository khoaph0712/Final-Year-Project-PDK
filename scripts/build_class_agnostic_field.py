#!/usr/bin/env python
"""Assemble the class-agnostic (nc=1) field-expansion training set.

Realigns the detector with the intended architecture: YOLO finds 'litter', the
classifier does material ID on crops. Collapses every source label to class 0 and
unions:
  - studio: hardcase train, non-TACO images   (x1)
  - field:  taco_field_clean_v1/train + plastopol_clean_v1/train  (xK, NEW data)
val = taco_field_clean_v1/val + plastopol_clean_v1/val (field early-stop).
Images hardlinked (distinct stems so Ultralytics can't dedup the oversampled field).

Run: .venv311\\Scripts\\python.exe scripts/build_class_agnostic_field.py --k 2
"""
from __future__ import annotations
import argparse, os, shutil
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HC = ROOT / "external_datasets" / "yolo26_hardcase_dataset_v1" / "train"
TACO = ROOT / "external_datasets" / "taco_field_clean_v1"
PP = ROOT / "external_datasets" / "plastopol_clean_v1"
OUT = ROOT / "external_datasets" / "class_agnostic_field_v1"
IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def link(src: Path, dst: Path):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def collapse_label(src_lbl: Path) -> str:
    out = []
    if src_lbl.exists():
        for ln in src_lbl.read_text().splitlines():
            t = ln.split()
            if len(t) >= 5:
                out.append("0 " + " ".join(t[1:5]))
    return "\n".join(out)


def emit(img: Path, lbl_dir: Path, dst_img_dir: Path, dst_lbl_dir: Path, stem: str) -> int:
    body = collapse_label(lbl_dir / (img.stem + ".txt"))
    if not body:
        return 0
    link(img, dst_img_dir / f"{stem}{img.suffix}")
    (dst_lbl_dir / f"{stem}.txt").write_text(body, encoding="utf-8")
    return 1


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=2, help="field oversample factor")
    a = ap.parse_args()

    if OUT.exists():
        shutil.rmtree(OUT)
    for sp in ("train", "val"):
        (OUT / sp / "images").mkdir(parents=True, exist_ok=True)
        (OUT / sp / "labels").mkdir(parents=True, exist_ok=True)
    ti, tl = OUT / "train" / "images", OUT / "train" / "labels"
    vi, vl = OUT / "val" / "images", OUT / "val" / "labels"
    c = Counter()

    # studio x1 (non-TACO hardcase train)
    for img in (HC / "images").iterdir():
        if img.suffix.lower() in IMG_EXT and not img.name.startswith("taco"):
            c["studio"] += emit(img, HC / "labels", ti, tl, f"studio_{img.stem}")

    # field xK (NEW + clean): taco_field_clean train + plastopol_clean train
    for tag, ds in (("taco", TACO), ("pp", PP)):
        for img in (ds / "train" / "images").iterdir():
            if img.suffix.lower() not in IMG_EXT:
                continue
            for k in range(a.k):
                c[f"field_{tag}"] += emit(img, ds / "train" / "labels", ti, tl, f"{tag}_{img.stem}_r{k}")

    # val: field val (x1)
    for tag, ds in (("taco", TACO), ("pp", PP)):
        for img in (ds / "val" / "images").iterdir():
            if img.suffix.lower() in IMG_EXT:
                c[f"val_{tag}"] += emit(img, ds / "val" / "labels", vi, vl, f"{tag}_{img.stem}")

    (OUT / "data.yaml").write_text(
        f"path: {OUT}\ntrain: train/images\nval: val/images\n"
        "nc: 1\nnames: ['litter']\n", encoding="utf-8")
    studio = c["studio"]; field = c["field_taco"] + c["field_pp"]
    print(dict(c))
    print(f"train studio={studio} field={field} field_share={field/(studio+field):.3f} "
          f"val={c['val_taco']+c['val_pp']}")


if __name__ == "__main__":
    main()
