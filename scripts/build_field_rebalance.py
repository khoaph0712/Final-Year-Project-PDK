#!/usr/bin/env python
"""Assemble the field-rebalanced fine-tune train set (Step 1 execution).

Deployed detector train is 95% studio (19,559 studio / 1,034 TACO field), which
drowns the field signal -> field small-box recall 0.318. This raises the field
share to ~25% by oversampling the CLEAN TACO field train split 6x (hardlinks with
distinct stems, so Ultralytics cannot dedup them), combined with the studio images
via a train image-list txt. Field labels come from taco_field_clean_v1 (F10 clean),
so no leaked/near-dup TACO labels enter.

val = taco_field_clean_v1/val (field) so early-stopping optimizes the target metric.

Run: .venv311\\Scripts\\python.exe scripts/build_field_rebalance.py --k 6
"""
from __future__ import annotations
import argparse, os, shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HC = ROOT / "external_datasets" / "yolo26_hardcase_dataset_v1"
FIELD = ROOT / "external_datasets" / "taco_field_clean_v1"
OUT = ROOT / "external_datasets" / "field_rebalance_v1"
CLASSES = ["plastic", "glass", "metal", "paper", "cardboard", "organic"]
IMG_EXT = {".jpg", ".jpeg", ".png"}


def link(src: Path, dst: Path):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=6, help="field oversample factor")
    a = ap.parse_args()

    ox_img = OUT / "field_x" / "images"
    ox_lbl = OUT / "field_x" / "labels"
    if (OUT / "field_x").exists():
        shutil.rmtree(OUT / "field_x")
    ox_img.mkdir(parents=True, exist_ok=True)
    ox_lbl.mkdir(parents=True, exist_ok=True)

    field_imgs = sorted(p for p in (FIELD / "train" / "images").iterdir() if p.suffix.lower() in IMG_EXT)
    n_field = 0
    for p in field_imgs:
        lbl = FIELD / "train" / "labels" / (p.stem + ".txt")
        if not lbl.exists():
            continue
        for k in range(a.k):
            link(p, ox_img / f"{p.stem}_r{k}{p.suffix}")
            link(lbl, ox_lbl / f"{p.stem}_r{k}.txt")
            n_field += 1

    # studio: every hardcase train image that is NOT a TACO photo (studio only) x1
    studio = [p for p in (HC / "train" / "images").iterdir()
              if p.suffix.lower() in IMG_EXT and not p.name.startswith("taco")]

    lines = [str(p.resolve()) for p in studio]
    lines += [str(p.resolve()) for p in ox_img.iterdir() if p.suffix.lower() in IMG_EXT]
    train_txt = OUT / "train_list.txt"
    train_txt.write_text("\n".join(lines), encoding="utf-8")

    (OUT / "data.yaml").write_text(
        f"train: {train_txt}\n"
        f"val: {FIELD / 'val' / 'images'}\n"
        f"test: {FIELD / 'test' / 'images'}\n"
        f"nc: {len(CLASSES)}\nnames: {CLASSES}\n",
        encoding="utf-8",
    )
    print(f"studio={len(studio)} field_oversampled={n_field} (k={a.k}) "
          f"field_share={n_field/(len(studio)+n_field):.3f} total={len(lines)}")
    print("data.yaml ->", OUT / "data.yaml")


if __name__ == "__main__":
    main()
