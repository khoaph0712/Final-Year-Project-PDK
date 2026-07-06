#!/usr/bin/env python
"""Leakage-audit the Roboflow PlastOPol export and finalize it for training (F11 next lever).

PlastOPol v4 (Roboflow, nc=1 'Trash', 2,418 imgs) is a new real-world field source. Before
using it we MUST check it does not overlap the clean field TEST (taco_field_clean_v1) or the
studio/field training data (hardcase train) - F1 lesson: community datasets recirculate the
same photos. Perceptual dHash (Hamming<=6) against every reference image; report overlaps per
split. Also writes a corrected absolute-path data.yaml (the Roboflow one uses broken '../'
relatives) so Ultralytics can train on it directly.

Run: .venv311\\Scripts\\python.exe scripts/audit_plastopol_roboflow.py
"""
from __future__ import annotations
import json
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
PP = ROOT / "external_datasets" / "PlastOPol.v4-plastopol-ver-4.yolo26"
REFS = {
    "taco_field_clean_v1": ROOT / "external_datasets" / "taco_field_clean_v1",
    "hardcase_train": ROOT / "external_datasets" / "yolo26_hardcase_dataset_v1" / "train",
}
IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
HAMMING = 6


def dhash(path: Path, size: int = 8) -> int | None:
    try:
        im = Image.open(path).convert("L").resize((size + 1, size), Image.LANCZOS)
    except Exception:
        return None
    a = np.asarray(im, dtype=np.int16)
    v = 0
    for b in (a[:, 1:] > a[:, :-1]).flatten():
        v = (v << 1) | int(b)
    return v


def imgs_in(d: Path):
    for sub in ("train", "val", "valid", "test", ""):
        p = d / sub / "images" if sub else d / "images"
        if p.exists():
            for f in p.iterdir():
                if f.suffix.lower() in IMG_EXT:
                    yield f


def main() -> None:
    ref_hashes = {}
    for name, d in REFS.items():
        hs = [h for h in (dhash(f) for f in imgs_in(d)) if h is not None]
        ref_hashes[name] = hs
        print(f"ref {name}: {len(hs)} images hashed")
    all_ref = [(n, h) for n, hs in ref_hashes.items() for h in hs]

    leaks = {"train": [], "valid": [], "test": []}
    counts = Counter()
    for split in ("train", "valid", "test"):
        for f in (PP / split / "images").iterdir():
            if f.suffix.lower() not in IMG_EXT:
                continue
            counts[split] += 1
            h = dhash(f)
            if h is None:
                continue
            hit = next((n for n, rh in all_ref if bin(h ^ rh).count("1") <= HAMMING), None)
            if hit:
                leaks[split].append({"image": f.name, "matches_ref": hit})

    report = {
        "plastopol_counts": dict(counts),
        "hamming_threshold": HAMMING,
        "reference_images": {k: len(v) for k, v in ref_hashes.items()},
        "leaks_found": {k: len(v) for k, v in leaks.items()},
        "leak_detail": leaks,
    }
    out = ROOT / "runs" / "audits" / "plastopol_leakage_audit.json"
    json.dump(report, open(out, "w"), indent=2)

    # corrected absolute-path data.yaml (nc=1)
    (PP / "data_fixed.yaml").write_text(
        f"path: {PP}\ntrain: train/images\nval: valid/images\ntest: test/images\n"
        "nc: 1\nnames: ['litter']\n", encoding="utf-8")

    print(json.dumps(report, indent=2))
    print("wrote", out)
    print("wrote", PP / "data_fixed.yaml")


if __name__ == "__main__":
    main()
