"""Build merged_dataset_v3 with improved class balance.

Changes vs v2:
- Drop noisy `BIODEGRADABLE` (rf_garbage_cls) and `food waste` (rf_taco_trash)
- Pull clean organic from rf_food_waste (all 32 food classes -> organic)
- Narrow `other`: keep styrofoam (taco) + add all Cigarette (rf_cigarettes);
  drop ambiguous battery / cap-lid / utensils / generic Trash / generic Waste
- Re-split glass (and any other class with 0 test) so every class has test images
"""
from __future__ import annotations

import random
import shutil
from collections import Counter, defaultdict
from pathlib import Path

random.seed(42)

ROOT = Path(r"C:\FYP_v2")
OUT = ROOT / "merged_dataset_v3"

# Target (canonical) class layout
TARGET_NAMES = ["plastic", "glass", "metal", "paper", "cardboard", "organic", "other"]
T = {n: i for i, n in enumerate(TARGET_NAMES)}

# Per-source-dataset mapping: source_cls_id -> target_cls_name or None (drop)
SOURCES: dict[str, dict[int, str | None]] = {
    # rf_taco_trash: 12 classes
    "rf_taco_trash": {
        0: "metal",        # aluminum
        1: None,           # battery  (narrowing other)
        2: "metal",        # can
        3: None,           # cap or lid (ambiguous)
        4: "cardboard",    # cardboard boxes and cartons
        5: None,           # food waste (replaced by rf_food_waste)
        6: "glass",        # glass
        7: "paper",        # paper
        8: "plastic",      # plastic bag
        9: "plastic",      # plastic container
        10: "other",       # styrofoam
        11: None,          # utensils and straw (ambiguous)
    },
    # rf_garbage_cls: 6 classes  (drop BIODEGRADABLE = noisy auto-tiled labels)
    "rf_garbage_cls": {
        0: None,           # BIODEGRADABLE
        1: "cardboard",
        2: "glass",
        3: "metal",
        4: "paper",
        5: "plastic",
    },
    # rf_uca_recyclable: 3 classes
    "rf_uca_recyclable": {
        0: "cardboard",
        1: "paper",
        2: "plastic",
    },
    # rf_waste_sorting: 6 classes  (drop generic Trash per narrowing)
    "rf_waste_sorting": {
        0: "cardboard",
        1: "glass",
        2: "metal",
        3: "paper",
        4: "plastic",
        5: None,           # Trash (too generic)
    },
    # rf_trash_detection: 1 class 'Waste' - too generic, skip entirely
    "rf_trash_detection": {0: None},
    # NEW: rf_food_waste: 32 food classes -> organic (all)
    "rf_food_waste": "organic",   # special: map every source class to organic
    # NEW: rf_cigarettes: 1 class Cigarette -> other
    "rf_cigarettes": {0: "other"},
}

SPLITS = ("train", "valid", "test")
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def remap_label_file(src_file: Path, mapping) -> list[str]:
    """Return list of remapped YOLO label lines. Empty list means drop the image."""
    out_lines: list[str] = []
    for raw in src_file.read_text().splitlines():
        parts = raw.strip().split()
        if not parts:
            continue
        try:
            cid = int(parts[0])
        except ValueError:
            continue
        # map
        if isinstance(mapping, str):
            target_name = mapping  # single-target source
        else:
            target_name = mapping.get(cid)
        if target_name is None:
            continue
        new_id = T[target_name]
        out_lines.append(" ".join([str(new_id), *parts[1:]]))
    return out_lines


def clear_out() -> None:
    if OUT.exists():
        shutil.rmtree(OUT)
    for split in SPLITS:
        (OUT / split / "images").mkdir(parents=True, exist_ok=True)
        (OUT / split / "labels").mkdir(parents=True, exist_ok=True)


def ingest_source(src_folder: str, mapping) -> Counter:
    src_root = ROOT / src_folder
    if not src_root.exists():
        print(f"  [SKIP] {src_folder}: folder not found")
        return Counter()
    stats = Counter()
    for split in SPLITS:
        img_dir = src_root / split / "images"
        lbl_dir = src_root / split / "labels"
        if not img_dir.exists():
            continue
        for lbl in lbl_dir.glob("*.txt"):
            out_lines = remap_label_file(lbl, mapping)
            if not out_lines:
                stats["dropped_images"] += 1
                continue
            # find the image
            img = None
            for ext in IMG_EXTS:
                cand = img_dir / (lbl.stem + ext)
                if cand.exists():
                    img = cand
                    break
            if img is None:
                stats["missing_images"] += 1
                continue
            # prefix to avoid filename clashes
            new_stem = f"{src_folder}__{split}__{lbl.stem}"
            (OUT / split / "images" / (new_stem + img.suffix)).write_bytes(img.read_bytes())
            (OUT / split / "labels" / (new_stem + ".txt")).write_text("\n".join(out_lines) + "\n")
            stats["kept_images"] += 1
            for line in out_lines:
                stats[f"box_{TARGET_NAMES[int(line.split()[0])]}"] += 1
    return stats


def load_class_set(lbl_file: Path) -> set[int]:
    classes = set()
    for line in lbl_file.read_text().splitlines():
        parts = line.strip().split()
        if parts:
            try:
                classes.add(int(parts[0]))
            except ValueError:
                pass
    return classes


def rebalance_test(min_test_per_class: int = 200) -> None:
    """For any class with fewer than min_test_per_class test boxes, move
    train images containing that class into test until the quota is met."""
    for cid, cname in enumerate(TARGET_NAMES):
        # count test boxes for this class
        test_boxes = 0
        for f in (OUT / "test" / "labels").glob("*.txt"):
            for line in f.read_text().splitlines():
                parts = line.strip().split()
                if parts and parts[0] == str(cid):
                    test_boxes += 1
        if test_boxes >= min_test_per_class:
            continue
        # find train images whose label files contain this class
        candidates = []
        for f in (OUT / "train" / "labels").glob("*.txt"):
            if cid in load_class_set(f):
                candidates.append(f)
        if not candidates:
            print(f"  [skip rebalance] no train images for {cname}")
            continue
        random.shuffle(candidates)
        # estimate: each train image contributes ~boxes_for_cid toward quota
        needed = min_test_per_class - test_boxes
        moved = 0
        moved_boxes = 0
        for f in candidates:
            # count boxes of this class in this image
            k = sum(
                1
                for line in f.read_text().splitlines()
                if line.strip().split()[:1] == [str(cid)]
            )
            # move image + label from train to test
            stem = f.stem
            img = None
            for ext in IMG_EXTS:
                cand = OUT / "train" / "images" / (stem + ext)
                if cand.exists():
                    img = cand
                    break
            if img is None:
                continue
            shutil.move(str(img), str(OUT / "test" / "images" / img.name))
            shutil.move(str(f), str(OUT / "test" / "labels" / f.name))
            moved += 1
            moved_boxes += k
            if moved_boxes >= needed:
                break
        print(f"  [rebalance] {cname}: moved {moved} imgs (~{moved_boxes} {cname} boxes) train->test")


def write_data_yaml() -> None:
    yaml = (
        f"path: {OUT}\n"
        "train: train/images\n"
        "val: valid/images\n"
        "test: test/images\n"
        f"nc: {len(TARGET_NAMES)}\n"
        "names:\n"
        + "\n".join(f"- {n}" for n in TARGET_NAMES)
        + "\n"
    )
    (OUT / "data.yaml").write_text(yaml)


def main() -> None:
    clear_out()
    grand: Counter = Counter()
    for folder, mapping in SOURCES.items():
        print(f"\n[+] ingesting {folder}")
        stats = ingest_source(folder, mapping)
        for k, v in stats.items():
            print(f"    {k}: {v}")
        grand.update(stats)

    print("\n[+] rebalancing test split (ensure every class has test samples)")
    rebalance_test(min_test_per_class=200)

    write_data_yaml()

    print("\n=== GRAND TOTALS ===")
    for k in sorted(grand):
        print(f"  {k}: {grand[k]}")
    print("\nWrote", OUT / "data.yaml")


if __name__ == "__main__":
    main()
