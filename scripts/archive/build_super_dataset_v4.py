#!/usr/bin/env python
"""
Final Year Project (FYP) - Waste Sorting & Classification System
Phase 7: Super Dataset Builder (v4) - Memory Optimized Version

This script aggregates crops from:
1. c:\\FYP_v2\\external_crops\\ (extracted from TACO, WaDaBa, TrashNet)
2. c:\\FYP_v2\\merged_dataset_v3\\ (backfill via on-the-fly cropping as needed)

And builds a strictly balanced crop dataset at c:\\FYP_v2\\merged_dataset_v4\\ containing:
- Exactly 2,000 training crop images per class.
- Exactly 500 testing crop images per class.
For a total of 17,500 crops across 7 target classes.

Memory Optimization: Avoids loading entire raw datasets into memory. Scans and crops
only the exact number of images needed for backfilling.
"""

import os
import sys
import yaml
import cv2
import random
import shutil
import numpy as np
from pathlib import Path

# Set random seeds for reproducibility
random.seed(42)
np.random.seed(42)

ROOT_DIR = Path(__file__).resolve().parent.parent
EXTERNAL_CROPS_DIR = ROOT_DIR / "external_crops"
V3_DATASET_DIR = ROOT_DIR / "merged_dataset_v3"
V4_DATASET_DIR = ROOT_DIR / "merged_dataset_v4"

TARGET_CLASSES = ["plastic", "glass", "metal", "paper", "cardboard", "organic", "Background"]
TRAIN_COUNT_PER_CLASS = 2000
TEST_COUNT_PER_CLASS = 500
TOTAL_NEEDED_PER_CLASS = TRAIN_COUNT_PER_CLASS + TEST_COUNT_PER_CLASS

def augment_image(img, seed_val):
    """Apply robust geometric and color augmentations to crop images."""
    rng = random.Random(seed_val)
    # 1. Random horizontal flip
    if rng.choice([True, False]):
        img = cv2.flip(img, 1)
    # 2. Random vertical flip
    if rng.choice([True, False]):
        img = cv2.flip(img, 0)
    # 3. Random minor rotation (-15 to 15 degrees)
    angle = rng.uniform(-15, 15)
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), angle, 1.0)
    img = cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REFLECT_101)
    # 4. Random brightness scale (85% to 115%)
    factor = rng.uniform(0.85, 1.15)
    img = np.clip(img * factor, 0, 255).astype(np.uint8)
    return img

def harvest_v3_crops_needed(v3_dir, target_class, class_names, num_needed):
    """Scan v3 and harvest exactly num_needed crops for target_class, then stop to save memory."""
    if num_needed <= 0:
        return []
        
    print(f"  [INFO] Harvesting exactly {num_needed} crops for '{target_class}' from v3...")
    harvested = []
    
    # Process train, valid, test splits in v3
    for split in ["train", "valid", "test"]:
        if len(harvested) >= num_needed:
            break
            
        img_dir = v3_dir / split / "images"
        lbl_dir = v3_dir / split / "labels"
        if not img_dir.exists() or not lbl_dir.exists():
            continue
            
        img_paths = list(img_dir.glob("*"))
        random.shuffle(img_paths)  # Shuffle so we get diverse crops
        
        for img_path in img_paths:
            if len(harvested) >= num_needed:
                break
            if img_path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}:
                continue
            lbl_path = lbl_dir / img_path.with_suffix(".txt").name
            if not lbl_path.exists():
                continue
                
            img = None  # Lazy loading: only read image if we find the class inside
            try:
                for line in lbl_path.read_text(encoding="utf-8").splitlines():
                    if len(harvested) >= num_needed:
                        break
                    parts = line.strip().split()
                    if len(parts) != 5:
                        continue
                    cid = int(float(parts[0]))
                    if 0 <= cid < len(class_names):
                        cname = class_names[cid]
                        if cname == target_class:
                            if img is None:
                                img = cv2.imread(str(img_path))
                                if img is None:
                                    break
                                h, w = img.shape[:2]
                                
                            cx, cy, bw, bh = [float(x) for x in parts[1:]]
                            x1 = int((cx - bw / 2.0) * w)
                            y1 = int((cy - bh / 2.0) * h)
                            x2 = int((cx + bw / 2.0) * w)
                            y2 = int((cy + bh / 2.0) * h)
                            
                            x1 = max(0, min(x1, w - 1))
                            y1 = max(0, min(y1, h - 1))
                            x2 = max(x1 + 1, min(x2, w))
                            y2 = max(y1 + 1, min(y2, h))
                            
                            if (x2 - x1) >= 10 and (y2 - y1) >= 10:
                                harvested.append(img[y1:y2, x1:x2])
            except Exception:
                pass
                
    print(f"  [OK] Successfully harvested {len(harvested)} crops for '{target_class}'.")
    return harvested

def harvest_v3_backgrounds_needed(v3_dir, num_needed, seed=42):
    """Scan v3 for images with empty label files and crop random background patches."""
    if num_needed <= 0:
        return []
        
    print(f"  [INFO] Harvesting exactly {num_needed} background crops from v3 empty labels...")
    bg_paths = []
    
    for split in ["train", "valid", "test"]:
        img_dir = v3_dir / split / "images"
        lbl_dir = v3_dir / split / "labels"
        if not img_dir.exists():
            continue
        for img_path in img_dir.glob("*"):
            if img_path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}:
                lbl_path = lbl_dir / img_path.with_suffix(".txt").name
                if not lbl_path.exists() or lbl_path.stat().st_size == 0:
                    bg_paths.append(img_path)
                    
    random.Random(seed).shuffle(bg_paths)
    
    harvested = []
    rng = random.Random(seed)
    for p in bg_paths:
        if len(harvested) >= num_needed:
            break
        img = cv2.imread(str(p))
        if img is None:
            continue
        h, w = img.shape[:2]
        
        # Extract up to 5 crops per image
        num_crops = min(5, num_needed - len(harvested))
        for _ in range(num_crops):
            c_size = rng.randint(64, min(h, w, 256))
            x = rng.randint(0, w - c_size)
            y = rng.randint(0, h - c_size)
            crop = img[y : y + c_size, x : x + c_size]
            harvested.append(crop)
            
    print(f"  [OK] Successfully harvested {len(harvested)} background crops.")
    return harvested

def main():
    print("====================================================")
    print("      FYP SUPER DATASET v4 BUILDER (MEMORY OPT)     ")
    print("====================================================")
    
    # Recreate the output dir cleanly
    if V4_DATASET_DIR.exists():
        print(f"[INFO] Cleaning prior merged_dataset_v4 directory: {V4_DATASET_DIR}")
        shutil.rmtree(V4_DATASET_DIR)
    V4_DATASET_DIR.mkdir(parents=True, exist_ok=True)
    
    # Read names from data.yaml in v3_dir to make sure mappings are correct
    v3_yaml = V3_DATASET_DIR / "data.yaml"
    if not v3_yaml.exists():
        print(f"[ERROR] merged_dataset_v3/data.yaml not found. Aborting.")
        return
    cfg = yaml.safe_load(v3_yaml.read_text(encoding="utf-8"))
    class_names = list(cfg["names"])
    
    # 1. Scan external crops pool
    external_crops = {c: [] for c in TARGET_CLASSES}
    print("[INFO] Scanning external crops pool...")
    for cname in TARGET_CLASSES:
        c_dir = EXTERNAL_CROPS_DIR / cname
        if c_dir.exists():
            for img_path in c_dir.glob("*"):
                if img_path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}:
                    external_crops[cname].append(img_path)
        print(f"  * {cname.upper():<12}: {len(external_crops[cname])} crops found on disk.")
        
    # 2. Process and balance each class
    print("\n--- Constructing Balanced Crops Pool (Memory Efficient) ---")
    rng = random.Random(42)
    
    for cname in TARGET_CLASSES:
        # Step A: Load the required amount of external crops
        pool = []
        ext_paths = list(external_crops[cname])
        rng.shuffle(ext_paths)
        
        # Load up to 2500 external crops
        num_ext_to_load = min(len(ext_paths), TOTAL_NEEDED_PER_CLASS)
        for p in ext_paths[:num_ext_to_load]:
            img = cv2.imread(str(p))
            if img is not None:
                pool.append(img)
                
        ext_count = len(pool)
        
        # Step B: Backfill from v3 if pool is insufficient
        backfill_count = 0
        if len(pool) < TOTAL_NEEDED_PER_CLASS:
            needed = TOTAL_NEEDED_PER_CLASS - len(pool)
            if cname == "Background":
                v3_bg = harvest_v3_backgrounds_needed(V3_DATASET_DIR, needed)
                pool.extend(v3_bg)
            else:
                v3_c = harvest_v3_crops_needed(V3_DATASET_DIR, cname, class_names, needed)
                pool.extend(v3_c)
            backfill_count = len(pool) - ext_count
            
        # Step C: Augment if still short
        augmented_count = 0
        if len(pool) < TOTAL_NEEDED_PER_CLASS:
            shortfall = TOTAL_NEEDED_PER_CLASS - len(pool)
            base_items = list(pool)
            
            if not base_items:
                print(f"  [WARN] No source images for '{cname}'! Creating synthetic placeholders.")
                for _ in range(TOTAL_NEEDED_PER_CLASS):
                    synthetic = np.random.randint(0, 256, (128, 128, 3), dtype=np.uint8)
                    pool.append(synthetic)
            else:
                for idx in range(shortfall):
                    base_img = rng.choice(base_items)
                    pool.append(augment_image(base_img, seed_val=42 + idx))
                augmented_count = shortfall
                
        # Shuffle final class pool and save to disk
        rng.shuffle(pool)
        class_pool = pool[:TOTAL_NEEDED_PER_CLASS]
        
        print(f"  => {cname.upper():<12}: Ext={ext_count:>4} | Backfill={backfill_count:>4} | Aug={augmented_count:>4} | Saving {len(class_pool)} crops...")
        
        # Save crops immediately to train/test folders to release memory
        for split, count, slice_start, slice_end in [
            ("train", TRAIN_COUNT_PER_CLASS, 0, TRAIN_COUNT_PER_CLASS),
            ("test", TEST_COUNT_PER_CLASS, TRAIN_COUNT_PER_CLASS, TOTAL_NEEDED_PER_CLASS)
        ]:
            out_class_dir = V4_DATASET_DIR / split / cname
            out_class_dir.mkdir(parents=True, exist_ok=True)
            
            crops_slice = class_pool[slice_start:slice_end]
            for idx, img in enumerate(crops_slice):
                img_name = f"{cname}_{split}_{idx:04d}.jpg"
                cv2.imwrite(str(out_class_dir / img_name), img)
                
        # Free up memory
        del pool
        del class_pool
        
    # 3. Write data.yaml configuration file
    data_yaml_content = {
        "path": str(V4_DATASET_DIR.resolve()).replace("\\", "/"),
        "train": "train",
        "val": "test",
        "test": "test",
        "nc": len(TARGET_CLASSES),
        "names": TARGET_CLASSES,
        "type": "classification"
    }
    
    yaml_path = V4_DATASET_DIR / "data.yaml"
    with open(yaml_path, "w") as f:
        yaml.safe_dump(data_yaml_content, f, sort_keys=False)
        
    print(f"\n[OK] Super Dataset v4 built successfully at: {V4_DATASET_DIR}")
    print(f"[OK] data.yaml saved at: {yaml_path}")
    print("="*60)

if __name__ == "__main__":
    main()
