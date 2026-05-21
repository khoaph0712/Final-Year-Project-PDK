#!/usr/bin/env python
"""
Final Year Project (FYP) - Waste Sorting & Classification System
CNN Super-Dataset Builder: Merges Kaggle Garbage Classification with merged_dataset_v4

This script:
1. Scans the newly downloaded Kaggle Garbage Classification dataset (12 classes).
2. Maps these 12 fine-grained classes onto our 7-class CNN schema:
   - 'plastic' -> 'plastic'
   - 'brown-glass', 'green-glass', 'white-glass' -> 'glass'
   - 'metal' -> 'metal'
   - 'paper' -> 'paper'
   - 'cardboard' -> 'cardboard'
   - 'biological' -> 'organic'
   - 'trash', 'battery', 'clothes', 'shoes' -> 'Background' (General non-recyclables/noise)
3. Merges these new high-quality crops with our existing 'merged_dataset_v4' crops.
4. Enforces a larger balanced quota (e.g. up to 4,000 train crops and 1,000 test crops per class).
5. Creates 'merged_dataset_v5' structured folders and writes data.yaml for train_cnn.py / train_ann.py.
"""

import os
import shutil
import random
from pathlib import Path
import yaml
from collections import Counter

# =====================================================================
# Configuration Paths
# =====================================================================
ROOT_DIR = Path(__file__).resolve().parent.parent
EXT_DIR = ROOT_DIR / "external_datasets"
KAGGER_DIR = EXT_DIR / "garbage_yolo" / "garbage_classification"
V4_DIR = ROOT_DIR / "merged_dataset_v4"
V5_DIR = ROOT_DIR / "merged_dataset_v5"

TARGET_CLASSES = ["plastic", "glass", "metal", "paper", "cardboard", "organic", "Background"]

# Class mapping from Kaggle 12 classes to our 7 target classes
KAGGL_CLASS_MAP = {
    "plastic": "plastic",
    "brown-glass": "glass",
    "green-glass": "glass",
    "white-glass": "glass",
    "metal": "metal",
    "paper": "paper",
    "cardboard": "cardboard",
    "biological": "organic",
    "trash": "Background",
    "battery": "Background",
    "clothes": "Background",
    "shoes": "Background"
}

def clean_and_create_dir(path):
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)

def collect_crops_from_directory(dir_path):
    """Recursively collects all image paths from a directory, group by class name."""
    crops_by_class = {cls: [] for cls in TARGET_CLASSES}
    
    if not dir_path.exists():
        return crops_by_class
        
    for cls in TARGET_CLASSES:
        cls_dir = dir_path / cls
        if cls_dir.exists():
            crops_by_class[cls].extend(list(cls_dir.glob("*.jpg")) + list(cls_dir.glob("*.png")) + list(cls_dir.glob("*.jpeg")))
            
    return crops_by_class

def main():
    print("=====================================================================")
    print("           CNN Super-Dataset Builder (V5 - Merged Kaggle)            ")
    print("=====================================================================")
    
    if not KAGGER_DIR.exists():
        print(f"[ERROR] Kaggle dataset directory not found at: {KAGGER_DIR}")
        print("Please ensure you extracted the zip into 'external_datasets/garbage_yolo/'")
        return
        
    if not V4_DIR.exists():
        print(f"[ERROR] Base merged_dataset_v4 not found at: {V4_DIR}")
        print("Please build base dataset first.")
        return
        
    # 1. Collect Base Crops from merged_dataset_v4
    print("[INFO] Collecting base crops from merged_dataset_v4...")
    v4_train = collect_crops_from_directory(V4_DIR / "train")
    v4_test = collect_crops_from_directory(V4_DIR / "test")
    
    # 2. Collect and Map Kaggle Crops
    print("[INFO] Collecting and mapping Kaggle Garbage Classification crops...")
    kaggle_mapped = {cls: [] for cls in TARGET_CLASSES}
    
    for sub_dir in KAGGER_DIR.iterdir():
        if sub_dir.is_dir() and not sub_dir.name.startswith("."):
            folder_name = sub_dir.name
            target_class = KAGGL_CLASS_MAP.get(folder_name)
            
            if target_class is not None:
                images = list(sub_dir.glob("*.jpg")) + list(sub_dir.glob("*.png")) + list(sub_dir.glob("*.jpeg")) + list(sub_dir.glob("*.JPG"))
                kaggle_mapped[target_class].extend(images)
                print(f"  - Mapped '{folder_name}' ({len(images)} images) -> target class '{target_class.upper()}'")
                
    # 3. Create Super-Dataset (merged_dataset_v5)
    print("\n[INFO] Initializing Super-Dataset folders under merged_dataset_v5...")
    for split in ["train", "test"]:
        for cls in TARGET_CLASSES:
            (V5_DIR / split / cls).mkdir(parents=True, exist_ok=True)
            
    # Configuration quotas for V5 (Increase dataset size to boost CNN accuracy!)
    # We will aim for up to 3,500 train crops and 800 test crops per class
    TRAIN_QUOTA = 3500
    TEST_QUOTA = 800
    
    random.seed(42)
    
    print(f"\n[INFO] Compiling split quotas (Target: {TRAIN_QUOTA} train / {TEST_QUOTA} test per class)...")
    for cls in TARGET_CLASSES:
        print(f"\n--- Class: {cls.upper()} ---")
        
        # Shuffle mapped Kaggle images
        kaggle_list = kaggle_mapped[cls]
        random.shuffle(kaggle_list)
        
        # Partition Kaggle images into 80/20 train/test
        split_idx = int(len(kaggle_list) * 0.8)
        kaggle_train = kaggle_list[:split_idx]
        kaggle_test = kaggle_list[split_idx:]
        
        # Combine base and Kaggle pools
        train_pool = v4_train[cls] + kaggle_train
        test_pool = v4_test[cls] + kaggle_test
        
        print(f"  - Total Available Train Pool: {len(train_pool)} (V4: {len(v4_train[cls])}, Kaggle: {len(kaggle_train)})")
        print(f"  - Total Available Test Pool: {len(test_pool)} (V4: {len(v4_test[cls])}, Kaggle: {len(kaggle_test)})")
        
        # Limit to quota
        final_train = random.sample(train_pool, min(len(train_pool), TRAIN_QUOTA))
        final_test = random.sample(test_pool, min(len(test_pool), TEST_QUOTA))
        
        print(f"  - Sampled for V5: {len(final_train)} train crops / {len(final_test)} test crops")
        
        # Copy Train Images
        for idx, img_path in enumerate(final_train):
            dest_name = f"c5_{idx}_{img_path.name}"
            shutil.copy2(img_path, V5_DIR / "train" / cls / dest_name)
            
        # Copy Test Images
        for idx, img_path in enumerate(final_test):
            dest_name = f"c5_{idx}_{img_path.name}"
            shutil.copy2(img_path, V5_DIR / "test" / cls / dest_name)
            
    # Generate data.yaml for V5
    data_yaml = {
        "path": str(V5_DIR.resolve()),
        "train": "train",
        "test": "test",
        "nc": len(TARGET_CLASSES),
        "names": TARGET_CLASSES
    }
    
    yaml_path = V5_DIR / "data.yaml"
    with open(yaml_path, 'w', encoding='utf-8') as yf:
        yaml.safe_dump(data_yaml, yf, sort_keys=False)
        
    print(f"\n[SUCCESS] CNN Super-Dataset (v5) successfully built at: {V5_DIR.name}")
    print(f"[OK] data.yaml generated successfully at: {yaml_path}")
    print("=====================================================================")

if __name__ == "__main__":
    main()
