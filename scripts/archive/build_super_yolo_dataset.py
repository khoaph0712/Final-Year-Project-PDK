#!/usr/bin/env python
"""
Waste Sorting & Classification System - Super YOLOv11 Dataset Builder
Combines three datasets:
1. taco_yolo (baseline TACO ~1,500 images)
2. rf_taco_trash (roboflow TACO ~12,800 images)
3. rf_garbage_cls (roboflow Garbage ~2,800 images)

Maps classes to our unified target schema:
['plastic', 'glass', 'metal', 'paper', 'cardboard', 'organic']
"""

import os
import shutil
from pathlib import Path
import yaml
from collections import Counter

# Configure paths
ROOT_DIR = Path(r"C:\FYP_v2")
EXT_DIR = ROOT_DIR / "external_datasets"
TACO_YOLO_DIR = EXT_DIR / "taco_yolo"
RF_TACO_DIR = ROOT_DIR / "rf_taco_trash"
RF_GARBAGE_DIR = ROOT_DIR / "rf_garbage_cls"
SUPER_DIR = EXT_DIR / "super_yolo_dataset"

TARGET_CLASSES = ["plastic", "glass", "metal", "paper", "cardboard", "organic"]
CLASS_TO_IDX = {name: idx for idx, name in enumerate(TARGET_CLASSES)}

def merge_split(split_name, src_datasets_configs):
    """
    Merges images and labels for a specific split (train, val, or test).
    src_datasets_configs is a list of dicts:
    {
        'name': 'taco_yolo',
        'dir': TACO_YOLO_DIR,
        'split_folder': 'train', # e.g. train, val, test or valid
        'class_map': { ... }
    }
    """
    dest_img_dir = SUPER_DIR / split_name / "images"
    dest_lbl_dir = SUPER_DIR / split_name / "labels"
    dest_img_dir.mkdir(parents=True, exist_ok=True)
    dest_lbl_dir.mkdir(parents=True, exist_ok=True)
    
    total_copied = 0
    class_counter = Counter()
    
    for config in src_datasets_configs:
        src_dir = config['dir']
        src_split_name = config['split_folder']
        class_map = config['class_map']
        prefix = config['name']
        
        src_split_img = src_dir / src_split_name / "images"
        src_split_lbl = src_dir / src_split_name / "labels"
        
        if not src_split_img.exists() or not src_split_lbl.exists():
            print(f"[WARNING] Directory not found for {prefix} {src_split_name} split: {src_split_img} or {src_split_lbl}")
            continue
            
        print(f"[INFO] Ingesting split '{src_split_name}' from '{prefix}'...")
        
        # Iterate over labels in source directory
        for label_file in src_split_lbl.glob("*.txt"):
            # Check matching image files
            img_found = None
            img_ext = None
            for ext in [".jpg", ".png", ".jpeg", ".JPG", ".PNG", ".JPEG"]:
                test_img = src_split_img / label_file.with_suffix(ext).name
                if test_img.exists():
                    img_found = test_img
                    img_ext = ext
                    break
                    
            if not img_found:
                continue # Skip if label has no matching image
                
            # Process label lines
            valid_lines = []
            with open(label_file, 'r', encoding='utf-8') as lf:
                for line in lf:
                    parts = line.strip().split()
                    if not parts:
                        continue
                    try:
                        old_cid = int(parts[0])
                    except ValueError:
                        continue
                        
                    # Map class ID
                    mapped_class_name = class_map.get(old_cid)
                    if mapped_class_name is None or mapped_class_name not in CLASS_TO_IDX:
                        continue # Skip classes not in target schema
                        
                    new_cid = CLASS_TO_IDX[mapped_class_name]
                    class_counter[mapped_class_name] += 1
                    
                    # Append updated line with mapped class index
                    valid_lines.append(f"{new_cid} " + " ".join(parts[1:]))
                    
            if not valid_lines:
                continue # Skip if no relevant classes found
                
            # Unique filename to avoid collisions
            unique_name = f"{prefix}_{label_file.stem}"
            dest_img_path = dest_img_dir / f"{unique_name}{img_ext}"
            dest_lbl_path = dest_lbl_dir / f"{unique_name}.txt"
            
            # Copy image and write labels
            shutil.copy2(img_found, dest_img_path)
            with open(dest_lbl_path, 'w', encoding='utf-8') as df:
                df.write("\n".join(valid_lines) + "\n")
                
            total_copied += 1
            
    print(f"[OK] Completed {split_name} split: Merged {total_copied} images.")
    print("Class distribution in split:")
    for cls, count in class_counter.items():
        print(f"  - {cls.upper()}: {count}")
    print("-" * 60)

def main():
    print("=====================================================================")
    print("                YOLOv11 SUPER DATASET BUILDER                        ")
    print("=====================================================================")
    
    # 1. Class Mappings to target schema
    taco_yolo_map = {0: "plastic", 1: "glass", 2: "metal", 3: "paper", 4: "cardboard", 5: "organic"}
    
    # rf_taco_trash mapping:
    # 0: aluminum -> metal
    # 1: battery -> metal
    # 2: can -> metal
    # 3: cap or lid -> plastic
    # 4: cardboard boxes and cartons -> cardboard
    # 5: food waste -> organic
    # 6: glass -> glass
    # 7: paper -> paper
    # 8: plastic bag -> plastic
    # 9: plastic container -> plastic
    # 10: styrofoam -> plastic
    # 11: utensils and straw -> plastic
    rf_taco_map = {
        0: "metal",
        1: "metal",
        2: "metal",
        3: "plastic",
        4: "cardboard",
        5: "organic",
        6: "glass",
        7: "paper",
        8: "plastic",
        9: "plastic",
        10: "plastic",
        11: "plastic"
    }
    
    # rf_garbage_cls mapping:
    # names: ['BIODEGRADABLE', 'CARDBOARD', 'GLASS', 'METAL', 'PAPER', 'PLASTIC']
    # 0: BIODEGRADABLE -> organic
    # 1: CARDBOARD -> cardboard
    # 2: GLASS -> glass
    # 3: METAL -> metal
    # 4: PAPER -> paper
    # 5: PLASTIC -> plastic
    rf_garbage_map = {
        0: "organic",
        1: "cardboard",
        2: "glass",
        3: "metal",
        4: "paper",
        5: "plastic"
    }
    
    # Clear previous merge directory to start fresh
    if SUPER_DIR.exists():
        print(f"[INFO] Cleaning up old directory: {SUPER_DIR}...")
        shutil.rmtree(SUPER_DIR)
    SUPER_DIR.mkdir(parents=True, exist_ok=True)
    
    # 2. Configure Source splits mapping (val vs valid)
    splits = {
        "train": [
            {"name": "taco_yolo", "dir": TACO_YOLO_DIR, "split_folder": "train", "class_map": taco_yolo_map},
            {"name": "rf_taco", "dir": RF_TACO_DIR, "split_folder": "train", "class_map": rf_taco_map},
            {"name": "rf_garbage", "dir": RF_GARBAGE_DIR, "split_folder": "train", "class_map": rf_garbage_map}
        ],
        "val": [
            {"name": "taco_yolo", "dir": TACO_YOLO_DIR, "split_folder": "val", "class_map": taco_yolo_map},
            {"name": "rf_taco", "dir": RF_TACO_DIR, "split_folder": "valid", "class_map": rf_taco_map},
            {"name": "rf_garbage", "dir": RF_GARBAGE_DIR, "split_folder": "valid", "class_map": rf_garbage_map}
        ],
        "test": [
            {"name": "taco_yolo", "dir": TACO_YOLO_DIR, "split_folder": "test", "class_map": taco_yolo_map},
            {"name": "rf_taco", "dir": RF_TACO_DIR, "split_folder": "test", "class_map": rf_taco_map},
            {"name": "rf_garbage", "dir": RF_GARBAGE_DIR, "split_folder": "test", "class_map": rf_garbage_map}
        ]
    }
    
    # Merge train, val, and test splits
    for target_split, sources in splits.items():
        merge_split(target_split, sources)
        
    # Generate data.yaml config for YOLOv11 training
    data_yaml = {
        "path": str(SUPER_DIR.resolve()),
        "train": "train/images",
        "val": "val/images",
        "test": "test/images",
        "nc": len(TARGET_CLASSES),
        "names": TARGET_CLASSES
    }
    
    yaml_path = SUPER_DIR / "data.yaml"
    with open(yaml_path, 'w', encoding='utf-8') as yf:
        yaml.safe_dump(data_yaml, yf, sort_keys=False)
        
    print(f"\n[SUCCESS] Super YOLOv11 dataset created at: {SUPER_DIR}")
    print(f"[OK] data.yaml successfully generated at: {yaml_path}")
    print("=====================================================================")

if __name__ == "__main__":
    main()
