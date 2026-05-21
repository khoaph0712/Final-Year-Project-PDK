#!/usr/bin/env python
"""
Final Year Project (FYP) - Waste Sorting & Classification System
YOLO Dataset Merger: Combines TACO YOLO and External Garbage YOLO Datasets

This script:
1. Scans two YOLO-formatted datasets (e.g. taco_yolo and garbage_yolo).
2. Maps their respective class IDs onto our unified target schema:
   ['plastic', 'glass', 'metal', 'paper', 'cardboard', 'organic']
3. Merges images and rewrites normalized label text files with aligned class IDs.
4. Generates a unified 'merged_yolo' dataset structured with train/val/test splits.
5. Produces a consolidated data.yaml config file ready for YOLOv11 training.
"""

import os
import shutil
from pathlib import Path
import yaml
from collections import Counter

# =====================================================================
# Configuration Paths
# =====================================================================
ROOT_DIR = Path(__file__).resolve().parent.parent
EXT_DIR = ROOT_DIR / "external_datasets"
TACO_YOLO_DIR = EXT_DIR / "taco_yolo"
GARBAGE_YOLO_DIR = EXT_DIR / "garbage_yolo"  # User will extract Kaggle zip here
MERGED_DIR = EXT_DIR / "merged_yolo"

TARGET_CLASSES = ["plastic", "glass", "metal", "paper", "cardboard", "organic"]
CLASS_TO_IDX = {name: idx for idx, name in enumerate(TARGET_CLASSES)}

def merge_split(split_name, src_dirs_with_maps):
    """
    Merges images and labels for a specific split (train, val, or test)
    from multiple sources with their respective class mappings.
    """
    dest_img_dir = MERGED_DIR / split_name / "images"
    dest_lbl_dir = MERGED_DIR / split_name / "labels"
    dest_img_dir.mkdir(parents=True, exist_ok=True)
    dest_lbl_dir.mkdir(parents=True, exist_ok=True)
    
    total_copied = 0
    class_counter = Counter()
    
    for src_dir, class_map in src_dirs_with_maps:
        src_split_img = src_dir / split_name / "images"
        src_split_lbl = src_dir / split_name / "labels"
        
        if not src_split_img.exists() or not src_split_lbl.exists():
            continue
            
        print(f"[INFO] Ingesting {split_name} split from: {src_dir.name}...")
        
        # Iterate over labels in source directory
        for label_file in src_split_lbl.glob("*.txt"):
            img_name_jpg = label_file.with_suffix(".jpg").name
            img_name_png = label_file.with_suffix(".png").name
            img_name_jpeg = label_file.with_suffix(".jpeg").name
            
            # Find matching image
            src_img_path = None
            img_name = None
            for name in [img_name_jpg, img_name_png, img_name_jpeg, label_file.with_suffix(".JPG").name]:
                test_path = src_split_img / name
                if test_path.exists():
                    src_img_path = test_path
                    img_name = name
                    break
                    
            if src_img_path is None:
                continue # Skip if label has no matching image
                
            # Process label lines
            valid_lines = []
            with open(label_file, 'r', encoding='utf-8') as lf:
                for line in lf:
                    parts = line.strip().split()
                    if not parts:
                        continue
                    old_cid = int(parts[0])
                    
                    # Map class ID
                    mapped_class_name = class_map.get(old_cid)
                    if mapped_class_name is None or mapped_class_name not in CLASS_TO_IDX:
                        continue # Skip classes not in target schema
                        
                    new_cid = CLASS_TO_IDX[mapped_class_name]
                    class_counter[mapped_class_name] += 1
                    
                    # Append updated line
                    valid_lines.append(f"{new_cid} " + " ".join(parts[1:]))
                    
            if not valid_lines:
                continue # Skip images with no relevant classes
                
            # Copy image and write new label
            unique_prefix = f"{src_dir.name}_"
            dest_img_path = dest_img_dir / f"{unique_prefix}{img_name}"
            dest_lbl_path = dest_lbl_dir / f"{unique_prefix}{label_file.name}"
            
            shutil.copy2(src_img_path, dest_img_path)
            with open(dest_lbl_path, 'w', encoding='utf-8') as df:
                df.write("\n".join(valid_lines) + "\n")
                
            total_copied += 1
            
    print(f"[OK] Completed {split_name} split: Merged {total_copied} images.")
    print("Class distribution in split:")
    for cls, count in class_counter.items():
        print(f"  - {cls.upper()}: {count}")
    print("-" * 50)

def main():
    print("=====================================================================")
    print("                YOLOv11 Multi-Dataset Unified Merger                 ")
    print("=====================================================================")
    
    if not TACO_YOLO_DIR.exists():
        print(f"[ERROR] TACO YOLO dataset not found at: {TACO_YOLO_DIR}")
        print("Please build it first using 'prepare_taco_yolo.py'.")
        return
        
    if not GARBAGE_YOLO_DIR.exists():
        print(f"[WARNING] External Garbage YOLO dataset not found at: {GARBAGE_YOLO_DIR}")
        print("Please download the Kaggle dataset, create the folder 'external_datasets/garbage_yolo',")
        print("and extract the files there to enable multi-dataset merging.")
        return
        
    # Define source dataset class mappings to our unified 6 classes:
    # ['plastic', 'glass', 'metal', 'paper', 'cardboard', 'organic']
    
    # 1. TACO YOLO map (built by prepare_taco_yolo.py)
    taco_class_map = {
        0: "plastic",
        1: "glass",
        2: "metal",
        3: "paper",
        4: "cardboard",
        5: "organic"
    }
    
    # 2. Kaggle Garbage YOLO map (typically has: 0: cardboard, 1: glass, 2: metal, 3: paper, 4: plastic, 5: trash)
    # Adjust this map to match the exact categories in the Kaggle dataset metadata
    garbage_class_map = {
        0: "cardboard",
        1: "glass",
        2: "metal",
        3: "paper",
        4: "plastic",
        5: "organic",     # Map food waste/biodegradable to organic if exists
        6: None           # Skip general/trash category to avoid noisy features
    }
    
    src_datasets = [
        (TACO_YOLO_DIR, taco_class_map),
        (GARBAGE_YOLO_DIR, garbage_class_map)
    ]
    
    # Clear previous merge
    if MERGED_DIR.exists():
        print(f"[INFO] Cleaning up old merged directory: {MERGED_DIR.name}...")
        shutil.rmtree(MERGED_DIR)
        
    MERGED_DIR.mkdir(parents=True, exist_ok=True)
    
    # Merge splits
    for split in ["train", "val", "test"]:
        merge_split(split, src_datasets)
        
    # Generate unified data.yaml
    data_yaml = {
        "path": str(MERGED_DIR.resolve()),
        "train": "train/images",
        "val": "val/images",
        "test": "test/images",
        "nc": len(TARGET_CLASSES),
        "names": TARGET_CLASSES
    }
    
    yaml_path = MERGED_DIR / "data.yaml"
    with open(yaml_path, 'w', encoding='utf-8') as yf:
        yaml.safe_dump(data_yaml, yf, sort_keys=False)
        
    # Update train_yolo.py pointer
    print(f"[SUCCESS] Unified YOLOv11 dataset created at: {MERGED_DIR.name}")
    print(f"[OK] Consolidated data.yaml generated successfully at: {yaml_path}")
    print("\nTo train YOLOv11 on this super-dataset, simply update DATA_YAML in 'train_yolo.py'")
    print("to point to: 'external_datasets/merged_yolo/data.yaml'")
    print("=====================================================================")

if __name__ == "__main__":
    main()
