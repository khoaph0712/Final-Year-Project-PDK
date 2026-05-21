#!/usr/bin/env python
"""
Final Year Project (FYP) - Waste Sorting & Classification System
YOLO Dataset Builder: TACO COCO JSON to YOLO Bounding Box Format (Mapped to 6 Core Classes)

This script:
1. Loads the official TACO annotations.json (COCO Format).
2. Maps the 60 highly sparse TACO categories to our 6 dense core classes:
   ['plastic', 'glass', 'metal', 'paper', 'cardboard', 'organic'] (Background is implicit in YOLO).
3. Converts COCO bounding boxes [x_min, y_min, width, height] to YOLO normalized [x_center, y_center, w_norm, h_norm].
4. Partitions the images into reproducible splits (80% Train, 10% Val, 10% Test) using a fixed seed.
5. Copies images and writes corresponding YOLO bounding box labels (.txt) to structured folders.
6. Generates a data.yaml file required for training YOLOv8/YOLOv11 natively.
"""

import os
import json
import random
import shutil
from collections import Counter
from pathlib import Path
import cv2
import yaml

# =====================================================================
# Configuration Paths
# =====================================================================
ROOT_DIR = Path(__file__).resolve().parent.parent
TACO_DATA_DIR = ROOT_DIR / "external_datasets" / "taco_official" / "data"
OUT_YOLO_DIR = ROOT_DIR / "external_datasets" / "taco_yolo"

TARGET_CLASSES = ["plastic", "glass", "metal", "paper", "cardboard", "organic"]
CLASS_TO_IDX = {name: idx for idx, name in enumerate(TARGET_CLASSES)}

# 60 official TACO categories mapped to our 6 core classes
TACO_CLASS_MAP = {
    0: "metal",           # Aluminium foil
    1: None,              # Battery (skip to prevent noise)
    2: "metal",           # Aluminium blister pack
    3: "cardboard",       # Carded blister pack
    4: "plastic",         # Other plastic bottle
    5: "plastic",         # Clear plastic bottle
    6: "glass",           # Glass bottle
    7: "plastic",         # Plastic bottle cap
    8: "metal",           # Metal bottle cap
    9: "glass",           # Broken glass
    10: "metal",          # Food Can
    11: "metal",          # Aerosol
    12: "metal",          # Drink can
    13: "cardboard",      # Toilet tube
    14: "cardboard",      # Other carton
    15: "cardboard",      # Egg carton
    16: "cardboard",      # Drink carton
    17: "cardboard",      # Corrugated carton
    18: "cardboard",      # Meal carton
    19: "cardboard",      # Pizza box
    20: "paper",          # Paper cup
    21: "plastic",        # Disposable plastic cup
    22: "plastic",        # Foam cup
    23: "glass",          # Glass cup
    24: "plastic",        # Other plastic cup
    25: "organic",        # Food waste
    26: "glass",          # Glass jar
    27: "plastic",        # Plastic lid
    28: "metal",          # Metal lid
    29: "plastic",        # Other plastic
    30: "paper",          # Magazine paper
    31: "paper",          # Tissues
    32: "paper",          # Wrapping paper
    33: "paper",          # Normal paper
    34: "paper",          # Paper bag
    35: "paper",          # Plastified paper bag
    36: "plastic",        # Plastic film
    37: "plastic",        # Six pack rings
    38: "plastic",        # Garbage bag
    39: "plastic",        # Other plastic wrapper
    40: "plastic",        # Single-use carrier bag
    41: "plastic",        # Polypropylene bag
    42: "plastic",        # Crisp packet
    43: "plastic",        # Spread tub
    44: "plastic",        # Tupperware
    45: "plastic",        # Disposable food container
    46: "plastic",        # Foam food container
    47: "plastic",        # Other plastic container
    48: "plastic",        # Plastic glooves
    49: "plastic",        # Plastic utensils
    50: "metal",          # Pop tab
    51: None,             # Rope & strings (skip to prevent noise)
    52: "metal",          # Scrap metal
    53: None,             # Shoe (skip)
    54: "plastic",        # Squeezable tube
    55: "plastic",        # Plastic straw
    56: "paper",          # Paper straw
    57: "plastic",        # Styrofoam piece
    58: None,             # Unlabeled litter (skip)
    59: None              # Cigarette (skip)
}

def to_yolo_format(bbox, img_w, img_h):
    """
    Converts COCO bbox [x_min, y_min, width, height] to YOLO normalized
    [x_center, y_center, w_norm, h_norm]
    """
    x_min, y_min, w_box, h_box = bbox
    if w_box <= 0 or h_box <= 0 or img_w <= 0 or img_h <= 0:
        return None
        
    x_center = x_min + (w_box / 2.0)
    y_center = y_min + (h_box / 2.0)
    
    # Normalize coordinates
    x_center_norm = x_center / img_w
    y_center_norm = y_center / img_h
    w_norm = w_box / img_w
    h_norm = h_box / img_h
    
    # Clip limits safely inside image boundary [0, 1]
    x_c = max(0.0, min(1.0, x_center_norm))
    y_c = max(0.0, min(1.0, y_center_norm))
    w_n = max(0.0, min(1.0, w_norm))
    h_n = max(0.0, min(1.0, h_norm))
    
    return x_c, y_c, w_n, h_n

def main():
    print("=====================================================================")
    print("          TACO Bounding Box Parser & YOLO Dataset Builder            ")
    print("=====================================================================")
    
    ann_file = TACO_DATA_DIR / "annotations.json"
    if not ann_file.exists():
        print(f"[ERROR] TACO annotations.json not found at: {ann_file}")
        return
        
    print(f"[INFO] Ingesting annotations from: {ann_file.name}...")
    with open(ann_file, 'r', encoding='utf-8') as f:
        coco = json.load(f)
        
    # Re-map categories
    images = {int(img["id"]): img for img in coco["images"]}
    annotations = coco.get("annotations", [])
    
    # Prepare index mapping by matching class tags
    mapped_anns_by_img = {}
    total_annotations = 0
    discarded_annotations = 0
    
    class_stats = Counter()
    
    for ann in annotations:
        total_annotations += 1
        img_id = int(ann["image_id"])
        cat_id = int(ann["category_id"])
        
        target_name = TACO_CLASS_MAP.get(cat_id)
        if target_name is None:
            discarded_annotations += 1
            continue
            
        yolo_cid = CLASS_TO_IDX[target_name]
        bbox = ann["bbox"]
        
        mapped_anns_by_img.setdefault(img_id, []).append({
            "class_id": yolo_cid,
            "class_name": target_name,
            "bbox": bbox
        })
        class_stats[target_name] += 1
        
    print(f"[OK] Ingested {total_annotations} total annotations.")
    print(f"  - Mapped to target classes: {total_annotations - discarded_annotations}")
    print(f"  - Discarded (ambiguous/noise): {discarded_annotations}")
    print("\n[INFO] Class Distribution after Mapping:")
    for cls, count in class_stats.items():
        print(f"  - {cls.upper()}: {count} annotations")
        
    # Filter images to only those containing valid target annotations
    valid_image_ids = [iid for iid, anns in mapped_anns_by_img.items() if anns]
    print(f"\n[OK] Valid images containing target waste classes: {len(valid_image_ids)}")
    
    # Split Dataset (80% Train, 10% Val, 10% Test)
    random.seed(42)
    random.shuffle(valid_image_ids)
    
    n_total = len(valid_image_ids)
    n_train = int(round(n_total * 0.80))
    n_val = int(round(n_total * 0.10))
    
    splits = {
        "train": valid_image_ids[:n_train],
        "val": valid_image_ids[n_train:n_train + n_val],
        "test": valid_image_ids[n_train + n_val:]
    }
    
    print("\n[INFO] Creating split distributions...")
    print(f"  - Train Split: {len(splits['train'])} images")
    print(f"  - Val Split: {len(splits['val'])} images")
    print(f"  - Test Split: {len(splits['test'])} images")
    
    # Create output directories
    for split in ["train", "val", "test"]:
        (OUT_YOLO_DIR / split / "images").mkdir(parents=True, exist_ok=True)
        (OUT_YOLO_DIR / split / "labels").mkdir(parents=True, exist_ok=True)
        
    print("\n[INFO] Writing files and labels to disk...")
    missing_images = 0
    written_images = 0
    
    for split_name, img_ids in splits.items():
        for iid in img_ids:
            img_info = images[iid]
            src_img_path = TACO_DATA_DIR / img_info["file_name"]
            
            if not src_img_path.exists():
                # Fallback to search inside folders recursively in case file name has paths
                match_files = list(TACO_DATA_DIR.rglob(Path(img_info["file_name"]).name))
                if match_files:
                    src_img_path = match_files[0]
                else:
                    missing_images += 1
                    continue
                    
            # Decode dimensions to verify
            img = cv2.imread(str(src_img_path))
            if img is None:
                missing_images += 1
                continue
                
            img_h, img_w = img.shape[:2]
            
            # Format lines
            yolo_lines = []
            for ann in mapped_anns_by_img[iid]:
                y_box = to_yolo_format(ann["bbox"], img_w, img_h)
                if y_box is not None:
                    cx, cy, bw, bh = y_box
                    yolo_lines.append(f"{ann['class_id']} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
                    
            if not yolo_lines:
                continue
                
            # Write image and label
            safe_name = img_info["file_name"].replace("/", "__").replace("\\", "__")
            dest_img_path = OUT_YOLO_DIR / split_name / "images" / safe_name
            dest_lbl_path = OUT_YOLO_DIR / split_name / "labels" / Path(safe_name).with_suffix(".txt").name
            
            shutil.copy2(src_img_path, dest_img_path)
            with open(dest_lbl_path, 'w', encoding='utf-8') as lf:
                lf.write("\n".join(yolo_lines) + "\n")
                
            written_images += 1
            
    print(f"\n[SUCCESS] Formatted and wrote dataset to: {OUT_YOLO_DIR.name}")
    print(f"  - Total successfully written images: {written_images}")
    print(f"  - Missing/skipped images: {missing_images}")
    
    # Generate data.yaml
    data_yaml = {
        "path": str(OUT_YOLO_DIR.resolve()),
        "train": "train/images",
        "val": "val/images",
        "test": "test/images",
        "nc": len(TARGET_CLASSES),
        "names": TARGET_CLASSES
    }
    
    yaml_path = OUT_YOLO_DIR / "data.yaml"
    with open(yaml_path, 'w', encoding='utf-8') as yf:
        yaml.safe_dump(data_yaml, yf, sort_keys=False)
        
    print(f"[OK] data.yaml generated successfully at: {yaml_path}")
    print("=====================================================================")

if __name__ == "__main__":
    main()
