#!/usr/bin/env python
"""
Final Year Project (FYP) - Waste Sorting & Classification System
Phase 7: Programmatic External Crop Extractor

This script parses and extracts clean cropped images of objects from official raw datasets:
1. TACO: Parses annotations.json, crops high-resolution objects, maps 60 categories to 7 target classes.
2. TACO Backgrounds: Programmatically extracts random background patches (non-waste regions) to serve as negative training samples.
3. WaDaBa: Copies high-quality plastic waste items (PET, PP, HDPE, PS) directly into the 'plastic' class.
4. TrashNet: Copies clean, curated benchmark images into matching categories.

All crops are saved under C:\\FYP_v2\\external_crops\\<class_name>\\
"""

import os
import sys
import json
import random
import shutil
import time
import cv2
import numpy as np
from pathlib import Path
from collections import Counter

random.seed(42)

ROOT_DIR = Path(__file__).resolve().parent.parent
TACO_DIR = ROOT_DIR / "external_datasets" / "taco_official" / "data"
WADABA_DIR = ROOT_DIR / "external_datasets" / "wadaba"
TRASHNET_DIR = ROOT_DIR / "external_datasets" / "trashnet" / "data" / "dataset-resized"
OUT_DIR = ROOT_DIR / "external_crops"

TARGET_CLASSES = ["plastic", "glass", "metal", "paper", "cardboard", "organic", "Background"]

# Smart mapping from 60 official TACO categories to our 7 target classes
# None means the category is dropped/skipped due to ambiguity or lack of clean samples.
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
    51: None,              # Rope & strings (skip to prevent noise)
    52: "metal",          # Scrap metal
    53: None,              # Shoe (skip)
    54: "plastic",        # Squeezable tube
    55: "plastic",        # Plastic straw
    56: "paper",          # Paper straw
    57: "plastic",        # Styrofoam piece
    58: None,              # Unlabeled litter (skip)
    59: None               # Cigarette (skip)
}

def box_intersection_over_area(boxA, boxB):
    """Compute how much boxA overlaps with boxB relative to boxA's area."""
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[0] + boxA[2], boxB[0] + boxB[2])
    yB = min(boxA[1] + boxA[3], boxB[1] + boxB[3])
    
    interW = max(0, xB - xA)
    interH = max(0, yB - yA)
    interArea = interW * interH
    
    areaA = boxA[2] * boxA[3]
    if areaA == 0:
        return 0
    return interArea / areaA

def extract_taco_crops():
    print("\n--- Ingesting and Cropping TACO Official Dataset ---")
    ann_file = TACO_DIR / "annotations.json"
    if not ann_file.exists():
        print(f"[WARN] TACO annotations not found at: {ann_file}. Skipping TACO.")
        return 0
        
    with open(ann_file, "r") as f:
        coco = json.load(f)
        
    images = {img["id"]: img for img in coco["images"]}
    annotations = coco["annotations"]
    
    # Pre-group annotations by image_id
    anns_by_img = {}
    for ann in annotations:
        iid = ann["image_id"]
        anns_by_img.setdefault(iid, []).append(ann)
        
    crop_counts = Counter()
    bg_counts = 0
    
    print(f"[INFO] Loaded {len(images)} images and {len(annotations)} annotations.")
    
    for idx, (iid, img_info) in enumerate(images.items()):
        if idx > 0 and idx % 200 == 0:
            print(f"  Processed {idx}/{len(images)} images...")
            
        file_path = TACO_DIR / img_info["file_name"]
        if not file_path.exists():
            continue
            
        img = cv2.imread(str(file_path))
        if img is None:
            continue
            
        h_img, w_img = img.shape[:2]
        img_anns = anns_by_img.get(iid, [])
        
        # 1. Extract Waste Crops
        waste_boxes = []
        for ann_idx, ann in enumerate(img_anns):
            cat_id = ann["category_id"]
            target_class = TACO_CLASS_MAP.get(cat_id)
            if target_class is None:
                continue
                
            # COCO bbox: [x_min, y_min, width, height]
            x, y, w, h = [int(v) for v in ann["bbox"]]
            
            # Save actual bounding box coordinates for background generation checking
            waste_boxes.append((x, y, w, h))
            
            # Add mild padding
            pad_x = int(w * 0.05)
            pad_y = int(h * 0.05)
            x1 = max(0, x - pad_x)
            y1 = max(0, y - pad_y)
            x2 = min(w_img, x + w + pad_x)
            y2 = min(h_img, y + h + pad_y)
            
            crop = img[y1:y2, x1:x2]
            if crop.size > 0 and w > 15 and h > 15:
                # Save crop
                class_dir = OUT_DIR / target_class
                class_dir.mkdir(parents=True, exist_ok=True)
                
                crop_name = f"taco_img_{iid}_ann_{ann['id']}.jpg"
                cv2.imwrite(str(class_dir / crop_name), crop)
                crop_counts[target_class] += 1
                
        # 2. Extract Background Negative Crops (to handle complex lawns, pavements)
        # Try to generate up to 2 background crops per image
        if len(waste_boxes) > 0:
            for attempt in range(10):
                bg_w = random.randint(200, 400)
                bg_h = random.randint(200, 400)
                bg_x = random.randint(0, w_img - bg_w)
                bg_y = random.randint(0, h_img - bg_h)
                bg_box = (bg_x, bg_y, bg_w, bg_h)
                
                # Check overlap with any active waste box
                overlap = False
                for w_box in waste_boxes:
                    if box_intersection_over_area(bg_box, w_box) > 0.02:
                        overlap = True
                        break
                        
                if not overlap:
                    bg_crop = img[bg_y:bg_y+bg_h, bg_x:bg_x+bg_w]
                    if bg_crop.size > 0:
                        class_dir = OUT_DIR / "Background"
                        class_dir.mkdir(parents=True, exist_ok=True)
                        
                        crop_name = f"taco_bg_img_{iid}_attempt_{attempt}.jpg"
                        cv2.imwrite(str(class_dir / crop_name), bg_crop)
                        bg_counts += 1
                        if bg_counts >= 1500:  # Cap TACO background generation
                            break
                            
    print(f"[OK] Successfully extracted TACO crops:")
    for cls, cnt in crop_counts.items():
        print(f"  - {cls.upper()}: {cnt} crops")
    print(f"  - BACKGROUND: {bg_counts} negative crops")
    return sum(crop_counts.values()) + bg_counts

def extract_wadaba_crops():
    print("\n--- Ingesting WaDaBa Plastic Waste Dataset ---")
    if not WADABA_DIR.exists():
        print(f"[WARN] WaDaBa folder not found at: {WADABA_DIR}. Skipping WaDaBa.")
        return 0
        
    class_dir = OUT_DIR / "plastic"
    class_dir.mkdir(parents=True, exist_ok=True)
    
    count = 0
    img_exts = {".jpg", ".jpeg", ".png", ".bmp"}
    
    # Iterate through all 20 directories
    for sub in WADABA_DIR.glob("WaDaBa_*"):
        if sub.is_dir() and sub.name != "_zips":
            for img_path in sub.glob("*"):
                if img_path.suffix.lower() in img_exts:
                    # In WaDaBa, the item is already center-focused on a desk/conveyor belt
                    # We copy the entire image as a valid crop to class 'plastic'
                    dest_name = f"wadaba_{sub.name}_{img_path.name}"
                    shutil.copy2(img_path, class_dir / dest_name)
                    count += 1
                    
    print(f"[OK] Successfully extracted {count} WaDaBa PLASTIC crops.")
    return count

def extract_trashnet_crops():
    print("\n--- Ingesting TrashNet Stanford Benchmark ---")
    if not TRASHNET_DIR.exists():
        print(f"[WARN] TrashNet folder not found at: {TRASHNET_DIR}. Skipping TrashNet.")
        return 0
        
    class_mapping = {
        "glass": "glass",
        "paper": "paper",
        "cardboard": "cardboard",
        "plastic": "plastic",
        "metal": "metal",
        "trash": None  # Skip ambiguous generic trash to preserve clean classes
    }
    
    img_exts = {".jpg", ".jpeg", ".png"}
    total_copied = 0
    
    for src_cls, target_cls in class_mapping.items():
        if target_cls is None:
            continue
            
        src_dir = TRASHNET_DIR / src_cls
        if not src_dir.exists():
            continue
            
        dest_dir = OUT_DIR / target_cls
        dest_dir.mkdir(parents=True, exist_ok=True)
        
        count = 0
        for img_path in src_dir.glob("*"):
            if img_path.suffix.lower() in img_exts:
                dest_name = f"trashnet_{img_path.name}"
                shutil.copy2(img_path, dest_dir / dest_name)
                count += 1
                total_copied += 1
                
        print(f"  - {target_cls.upper()}: {count} crops copied")
        
    print(f"[OK] Successfully extracted {total_copied} TrashNet crops.")
    return total_copied

def main():
    if OUT_DIR.exists():
        print(f"[INFO] Cleaning up prior external crops directory: {OUT_DIR}")
        shutil.rmtree(OUT_DIR)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    
    t_start = time.time()
    
    taco_total = extract_taco_crops()
    wadaba_total = extract_wadaba_crops()
    trashnet_total = extract_trashnet_crops()
    
    total = taco_total + wadaba_total + trashnet_total
    print("\n" + "="*60)
    print("      EXTERNAL CROPS EXTRACTION COMPLETE      ")
    print("="*60)
    print(f"  - Total TACO Crops: {taco_total}")
    print(f"  - Total WaDaBa Crops: {wadaba_total}")
    print(f"  - Total TrashNet Crops: {trashnet_total}")
    print(f"  - GRAND TOTAL SAVED CROPS: {total}")
    print(f"  - Time elapsed: {time.time() - t_start:.2f} seconds.")
    print("="*60)
    
    # Audit counts per folder
    print("\nAuditing Final Folder Distribution:")
    for cls in TARGET_CLASSES:
        cls_path = OUT_DIR / cls
        cnt = len(list(cls_path.glob("*"))) if cls_path.exists() else 0
        print(f"  * {cls.upper():<12}: {cnt} images")

if __name__ == "__main__":
    main()
