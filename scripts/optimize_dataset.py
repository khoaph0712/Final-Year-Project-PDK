#!/usr/bin/env python
"""
Final Year Project (FYP) - Waste Sorting & Classification System
Dataset Optimizer & High-Integrity Cleaner Utility (Bảo dưỡng & Làm sạch Dataset)

This script performs 4 key diagnostic and cleaning operations on the YOLO dataset:
1. File Integrity Scan: Identifies and prunes corrupted or unreadable images.
2. Label & Coordinate Sanitization: Clamps out-of-bounds YOLO coordinates [0, 1] and fixes format issues.
3. Microscopic Noise Filter: Removes micro bounding boxes (smaller than 12x12 px) that act as training noise.
4. Train-Test Leakage Audit: Identifies and deletes duplicate images between train/val/test splits via MD5 hashes.
"""

import os
import sys
import hashlib
import yaml
from pathlib import Path
from collections import Counter
import cv2
import numpy as np

# Setup paths
SCRIPTS_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPTS_DIR.parent
DATASET_DIR = ROOT_DIR / "external_datasets" / "super_yolo_dataset"
DATA_YAML_PATH = DATASET_DIR / "data.yaml"

def calculate_md5(file_path):
    """Calculate MD5 checksum to find exact duplicates."""
    hash_md5 = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()

def clean_and_optimize_dataset(dataset_dir):
    print("=====================================================================")
    print("      FYP DATASET OPTIMIZER & CLEANER: DETECT & SANITIZE NOISE       ")
    print("=====================================================================")
    
    if not dataset_dir.exists():
        print(f"[ERROR] Target dataset directory not found: {dataset_dir}")
        sys.exit(1)
        
    splits = ["train", "val", "test"]
    
    # Statistics tracker
    stats = {
        "corrupted_images_removed": 0,
        "empty_labels_removed": 0,
        "out_of_bounds_coords_fixed": 0,
        "micro_boxes_pruned": 0,
        "duplicate_leaks_removed": 0,
        "valid_images_scanned": 0,
        "class_counts_before": Counter(),
        "class_counts_after": Counter()
    }
    
    # Step 1: Scan for Data Leakage (Duplicates across Train, Val, Test splits)
    print("\n[STEP 1] Running Train-Test Data Leakage Audit (MD5 Fingerprinting)...")
    split_hashes = {split: {} for split in splits}
    all_hashes = {}
    
    # Collect MD5 finger prints
    for split in splits:
        img_dir = dataset_dir / split / "images"
        if not img_dir.exists():
            continue
        for img_path in img_dir.glob("*"):
            if img_path.suffix.lower() not in [".jpg", ".jpeg", ".png", ".bmp", ".webp"]:
                continue
            try:
                h = calculate_md5(img_path)
                split_hashes[split][h] = img_path
                all_hashes[h] = all_hashes.get(h, []) + [(split, img_path)]
            except Exception as e:
                print(f"[WARNING] Could not calculate hash for {img_path}: {e}")
                
    # Detect duplicates
    duplicates_to_prune = []
    for h, locations in all_hashes.items():
        if len(locations) > 1:
            # We have a duplicate! If it appears in different splits, keep it ONLY in 'train'
            # and delete it from 'val' or 'test' to prevent data leakage.
            splits_present = [loc[0] for loc in locations]
            if len(set(splits_present)) > 1: # Leakage across splits!
                # Sort: keep 'train', then 'val', then 'test'
                locations_sorted = sorted(locations, key=lambda x: {"train": 0, "val": 1, "test": 2}[x[0]])
                keep_loc = locations_sorted[0]
                prune_locs = locations_sorted[1:]
                
                print(f"[DATA LEAKAGE DETECTED] Image hash {h[:8]}...")
                print(f"  - Keeping in split: '{keep_loc[0]}' ({keep_loc[1].name})")
                for p in prune_locs:
                    print(f"  - Pruning duplicate from split: '{p[0]}' ({p[1].name}) [LEAK PREVENTED]")
                    duplicates_to_prune.append(p[1])
            else:
                # Duplicate within the same split: keep the first one, prune the rest
                for p in locations[1:]:
                    print(f"[DUPLICATE IN SPLIT] Pruning extra copy {p[1].name} inside '{p[0]}'")
                    duplicates_to_prune.append(p[1])
                    
    # Prune duplicate files
    for dup_img_path in duplicates_to_prune:
        dup_lbl_path = Path(str(dup_img_path).replace("/images", "/labels").replace("\\images", "\\labels")).with_suffix(".txt")
        try:
            if dup_img_path.exists():
                dup_img_path.unlink()
            if dup_lbl_path.exists():
                dup_lbl_path.unlink()
            stats["duplicate_leaks_removed"] += 1
        except Exception as e:
            print(f"[ERROR] Failed to delete duplicate {dup_img_path}: {e}")

    # Step 2: Scan for Image Integrity & Bounding Box Optimization
    print("\n[STEP 2] Sanitizing File Integrity, Coordinates, and Small Bounding Boxes...")
    
    # Class names mapping
    classes = ["plastic", "glass", "metal", "paper", "cardboard", "organic"]
    
    for split in splits:
        img_dir = dataset_dir / split / "images"
        lbl_dir = dataset_dir / split / "labels"
        
        if not img_dir.exists() or not lbl_dir.exists():
            continue
            
        print(f"Processing split: '{split}'...")
        
        for img_path in list(img_dir.glob("*")):
            if img_path.suffix.lower() not in [".jpg", ".jpeg", ".png", ".bmp", ".webp"]:
                continue
                
            lbl_path = lbl_dir / img_path.with_suffix(".txt").name
            
            # A: Verify Image Integrity (Try to decode image bytes via OpenCV)
            img = cv2.imread(str(img_path))
            if img is None:
                print(f"  [CORRUPTED IMAGE] Pruning unreadable image: {img_path.name}")
                try:
                    img_path.unlink()
                    if lbl_path.exists():
                        lbl_path.unlink()
                    stats["corrupted_images_removed"] += 1
                except Exception as e:
                    print(f"    Failed to delete: {e}")
                continue
                
            h_img, w_img = img.shape[:2]
            stats["valid_images_scanned"] += 1
            
            # B: Optimize labels if text file exists
            if lbl_path.exists():
                valid_lines = []
                has_updates = False
                
                with open(lbl_path, "r", encoding="utf-8") as lf:
                    lines = lf.readlines()
                    
                for line in lines:
                    parts = line.strip().split()
                    if len(parts) < 5:
                        continue
                    try:
                        cid = int(parts[0])
                        x_center = float(parts[1])
                        y_center = float(parts[2])
                        w_box = float(parts[3])
                        h_box = float(parts[4])
                    except ValueError:
                        continue
                        
                    if cid < 0 or cid >= len(classes):
                        continue
                        
                    stats["class_counts_before"][classes[cid]] += 1
                    
                    # 1. Coordinate Sanitization: clamp values between 0.0 and 1.0
                    x_clamped = np.clip(x_center, 0.0, 1.0)
                    y_clamped = np.clip(y_center, 0.0, 1.0)
                    w_clamped = np.clip(w_box, 0.001, 1.0)
                    h_clamped = np.clip(h_box, 0.001, 1.0)
                    
                    if (x_clamped != x_center or y_clamped != y_center or 
                        w_clamped != w_box or h_clamped != h_box):
                        has_updates = True
                        stats["out_of_bounds_coords_fixed"] += 1
                        
                    # 2. Microscopic Bounding Box Filtering
                    # Bboxes smaller than 12x12 px on original resolution act as label noise
                    w_pixels = w_clamped * w_img
                    h_pixels = h_clamped * h_img
                    
                    if w_pixels < 12.0 or h_pixels < 12.0:
                        stats["micro_boxes_pruned"] += 1
                        has_updates = True
                        # Skip this box entirely (prune it!)
                        print(f"  [MICRO BOX FILTERED] Pruned {w_pixels:.1f}x{h_pixels:.1f}px box from {img_path.name}")
                        continue
                        
                    valid_lines.append(f"{cid} {x_clamped:.6f} {y_clamped:.6f} {w_clamped:.6f} {h_clamped:.6f}")
                    stats["class_counts_after"][classes[cid]] += 1
                    
                # Rewrite label file if changes made
                if has_updates:
                    if valid_lines:
                        with open(lbl_path, "w", encoding="utf-8") as lf:
                            lf.write("\n".join(valid_lines) + "\n")
                    else:
                        # No valid boxes left! Prune the empty label and image to keep dataset clean
                        print(f"  [EMPTY LABEL] Bounding box audit emptied {img_path.name}. Pruning sample.")
                        try:
                            img_path.unlink()
                            lbl_path.unlink()
                            stats["empty_labels_removed"] += 1
                        except Exception as e:
                            print(f"    Failed to prune empty sample: {e}")
                            
    # Write diagnostic summary
    print("\n=====================================================================")
    print("                        DIAGNOSTIC REPORT SUMMARY                    ")
    print("=====================================================================")
    print(f"  * Valid Images Scanned: {stats['valid_images_scanned']}")
    print(f"  * Corrupted Images Removed: {stats['corrupted_images_removed']}")
    print(f"  * Out-Of-Bounds Coordinates Clamped: {stats['out_of_bounds_coords_fixed']}")
    print(f"  * Microscopic Box Noise Pruned (<12px): {stats['micro_boxes_pruned']}")
    print(f"  * Data Leakage Images Pruned (Duplicates): {stats['duplicate_leaks_removed']}")
    print(f"  * Empty Label Samples Cleaned: {stats['empty_labels_removed']}")
    
    print("\nClass Distribution Shifts (Before vs After Cleanup):")
    print(f"  {'Class Name':<15} | {'Before':<10} | {'After':<10} | {'Diff':<10}")
    print("-" * 50)
    for cname in classes:
        bef = stats["class_counts_before"][cname]
        aft = stats["class_counts_after"][cname]
        diff = aft - bef
        print(f"  {cname:<15} | {bef:<10} | {aft:<10} | {diff:<10}")
    print("=====================================================================")
    
    # Save a comprehensive markdown report for thesis documentation
    report_path = ROOT_DIR / "runs" / "dataset_eda" / "dataset_clean_report.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("# Báo Cáo Bảo Dưỡng & Làm Sạch Dữ Liệu Học Máy (Dataset Audit & Clean Report)\n\n")
        f.write("Báo cáo này lập hồ sơ lưu trữ khoa học về việc loại bỏ nhiễu nhãn và rò rỉ dữ liệu để chuẩn bị cho hội đồng bảo vệ khóa luận.\n\n")
        
        f.write("## 1. Kết Quả Bảo Dưỡng Tổng Quan\n")
        f.write(f"- **Tổng số ảnh quét thành công:** {stats['valid_images_scanned']}\n")
        f.write(f"- **Số lượng ảnh lỗi/hỏng bị loại bỏ:** {stats['corrupted_images_removed']}\n")
        f.write(f"- **Số lượng tọa độ nhãn tràn viền được chuẩn hóa:** {stats['out_of_bounds_coords_fixed']}\n")
        f.write(f"- **Số lượng hộp giới hạn siêu nhỏ (Nhiễu <12x12px) bị cắt bỏ:** {stats['micro_boxes_pruned']}\n")
        f.write(f"- **Số lượng ảnh trùng lặp gây rò rỉ dữ liệu (Data Leakage) giữa Train-Test bị xóa:** {stats['duplicate_leaks_removed']}\n")
        f.write(f"- **Số lượng mẫu trống (không còn hộp giới hạn) bị loại bỏ:** {stats['empty_labels_removed']}\n\n")
        
        f.write("## 2. Thống Kê Sự Thay Đổi Số Lượng Nhãn\n")
        f.write("| Tên Lớp (Class) | Trước Khi Lọc | Sau Khi Lọc | Thay Đổi (Cắt Nhiễu) |\n")
        f.write("| :--- | :---: | :---: | :---: |\n")
        for cname in classes:
            bef = stats["class_counts_before"][cname]
            aft = stats["class_counts_after"][cname]
            diff = aft - bef
            f.write(f"| **{cname.upper()}** | {bef} | {aft} | {diff} |\n")
            
        f.write("\n## 3. Ý Nghĩa Khoa Học Của Việc Làm Sạch Dữ Liệu\n")
        f.write("> [!IMPORTANT]\n")
        f.write("> **Chống rò rỉ dữ liệu (Data Leakage):** Việc loại bỏ hoàn toàn các ảnh trùng lặp vật lý giữa tập huấn luyện (Train) và tập kiểm thử (Test) đảm bảo rằng các chỉ số F1-Score, mAP đạt được ở giai đoạn đánh giá là **hoàn toàn chính xác, trung thực và khoa học**, không bị thổi phồng ảo.\n")
        f.write(">\n")
        f.write("> [!TIP]\n")
        f.write("> **Cắt lọc nhiễu vi mô (Microscopic Noise Filtering):** Các hộp giới hạn siêu nhỏ (<12px) thường sinh ra do lỗi đánh nhãn bằng tay trên các điểm ảnh mờ ở hậu cảnh xa. Nếu giữ lại, chúng sẽ ép mô hình YOLO phải học các chi tiết không có thật, làm giảm khả năng hội tụ và gây ra hiện tượng Overfitting.\n")

    print(f"[OK] High-integrity markdown report written to: {report_path}")
    print("[SUCCESS] Dataset cleaning and optimization complete!")

if __name__ == "__main__":
    clean_and_optimize_dataset(DATASET_DIR)
