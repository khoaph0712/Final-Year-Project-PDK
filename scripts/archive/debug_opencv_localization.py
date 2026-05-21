import sys
import os
import random
import time
from pathlib import Path
import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
TACO_DIR = ROOT / "external_datasets" / "taco_official" / "data"
OUT_DIR = ROOT / "runs" / "dl" / "hybrid_pipeline_results" / "debug"

def analyze_image_proposals(image_path, min_area=400):
    img = cv2.imread(str(image_path))
    if img is None:
        return None
        
    h_img, w_img = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # 1. Otsu Thresholding
    _, thresh_otsu = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # 2. Adaptive Gaussian Thresholding (Handles shadows and complex lighting better)
    thresh_adaptive = cv2.adaptiveThreshold(
        blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY_INV, 11, 2
    )
    
    # 3. Canny Edges
    edges = cv2.Canny(blur, 30, 100)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    dilated_edges = cv2.dilate(edges, kernel, iterations=1)
    
    # Combined Otsu vs Combined Adaptive
    combined_otsu = cv2.bitwise_or(thresh_otsu, dilated_edges)
    combined_otsu = cv2.morphologyEx(combined_otsu, cv2.MORPH_CLOSE, kernel)
    
    combined_adaptive = cv2.bitwise_or(thresh_adaptive, dilated_edges)
    combined_adaptive = cv2.morphologyEx(combined_adaptive, cv2.MORPH_CLOSE, kernel)
    
    # Count contours
    contours_otsu, _ = cv2.findContours(combined_otsu, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours_adaptive, _ = cv2.findContours(combined_adaptive, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    valid_otsu = sum(1 for c in contours_otsu if cv2.contourArea(c) > min_area)
    valid_adaptive = sum(1 for c in contours_adaptive if cv2.contourArea(c) > min_area)
    
    # Analyze aspect ratio issues (boxes that are too huge)
    huge_boxes_otsu = 0
    img_area = h_img * w_img
    for c in contours_otsu:
        if cv2.contourArea(c) > min_area:
            _, _, w, h = cv2.boundingRect(c)
            if (w * h) > img_area * 0.4:  # Bounding box takes up > 40% of the image
                huge_boxes_otsu += 1
                
    return {
        "size": (w_img, h_img),
        "otsu_total_cnt": len(contours_otsu),
        "otsu_valid_cnt": valid_otsu,
        "otsu_huge_boxes": huge_boxes_otsu,
        "adaptive_total_cnt": len(contours_adaptive),
        "adaptive_valid_cnt": valid_adaptive,
        "masks": {
            "otsu": combined_otsu,
            "adaptive": combined_adaptive,
            "img": img
        }
    }

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if not TACO_DIR.exists():
        print(f"[ERROR] TACO data not found at {TACO_DIR}")
        return
        
    taco_images = []
    for batch_dir in TACO_DIR.glob("batch_*"):
        if batch_dir.is_dir():
            taco_images.extend(list(batch_dir.glob("*.jpg")) + list(batch_dir.glob("*.JPG")))
            
    if not taco_images:
        print("[ERROR] No images found inside TACO")
        return
        
    random.seed(42)
    selected = random.sample(taco_images, min(5, len(taco_images)))
    
    print("=====================================================================")
    print("      OPENCV LOCALIZATION FAILURE DIAGNOSTICS & AUDIT SCRIPT         ")
    print("=====================================================================")
    
    for idx, p in enumerate(selected):
        res = analyze_image_proposals(p)
        if res is None:
            continue
            
        print(f"\nImage #{idx+1}: {p.name} | Dimensions: {res['size'][0]}x{res['size'][1]}")
        print(f"  - Otsu Thresholding: {res['otsu_total_cnt']} contours total, {res['otsu_valid_cnt']} valid (>400px)")
        print(f"  - Otsu Huge Box Failures (Box takes >40% of image): {res['otsu_huge_boxes']}")
        print(f"  - Adaptive Gaussian Thresholding: {res['adaptive_total_cnt']} contours total, {res['adaptive_valid_cnt']} valid (>400px)")
        
        # Save side-by-side debug masks for visual diagnostics
        img = res["masks"]["img"]
        # Draw all Otsu contours on a copy of the image to visualize the boxes
        otsu_draw = img.copy()
        for cnt in cv2.findContours(res["masks"]["otsu"], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]:
            if cv2.contourArea(cnt) > 400:
                x, y, w, h = cv2.boundingRect(cnt)
                cv2.rectangle(otsu_draw, (x, y), (x+w, y+h), (0, 0, 255), 3)
                
        # Draw all Adaptive contours on another copy
        adaptive_draw = img.copy()
        for cnt in cv2.findContours(res["masks"]["adaptive"], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]:
            if cv2.contourArea(cnt) > 400:
                x, y, w, h = cv2.boundingRect(cnt)
                cv2.rectangle(adaptive_draw, (x, y), (x+w, y+h), (0, 255, 0), 3)
                
        # Write side-by-side original-with-contours and thresholds
        h, w = img.shape[:2]
        # Resize to manageable size for saving
        scale = 800 / max(h, w)
        img_small = cv2.resize(img, (int(w*scale), int(h*scale)))
        otsu_draw_small = cv2.resize(otsu_draw, (int(w*scale), int(h*scale)))
        adaptive_draw_small = cv2.resize(adaptive_draw, (int(w*scale), int(h*scale)))
        
        combined_debug = np.hstack((img_small, otsu_draw_small, adaptive_draw_small))
        out_path = OUT_DIR / f"debug_loc_{p.stem}.png"
        cv2.imwrite(str(out_path), combined_debug)
        print(f"  [OK] Diagnostic visual saved to: {out_path}")

if __name__ == "__main__":
    main()
