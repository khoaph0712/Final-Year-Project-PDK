#!/usr/bin/env python
"""
Final Year Project (FYP) - Waste Sorting & Classification System
YOLOv8 Inference & Visual Evaluation Script

This script:
1. Loads our trained YOLOv8 model weights (or defaults to pre-trained yolov8n.pt if not yet trained).
2. Performs full image end-to-end detection and classification.
3. Overlays bounding boxes, class labels, and confidence percentages.
4. Saves visualization plots under 'runs/detect/inference_results/'.
5. Measures and reports edge inference latency (ms) for performance comparison.
"""

import os
import sys
import time
from pathlib import Path
import cv2
import matplotlib.pyplot as plt

# Setup paths
ROOT_DIR = Path(__file__).resolve().parent.parent
DEFAULT_WEIGHTS = ROOT_DIR / "runs" / "detect" / "yolov11_taco" / "weights" / "best.pt"
OUTPUT_DIR = ROOT_DIR / "runs" / "detect" / "inference_results"

def check_ultralytics():
    try:
        import ultralytics
        return True
    except ImportError:
        print("[ERROR] 'ultralytics' package is missing.")
        print("Please install it in your active virtual environment via: pip install ultralytics")
        return False

def main():
    print("=====================================================================")
    print("                 YOLOv11 Edge Inference & Visualizer                 ")
    print("=====================================================================")
    
    if not check_ultralytics():
        sys.exit(1)
        
    from ultralytics import YOLO
    
    # 1. Load weights
    weights_path = DEFAULT_WEIGHTS
    if not weights_path.exists():
        print(f"[WARNING] Mapped custom weights not found at: {weights_path}")
        print("[INFO] Falling back to official COCO pre-trained 'yolo11n.pt' for structural verification...")
        weights_path = "yolo11n.pt"
    else:
        print(f"[OK] Found custom trained YOLOv11 weights at: {weights_path}")
        
    try:
        model = YOLO(weights_path)
        print("[OK] YOLOv8 Model loaded successfully.")
    except Exception as e:
        print(f"[ERROR] Failed to load YOLOv8 model: {str(e)}")
        sys.exit(1)
        
    # 2. Select Test Image
    test_image_dir = ROOT_DIR / "external_datasets" / "taco_official" / "data"
    
    # Let's search for an image in TACO dataset
    test_images = list(test_image_dir.rglob("*.jpg")) + list(test_image_dir.rglob("*.JPG")) + list(test_image_dir.rglob("*.png"))
    if not test_images:
        print(f"[ERROR] No images found under TACO directory: {test_image_dir}")
        sys.exit(1)
        
    # Pick a popular test image
    test_image = None
    for img_path in test_images:
        if "000033" in img_path.name or "000090" in img_path.name:
            test_image = img_path
            break
            
    if test_image is None:
        test_image = test_images[0]
        
    print(f"[INFO] Running inference on sample image: {test_image.name}")
    
    # 3. Read and Run Inference
    start_time = time.time()
    results = model(str(test_image))
    latency_ms = (time.time() - start_time) * 1000
    
    print(f"[OK] Detection complete! Inference Latency: {latency_ms:.2f} ms")
    
    # 4. Save and Plot Result
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    res = results[0]
    
    # Plotting using matplotlib alongside original
    annotated_img_bgr = res.plot()  # Returns BGR numpy array
    annotated_img_rgb = cv2.cvtColor(annotated_img_bgr, cv2.COLOR_BGR2RGB)
    
    dest_path = OUTPUT_DIR / f"yolo_det_{test_image.name}"
    cv2.imwrite(str(dest_path), annotated_img_bgr)
    print(f"[SUCCESS] Annotated visual result saved to: {dest_path}")
    
    # Print out bounding boxes details
    print("\n--- Detected Objects Summary ---")
    boxes = res.boxes
    if len(boxes) == 0:
        print("  - No objects detected.")
    else:
        for idx, box in enumerate(boxes):
            cls_id = int(box.cls[0])
            cls_name = model.names[cls_id]
            conf = float(box.conf[0]) * 100
            xyxy = box.xyxy[0].tolist()
            print(f"  [{idx + 1}] Class: {cls_name.upper()} | Conf: {conf:.1f}% | Box: [{', '.join(f'{coord:.1f}' for coord in xyxy)}]")
            
    print("=====================================================================")

if __name__ == "__main__":
    main()
