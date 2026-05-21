#!/usr/bin/env python
"""
Final Year Project (FYP) - Waste Sorting & Classification System
2-Stage Real-World Hierarchical Pipeline (YOLOv11 Detector + MobileNetV2 CNN Classifier)

This script implements the ultimate 2-stage hierarchical waste-sorting pipeline:
- Stage 1 (YOLOv11): Performs fast, state-of-the-art waste object detection and localization (bounding box extraction).
- Stage 2 (MobileNetV2 CNN): Validates each object crop using a highly-trained, balanced deep classifier to reduce false alarms and increase classification precision.
"""

import sys
import os
import argparse
import random
import time
from pathlib import Path
import numpy as np
import cv2

# Quiet TensorFlow warnings
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import tensorflow as tf

# Robust Keras imports
try:
    import tf_keras as keras
except ImportError:
    from tensorflow import keras

# Setup paths
SCRIPTS_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPTS_DIR.parent
sys.path.append(str(SCRIPTS_DIR))

# Configuration Paths
DEFAULT_YOLO_WEIGHTS = ROOT_DIR / "runs" / "detect" / "yolov11_taco" / "weights" / "best.pt"
DEFAULT_CNN_WEIGHTS = ROOT_DIR / "runs" / "dl" / "cnn_mobilenet" / "best_mobilenet.h5"
TACO_DIR = ROOT_DIR / "external_datasets" / "super_yolo_dataset" / "test" / "images"
DEFAULT_OUT_DIR = ROOT_DIR / "runs" / "detect" / "yolo_mobilenet_pipeline"

# 6 Core classes for YOLOv11 detector, and 7 classes for MobileNetV2 CNN (includes Background)
YOLO_CLASSES = ["plastic", "glass", "metal", "paper", "cardboard", "organic"]
CNN_CLASSES = ["plastic", "glass", "metal", "paper", "cardboard", "organic", "Background"]

# Harmonics Color Palette for Premium Visualization
CLASS_COLORS = {
    "plastic": (46, 204, 113),     # Green
    "glass": (52, 152, 219),       # Blue
    "metal": (231, 76, 60),        # Red
    "paper": (241, 196, 15),       # Yellow
    "cardboard": (155, 89, 182),   # Purple/Magenta
    "organic": (26, 188, 156),     # Teal
    "Background": (149, 165, 166)  # Gray
}

def preprocess_crop(crop, target_size=(128, 128)):
    """Resize crop and preprocess for MobileNetV2."""
    resized = cv2.resize(crop, target_size, interpolation=cv2.INTER_LINEAR)
    resized_rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    crop_norm = (resized_rgb.astype(np.float32) / 127.5) - 1.0
    return np.expand_dims(crop_norm, axis=0)

def main():
    parser = argparse.ArgumentParser(description="FYP Waste Management: 2-Stage YOLOv11 + MobileNetV2 Hierarchical Pipeline")
    parser.add_argument("--image", type=str, help="Path to a complex test image.")
    parser.add_argument("--random", action="store_true", help="Auto-pick a random complex image from the official TACO dataset.")
    parser.add_argument("--yolo-weights", type=str, default=str(DEFAULT_YOLO_WEIGHTS), help="YOLOv11 weights path.")
    parser.add_argument("--cnn-weights", type=str, default=str(DEFAULT_CNN_WEIGHTS), help="MobileNetV2 Keras model weights path.")
    parser.add_argument("--yolo-conf", type=float, default=0.25, help="YOLOv11 detection confidence threshold.")
    parser.add_argument("--cnn-conf", type=float, default=0.40, help="MobileNetV2 verification confidence threshold.")
    parser.add_argument("--out-dir", type=str, default=str(DEFAULT_OUT_DIR), help="Output folder to write visual detection sheets.")
    parser.add_argument("--padding", type=int, default=8, help="Bounding box crop padding to preserve complete geometry details.")
    
    args = parser.parse_args()
    
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    print("=====================================================================")
    print("      FYP 2-Stage YOLOv11 + MobileNetV2 Hierarchical Pipeline       ")
    print("=====================================================================")
    
    # 1. Resolve and validate model weights
    yolo_weights_path = Path(args.yolo_weights)
    cnn_weights_path = Path(args.cnn_weights)
    
    if not yolo_weights_path.exists():
        print(f"[WARNING] Custom YOLOv11 weights not found at: {yolo_weights_path}")
        print("[INFO] Falling back to official 'yolo11n.pt' for structural verification...")
        yolo_weights_path = Path("yolo11n.pt")
        
    if not cnn_weights_path.exists():
        print(f"[ERROR] MobileNetV2 CNN weights not found at: {cnn_weights_path}")
        sys.exit(1)
        
    # 2. Select Image Input
    selected_image_path = None
    if args.image:
        selected_image_path = Path(args.image)
    elif args.random:
        if not TACO_DIR.exists():
            print(f"[ERROR] TACO dataset folder not found at: {TACO_DIR}")
            sys.exit(1)
            
        print(f"[INFO] Scanning test images at {TACO_DIR} for high-quality complex images...")
        taco_images = list(TACO_DIR.glob("*.jpg")) + list(TACO_DIR.glob("*.JPG")) + list(TACO_DIR.glob("*.png"))
                
        if not taco_images:
            print(f"[ERROR] No images found inside {TACO_DIR}!")
            sys.exit(1)
            
        selected_image_path = random.choice(taco_images)
        print(f"[OK] Randomly selected complex scene: {selected_image_path.name}")
    else:
        # Fallback check
        fallback_dir = TACO_DIR
        if fallback_dir.exists():
            test_images = list(fallback_dir.glob("*.jpg")) + list(fallback_dir.glob("*.png"))
            if test_images:
                selected_image_path = random.choice(test_images)
                print(f"[WARN] No image specified. Fallback to random test image: {selected_image_path.name}")
                
    if selected_image_path is None or not selected_image_path.exists():
        print(f"[ERROR] Input image path does not exist or is invalid! Path: {selected_image_path}")
        print("Please provide --image <path> or use --random to pull from the TACO dataset.")
        sys.exit(1)
        
    # 3. Load Image
    frame = cv2.imread(str(selected_image_path))
    if frame is None:
        print(f"[ERROR] Failed to decode image file: {selected_image_path}")
        sys.exit(1)
        
    annotated_frame = frame.copy()
    h_img, w_img = frame.shape[:2]
    print(f"[INFO] Image Dimension: {w_img}x{h_img} pixels.")
    
    # 4. Load Models into RAM
    print("\n[INFO] Loading Stage 1 and Stage 2 models into RAM...")
    t_start_load = time.time()
    
    # Stage 1: YOLOv11
    from ultralytics import YOLO
    yolo_model = YOLO(str(yolo_weights_path))
    print(f"  - Stage 1: YOLOv11 model loaded from {yolo_weights_path}")
    
    # Stage 2: MobileNetV2 CNN
    cnn_model = keras.models.load_model(str(cnn_weights_path))
    print(f"  - Stage 2: MobileNetV2 CNN model loaded from {cnn_weights_path}")
    
    print(f"[OK] Models loaded successfully in {time.time() - t_start_load:.2f} seconds.")
    
    # 5. Stage 1: Run YOLOv11 Object Detection
    print("\n[INFO] Running Stage 1: YOLOv11 object proposal & rough classification...")
    t_start_yolo = time.time()
    yolo_results = yolo_model.predict(str(selected_image_path), conf=args.yolo_conf, verbose=False)
    t_yolo_elapsed = time.time() - t_start_yolo
    
    boxes = yolo_results[0].boxes
    print(f"[OK] YOLOv11 proposed {len(boxes)} candidate boxes in {t_yolo_elapsed*1000:.1f}ms.")
    
    # 6. Stage 2: Verification using Keras MobileNetV2
    print("\n" + "="*85)
    print(f"{'Box ID':<8} | {'YOLO Class (Conf)':<22} | {'CNN Verified Class (Conf)':<28} | {'Decision'}")
    print("="*85)
    
    valid_detections = 0
    t_start_cnn = time.time()
    
    for idx, box in enumerate(boxes):
        box_id = idx + 1
        
        # Get coordinates
        xyxy = box.xyxy[0].tolist()
        x1, y1, x2, y2 = [int(v) for v in xyxy]
        
        # Add safe padding
        x1_pad = max(0, x1 - args.padding)
        y1_pad = max(0, y1 - args.padding)
        x2_pad = min(w_img, x2 + args.padding)
        y2_pad = min(h_img, y2 + args.padding)
        
        # Crop target object from high-resolution image
        crop = frame[y1_pad:y2_pad, x1_pad:x2_pad]
        if crop.size == 0:
            continue
            
        # Get YOLO rough predictions
        yolo_cls_id = int(box.cls[0])
        yolo_class = yolo_model.names[yolo_cls_id] if hasattr(yolo_model, 'names') else YOLO_CLASSES[yolo_cls_id]
        yolo_conf = float(box.conf[0])
        
        # Preprocess and feed crop to MobileNetV2
        crop_input = preprocess_crop(crop)
        cnn_probs = cnn_model.predict(crop_input, verbose=0)[0]
        cnn_pred_idx = np.argmax(cnn_probs)
        cnn_class = CNN_CLASSES[cnn_pred_idx]
        cnn_conf = cnn_probs[cnn_pred_idx]
        
        yolo_str = f"{yolo_class.upper()} ({yolo_conf*100:.1f}%)"
        cnn_str = f"{cnn_class.upper()} ({cnn_conf*100:.1f}%)"
        
        # Decision logic:
        # 1. If CNN predicts Background -> Filter out as false positive (Ghost Waste)
        # 2. If CNN confidence is below threshold -> Filter out as unreliable detection
        # 3. Otherwise -> Accept and use the unified decision
        if cnn_class == "Background":
            decision = "REJECTED (Ghost Waste Background)"
        elif cnn_conf < args.cnn_conf:
            decision = f"REJECTED (CNN Conf {cnn_conf*100:.1f}% < {args.cnn_conf*100:.0f}%)"
        else:
            decision = "ACCEPTED & VERIFIED ✅"
            valid_detections += 1
            
            # Draw premium bounding box
            color = CLASS_COLORS.get(cnn_class, (0, 255, 0))
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 3)
            cv2.circle(annotated_frame, (x1, y1), 5, color, cv2.FILLED)
            
            # Draw text label card
            label_text = f"{cnn_class.upper()} (Verified: {cnn_conf*100:.1f}%)"
            (text_w, text_h), baseline = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
            cv2.rectangle(
                annotated_frame, 
                (x1, y1 - text_h - 10), 
                (x1 + text_w + 10, y1), 
                color, 
                cv2.FILLED
            )
            cv2.putText(
                annotated_frame, 
                label_text, 
                (x1 + 5, y1 - 5), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                0.45, 
                (255, 255, 255) if color != (241, 196, 15) else (0, 0, 0),
                1, 
                cv2.LINE_AA
            )
            
        print(f"Box #{box_id:<5} | {yolo_str:<22} | {cnn_str:<28} | {decision}")
        
    t_cnn_elapsed = time.time() - t_start_cnn
    print("="*85)
    print(f"[OK] Stage 2 verification completed in {t_cnn_elapsed*1000:.1f}ms.")
    print(f"[SUCCESS] Pipeline complete! Localized and verified {valid_detections} valid waste objects.")
    
    # Save visual result
    out_img_name = f"stage2_hierarchical_{selected_image_path.stem}.png"
    out_img_path = out_dir / out_img_name
    cv2.imwrite(str(out_img_path), annotated_frame)
    print(f"\n[SUCCESS] Final verified visualization saved to:")
    print(f"  --> {out_img_path}")
    print("=====================================================================")

if __name__ == "__main__":
    main()
