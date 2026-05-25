#!/usr/bin/env python
"""
Final Year Project (FYP) - Waste Sorting & Classification System
2-Stage Real-World Hierarchical Pipeline (YOLOv11 Detector + EfficientNetB0 CNN Classifier)

This script implements the ultimate 2-stage hierarchical waste-sorting pipeline:
- Stage 1 (YOLOv11): Performs fast, state-of-the-art waste object detection and localization (bounding box extraction).
- Stage 2 (EfficientNetB0 CNN): Validates each object crop using a highly-trained, balanced deep classifier to reduce false alarms and increase classification precision.
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
DEFAULT_YOLO_WEIGHTS = ROOT_DIR / "runs" / "detect" / "yolov11_super_dataset" / "weights" / "best.pt"
DEFAULT_CNN_WEIGHTS = ROOT_DIR / "runs" / "dl" / "cnn_efficientnet_tuned" / "best_efficientnet_tuned.h5"
TACO_DIR = ROOT_DIR / "external_datasets" / "super_yolo_dataset" / "test" / "images"
DEFAULT_OUT_DIR = ROOT_DIR / "runs" / "detect" / "yolo_efficientnet_pipeline"

# 6 Core classes for YOLOv11 detector, and 7 classes for EfficientNetB0 CNN (includes Background)
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

def preprocess_crop(crop, target_size=(224, 224)):
    """Resize crop and preprocess for EfficientNetB0."""
    resized = cv2.resize(crop, target_size, interpolation=cv2.INTER_LINEAR)
    resized_rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    crop_float = resized_rgb.astype(np.float32)
    crop_preprocessed = keras.applications.efficientnet.preprocess_input(crop_float)
    return np.expand_dims(crop_preprocessed, axis=0)

def apply_bayesian_fusion(model_probs, context="default"):
    """
    Applies Multi-Modal Bayesian Fusion between CV model probabilities and Context Prior Metadata.
    model_probs: numpy array of shape (7,) representing [plastic, glass, metal, paper, cardboard, organic, Background]
    context: string representing the environmental context
    """
    CONTEXT_PRIORS = {
        "beach":    [0.35, 0.25, 0.20, 0.05, 0.05, 0.10],
        "grass":    [0.30, 0.15, 0.15, 0.20, 0.05, 0.15],
        "indoor":   [0.20, 0.10, 0.15, 0.30, 0.20, 0.05],
        "street":   [0.30, 0.15, 0.15, 0.20, 0.10, 0.10],
        "default":  [1/6,  1/6,  1/6,  1/6,  1/6,  1/6]
    }
    
    prior = CONTEXT_PRIORS.get(context, CONTEXT_PRIORS["default"])
    
    # Extract non-background probabilities (first 6 classes)
    cv_probs = model_probs[:6]
    bg_prob = model_probs[6]
    
    # Bayesian multiplication: P(Class | Image, Context) ~ P(Image | Class) * P(Class | Context)
    fused_unnormalized = cv_probs * np.array(prior)
    sum_fused = np.sum(fused_unnormalized)
    
    if sum_fused > 0:
        fused_probs = fused_unnormalized / sum_fused
    else:
        fused_probs = cv_probs
        
    # Scale fused probabilities to sum to (1.0 - bg_prob) to maintain overall confidence balance
    fused_probs_scaled = fused_probs * (1.0 - bg_prob)
    
    # Return 7-class vector
    return np.append(fused_probs_scaled, bg_prob)

def auto_detect_context(img):
    """
    Automatically estimates the environmental context of the image using HSV color heuristics.
    Returns: 'beach', 'grass', or 'default'
    """
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    total_pixels = img.shape[0] * img.shape[1]
    
    # Grass detection: Hue in 35-85 (Greenish), Saturation > 40, Value > 40
    lower_green = np.array([35, 40, 40])
    upper_green = np.array([85, 255, 255])
    green_mask = cv2.inRange(hsv, lower_green, upper_green)
    green_pct = np.sum(green_mask > 0) / total_pixels
    
    # Beach Sand detection: Hue in 10-28 (Yellowish/Brownish Sand), Saturation in 20-150, Value > 80
    lower_sand = np.array([10, 20, 80])
    upper_sand = np.array([28, 150, 255])
    sand_mask = cv2.inRange(hsv, lower_sand, upper_sand)
    sand_pct = np.sum(sand_mask > 0) / total_pixels
    
    # Beach Water/Ocean detection: Hue in 90-130 (Blueish water), Saturation > 30, Value > 50
    lower_water = np.array([90, 30, 50])
    upper_water = np.array([130, 255, 255])
    water_mask = cv2.inRange(hsv, lower_water, upper_water)
    water_pct = np.sum(water_mask > 0) / total_pixels
    
    beach_pct = sand_pct + water_pct
    
    print(f"[CONTEXT DETECTOR] Color Analysis: Green={green_pct*100:.1f}%, Sand/Water={beach_pct*100:.1f}%")
    
    if green_pct >= 0.25:
        return "grass"
    elif beach_pct >= 0.20:
        return "beach"
    else:
        return "default"

def main():
    parser = argparse.ArgumentParser(description="FYP Waste Management: 2-Stage YOLOv11 + EfficientNetB0 Hierarchical Pipeline")
    parser.add_argument("--image", type=str, help="Path to a complex test image.")
    parser.add_argument("--random", action="store_true", help="Auto-pick a random complex image from the official TACO dataset.")
    parser.add_argument("--yolo-weights", type=str, default=str(DEFAULT_YOLO_WEIGHTS), help="YOLOv11 weights path.")
    parser.add_argument("--cnn-weights", type=str, default=str(DEFAULT_CNN_WEIGHTS), help="EfficientNetB0 CNN model weights path (.tflite or .h5).")
    parser.add_argument("--yolo-conf", type=float, default=0.20, help="YOLOv11 detection confidence threshold (Stage 2 will verify and filter out background).")
    parser.add_argument("--cnn-conf", type=float, default=0.40, help="CNN verification confidence threshold.")
    parser.add_argument("--out-dir", type=str, default=str(DEFAULT_OUT_DIR), help="Output folder to write visual detection sheets.")
    parser.add_argument("--padding", type=int, default=8, help="Bounding box crop padding to preserve complete geometry details.")
    parser.add_argument("--clahe", action="store_true", help="Apply Contrast-Limited Adaptive Histogram Equalization to handle harsh shadows/glare.")
    parser.add_argument("--class-specific", action="store_true", help="Enable class-specific confidence thresholds to suppress reflective false positives.")
    parser.add_argument("--context", type=str, default="auto", choices=["auto", "default", "beach", "grass", "indoor", "street"], help="Multi-modal context metadata to apply Bayesian fusion. 'auto' automatically estimates context.")
    
    args = parser.parse_args()
    
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    print("=====================================================================")
    print("      FYP 2-Stage YOLOv11 + EfficientNetB0 Hierarchical Pipeline       ")
    print("=====================================================================")
    
    # 1. Resolve and validate model weights
    yolo_weights_path = Path(args.yolo_weights)
    cnn_weights_path = Path(args.cnn_weights)
    
    if not yolo_weights_path.exists():
        print(f"[WARNING] Custom YOLOv11 weights not found at: {yolo_weights_path}")
        print("[INFO] Falling back to official 'yolo11n.pt' for structural verification...")
        yolo_weights_path = Path("yolo11n.pt")
        
    if not cnn_weights_path.exists():
        # Fallback to .h5 if .tflite is not found
        fallback_h5 = cnn_weights_path.with_suffix(".h5")
        if fallback_h5.exists():
            print(f"[INFO] TFLite model not found at {cnn_weights_path}. Falling back to H5 model at {fallback_h5}...")
            cnn_weights_path = fallback_h5
        else:
            print(f"[ERROR] CNN weights not found at: {cnn_weights_path}")
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
        
    # Auto-detect context modality
    detected_context = args.context
    if args.context == "auto":
        print("[INFO] Automatically detecting environmental context modality...")
        detected_context = auto_detect_context(frame)
        print(f"[OK] Automatically matched context modality: {detected_context.upper()}")
        
    if args.clahe:
        print("[INFO] Applying CLAHE (Contrast-Limited Adaptive Histogram Equalization) enhancement...")
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        cl = clahe.apply(l)
        limg = cv2.merge((cl, a, b))
        frame_enhanced = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
        yolo_input_img = frame_enhanced
    else:
        yolo_input_img = frame
        
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
    
    # Stage 2: EfficientNetB0 CNN (Supports H5 and TFLite)
    is_tflite = cnn_weights_path.suffix == ".tflite"
    if is_tflite:
        interpreter = tf.lite.Interpreter(model_path=str(cnn_weights_path))
        interpreter.allocate_tensors()
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        print(f"  - Stage 2: Quantized TFLite model loaded from {cnn_weights_path}")
    else:
        cnn_model = keras.models.load_model(str(cnn_weights_path))
        print(f"  - Stage 2: EfficientNetB0 CNN model loaded from {cnn_weights_path}")
    
    print(f"[OK] Models loaded successfully in {time.time() - t_start_load:.2f} seconds.")
    
    # 5. Stage 1: Run YOLOv11 Object Detection
    print("\n[INFO] Running Stage 1: YOLOv11 object proposal & rough classification...")
    t_start_yolo = time.time()
    yolo_results = yolo_model.predict(yolo_input_img, conf=args.yolo_conf, verbose=False)
    t_yolo_elapsed = time.time() - t_start_yolo
    
    boxes = yolo_results[0].boxes
    print(f"[OK] YOLOv11 proposed {len(boxes)} candidate boxes in {t_yolo_elapsed*1000:.1f}ms.")
    
    # --- 🧠 ADAPTIVE SCENE ENGINE (Plug & Play Optimization) ---
    adaptive_cnn_conf = args.cnn_conf
    is_adaptive_triggered = False
    adaptive_mode = "STANDARD"
    
    # Scenario 1: Extreme Small Object / Low Recall recovery
    if len(boxes) == 0 and args.yolo_conf > 0.05:
        print(f"[ADAPTIVE ENGINE] No waste objects detected at threshold {args.yolo_conf:.2f}.")
        print("[ADAPTIVE ENGINE] Retrying with High-Sensitivity Small Object Sweep (yolo_conf=0.05)...")
        yolo_results = yolo_model.predict(yolo_input_img, conf=0.05, verbose=False)
        boxes = yolo_results[0].boxes
        if len(boxes) > 0:
            print(f"[ADAPTIVE ENGINE] Successfully recovered {len(boxes)} candidates with sensitivity sweep!")
            adaptive_cnn_conf = 0.15  # Lower threshold for low-clarity small objects
            adaptive_mode = "SMALL_OBJECT_RECOVERY"
            is_adaptive_triggered = True
            
    # Scenario 2: Dense Clutter & Overlap Mitigation
    elif len(boxes) >= 8:
        print(f"[ADAPTIVE ENGINE] Dense waste pile detected ({len(boxes)} proposed objects).")
        print("[ADAPTIVE ENGINE] Automatically lowering CNN verification threshold to prevent occlusion/overlap rejection...")
        adaptive_cnn_conf = max(0.18, args.cnn_conf * 0.5)  # E.g. 0.40 -> 0.20
        adaptive_mode = "DENSE_CLUTTER_MITIGATION"
        is_adaptive_triggered = True
        
    if is_adaptive_triggered:
        print(f"[ADAPTIVE ENGINE] Mode: {adaptive_mode} | Dynamic Verification Threshold: {adaptive_cnn_conf:.2f}")
    
    # Stage 2: Verification using Keras EfficientNetB0
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
        w_box = x2 - x1
        h_box = y2 - y1
        
        # Physical size filter: reject tiny blurry crops (width or height < 24 px)
        if w_box < 24 or h_box < 24:
            yolo_cls_id = int(box.cls[0])
            yolo_class = yolo_model.names[yolo_cls_id] if hasattr(yolo_model, 'names') else YOLO_CLASSES[yolo_cls_id]
            yolo_conf = float(box.conf[0])
            yolo_str = f"{yolo_class.upper()} ({yolo_conf*100:.1f}%)"
            print(f"Box #{box_id:<5} | {yolo_str:<22} | {'TINY CROP':<28} | REJECTED (Tiny Texture Noise)")
            continue
        
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
        
        # Preprocess and feed crop to EfficientNetB0
        crop_input = preprocess_crop(crop)
        if is_tflite:
            interpreter.set_tensor(input_details[0]['index'], crop_input)
            interpreter.invoke()
            cnn_probs = interpreter.get_tensor(output_details[0]['index'])[0]
        else:
            cnn_probs = cnn_model.predict(crop_input, verbose=0)[0]
        cnn_pred_idx = np.argmax(cnn_probs)
        cnn_class = CNN_CLASSES[cnn_pred_idx]
        cnn_conf = float(cnn_probs[cnn_pred_idx])
        # --- Stage 2 Dynamic Alpha Ensemble Fusion & Background Gatekeeper ---
        # 1. First check if the CNN strongly predicts Background
        if cnn_class == "Background" and cnn_probs[6] >= 0.65:
            combined_class = "Background"
            combined_conf = float(cnn_probs[6])
        else:
            # 2. Blend YOLO and CNN probabilities dynamically based on YOLO's confidence
            yolo_probs = np.zeros(7)
            if yolo_class in CNN_CLASSES:
                yidx = CNN_CLASSES.index(yolo_class)
                yolo_probs[yidx] = yolo_conf
            
            # Dynamic Alpha:
            # - If YOLO is highly confident, we trust YOLO's global context (alpha = 0.70)
            # - If YOLO is moderately confident, we blend both equally (alpha = 0.40)
            # - If YOLO is uncertain, we let CNN's high-resolution texture classification lead (alpha = 0.15)
            if yolo_conf >= 0.65:
                alpha = 0.70
            elif yolo_conf >= 0.30:
                alpha = 0.40
            else:
                alpha = 0.15
                
            combined_probs = alpha * yolo_probs + (1.0 - alpha) * cnn_probs
            
            # Apply Multi-Modal Bayesian Context Fusion
            if detected_context != "default":
                combined_probs = apply_bayesian_fusion(combined_probs, detected_context)
                
            combined_pred_idx = np.argmax(combined_probs)
            
            if combined_pred_idx == 6:
                combined_class = "Background"
                combined_conf = float(combined_probs[6])
            else:
                combined_class = CNN_CLASSES[combined_pred_idx]
                combined_conf = float(combined_probs[combined_pred_idx])
            
        yolo_str = f"{yolo_class.upper()} ({yolo_conf*100:.1f}%)"
        cnn_str = f"{cnn_class.upper()} ({cnn_conf*100:.1f}%)"
        
        # Class-specific thresholds dictionary
        CLASS_THRESHOLDS = {
            "plastic": 0.25,     # reflective
            "glass": 0.30,       # highly reflective
            "metal": 0.18,       # distinct
            "paper": 0.20,
            "cardboard": 0.18,
            "organic": 0.22
        }

        # Dynamic verification threshold
        if args.class_specific and combined_class in CLASS_THRESHOLDS:
            current_threshold = CLASS_THRESHOLDS[combined_class]
        else:
            current_threshold = 0.20 if combined_class != "Background" else adaptive_cnn_conf
        
        if combined_class == "Background":
            decision = "REJECTED (Ghost Waste Background)"
        elif combined_conf < current_threshold:
            decision = f"REJECTED (Conf {combined_conf*100:.1f}% < {current_threshold*100:.0f}%)"
        else:
            decision = "ACCEPTED & VERIFIED ✅"
            valid_detections += 1
            
            # Draw premium bounding box
            color = CLASS_COLORS.get(combined_class, (0, 255, 0))
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 3)
            cv2.circle(annotated_frame, (x1, y1), 5, color, cv2.FILLED)
            
            # Draw text label card
            label_text = f"{combined_class.upper()} (Verified: {combined_conf*100:.1f}%)"
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
