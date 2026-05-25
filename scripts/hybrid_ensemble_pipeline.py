#!/usr/bin/env python
"""
Final Year Project (FYP) - Waste Sorting & Classification System
Hybrid Localization-Classification Pipeline (OpenCV Blob Proposal + CNN & ANN Soft Voting Ensemble)

This script:
1. Localizes potential waste candidate regions inside a complex scene (e.g. from the TACO dataset) using OpenCV contours and edge descriptors.
2. Crops each candidate region and applies a dual-model soft voting ensemble:
   - PyTorch ANN (Multi-Layer Perceptron) running on 637 handcrafted features (Color, Texture, Shape, HOG) extracted via custom_feature_extractor.py.
   - Keras MobileNetV2 CNN running on raw image crops normalized to [-1.0, 1.0].
3. Combines both probabilities using soft voting (weighted average) to obtain robust class labels.
4. Suppresses Background classes and low-confidence predictions to filter out false positives.
5. Saves annotated visualization with premium overlays (bounding boxes, class labels, confidence scores) and prints detailed diagnostic reports.
"""

import sys
import os
import argparse
import random
import pickle
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

import torch
import torch.nn as nn

# Include current directory in system paths to import local custom files
SCRIPTS_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPTS_DIR.parent
sys.path.append(str(SCRIPTS_DIR))

from custom_feature_extractor import extract_637_features

# =====================================================================
# Configuration Paths
# =====================================================================
ANN_DIR = ROOT_DIR / "runs" / "dl" / "ann_637"
CNN_DIR = ROOT_DIR / "runs" / "dl" / "cnn_mobilenet"
TACO_DIR = ROOT_DIR / "external_datasets" / "super_yolo_dataset" / "test" / "images"
DEFAULT_OUT_DIR = ROOT_DIR / "runs" / "dl" / "hybrid_pipeline_results"

TARGET_CLASSES = ["plastic", "glass", "metal", "paper", "cardboard", "organic", "Background"]

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

# =====================================================================
# Model Definition (ANN MLP)
# =====================================================================
class WasteMLP(nn.Module):
    def __init__(self, input_dim=637, num_classes=7):
        super(WasteMLP, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3),
            
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            nn.Linear(128, num_classes)
        )
        
    def forward(self, x):
        return self.net(x)

# =====================================================================
# Image Blob Proposal via OpenCV
# =====================================================================
def detect_candidate_blobs(image, min_area=500, max_area=None, padding=8):
    """
    Proposes candidate bounding boxes using grayscale binarization (Otsu's Thresholding)
    and Canny Edge Dilation, combined using morphological close operations.
    Supports resolution-independent scaling to handle high-resolution images (e.g. 4K TACO).
    """
    h_img, w_img = image.shape[:2]
    
    # Scale image to a standard width of 1024 for consistent contour detection
    target_w = 1024
    scale = 1.0
    if w_img > target_w:
        scale = target_w / w_img
        target_h = int(h_img * scale)
        proc_img = cv2.resize(image, (target_w, target_h), interpolation=cv2.INTER_AREA)
    else:
        proc_img = image.copy()
        
    h_proc, w_proc = proc_img.shape[:2]
    
    # Scale area and dimension constraints to the processed resolution to maintain scale-invariance
    min_area_proc = min_area * (scale * scale)
    if max_area is None:
        max_area_proc = (h_proc * w_proc) * 0.20  # Strict area limit: maximum 20% of image area
    else:
        max_area_proc = max_area * (scale * scale)
        
    # Scale minimum box dimensions to processed resolution (minimum 8 pixels)
    min_box_dim = max(8, int(16 * scale))

    # Convert to grayscale and apply Gaussian blur to eliminate high frequency sensor noise
    gray = cv2.cvtColor(proc_img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # 1. Otsu Thresholding (Isolates high-contrast solid foreground objects)
    _, thresh_otsu = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # 2. Canny edge boundaries
    edges = cv2.Canny(blur, 35, 120)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    dilated_edges = cv2.dilate(edges, kernel, iterations=1)
    
    # Combine Otsu binary mask and dilated edge boundaries
    combined_proc = cv2.bitwise_or(thresh_otsu, dilated_edges)
    combined_proc = cv2.morphologyEx(combined_proc, cv2.MORPH_CLOSE, kernel)
    
    # Extract contour groups
    contours, _ = cv2.findContours(combined_proc, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Strict bounding box dimensions bounds
    max_box_w = w_proc * 0.50  # Max 50% of image width
    max_box_h = h_proc * 0.50  # Max 50% of image height
    
    proposals = []
    for idx, cnt in enumerate(contours):
        area = cv2.contourArea(cnt)
        if min_area_proc < area < max_area_proc:
            x, y, w, h = cv2.boundingRect(cnt)
            # Remove degenerate structures and giant background regions
            if w > min_box_dim and h > min_box_dim and w < max_box_w and h < max_box_h:
                # Scale bounding box coordinates back to original image resolution
                x_orig = int(x / scale)
                y_orig = int(y / scale)
                w_orig = int(w / scale)
                h_orig = int(h / scale)
                
                # Add safe padding on original resolution
                x1 = max(0, x_orig - padding)
                y1 = max(0, y_orig - padding)
                x2 = min(w_img, x_orig + w_orig + padding)
                y2 = min(h_img, y_orig + h_orig + padding)
                
                crop = image[y1:y2, x1:x2]
                if crop.size > 0:
                    proposals.append({
                        "id": idx + 1,
                        "bbox": (x1, y1, x2 - x1, y2 - y1),
                        "crop": crop,
                        "area": area / (scale * scale) # normalized area
                    })
                    
    # Generate a matching size debug mask for visualization
    if scale != 1.0:
        combined = cv2.resize(combined_proc, (w_img, h_img), interpolation=cv2.INTER_NEAREST)
    else:
        combined = combined_proc
        
    return proposals, combined

# =====================================================================
# Main Executable Flow
# =====================================================================
def main():
    parser = argparse.ArgumentParser(description="FYP Waste Management: OpenCV + Deep Ensemble Hybrid Pipeline")
    parser.add_argument("--image", type=str, help="Absolute or relative path to a complex test image.")
    parser.add_argument("--random", action="store_true", help="Auto-pick a random complex image from the official TACO dataset.")
    parser.add_argument("--confidence", type=float, default=0.40, help="Confidence threshold to filter false positive detections.")
    parser.add_argument("--cnn-weight", type=float, default=0.50, help="Weighted soft voting weight for CNN model [0..1].")
    parser.add_argument("--ann-weight", type=float, default=0.50, help="Weighted soft voting weight for ANN model [0..1].")
    parser.add_argument("--out-dir", type=str, default=str(DEFAULT_OUT_DIR), help="Output folder to write visual detection sheets.")
    parser.add_argument("--min-area", type=int, default=500, help="Min pixel area contour threshold to filter out tiny background artifacts.")
    parser.add_argument("--padding", type=int, default=8, help="Bounding box crop padding to preserve complete geometry details.")
    
    args = parser.parse_args()
    
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    print("=====================================================================")
    print("      FYP Hybrid Waste Localization & Ensemble Classifier Script      ")
    print("=====================================================================")
    
    # 1. Resolve and validate models
    ann_weights_path = ANN_DIR / "best_ann.pt"
    ann_scaler_path = ANN_DIR / "scaler_ann.pkl"
    cnn_model_path = CNN_DIR / "best_mobilenet.h5"
    
    if not ann_weights_path.exists():
        print(f"[ERROR] PyTorch ANN MLP weights not found at: {ann_weights_path}")
        sys.exit(1)
    if not ann_scaler_path.exists():
        print(f"[ERROR] ANN Scaler object not found at: {ann_scaler_path}")
        sys.exit(1)
    if not cnn_model_path.exists():
        print(f"[ERROR] Keras CNN Model weights not found at: {cnn_model_path}")
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
        # Fallback to check default paths
        fallback_dir = ROOT_DIR / "data" / "merged_dataset_v5" / "test" / "images"
        if fallback_dir.exists():
            test_images = list(fallback_dir.glob("*.jpg")) + list(fallback_dir.glob("*.png"))
            if test_images:
                selected_image_path = random.choice(test_images)
                print(f"[WARN] No image specified. Fallback to random test image: {selected_image_path.name}")
                
    if selected_image_path is None or not selected_image_path.exists():
        print(f"[ERROR] Input image path does not exist or is invalid! Path: {selected_image_path}")
        print("Please provide --image <path> or use --random to pull from the TACO dataset.")
        sys.exit(1)
        
    # 3. Load Image and prepare copy
    frame = cv2.imread(str(selected_image_path))
    if frame is None:
        print(f"[ERROR] Failed to decode image file: {selected_image_path}")
        sys.exit(1)
        
    annotated_frame = frame.copy()
    h_img, w_img = frame.shape[:2]
    print(f"[INFO] Image Dimension: {w_img}x{h_img} pixels.")
    
    # 4. Load Models into RAM
    print("\n[INFO] Initializing Models & Scalers...")
    t_start_load = time.time()
    
    # A. PyTorch ANN Model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ann_model = WasteMLP(input_dim=637, num_classes=len(TARGET_CLASSES)).to(device)
    ann_model.load_state_dict(torch.load(ann_weights_path, map_location=device))
    ann_model.eval()
    
    with open(ann_scaler_path, "rb") as f:
        ann_scaler = pickle.load(f)
    print(f"  - PyTorch ANN MLP loaded successfully on: {device}")
    
    # B. Keras CNN Model
    cnn_model = keras.models.load_model(str(cnn_model_path))
    print("  - Keras CNN MobileNetV2 loaded successfully.")
    
    print(f"[OK] Models loaded in {time.time() - t_start_load:.2f} seconds.")
    
    # 5. Extract OpenCV Bounding Box proposals
    print("\n[INFO] Running OpenCV localization module (Greyscale + Gaussian + Otsu + Canny)...")
    t_start_loc = time.time()
    proposals, debug_mask = detect_candidate_blobs(
        frame, 
        min_area=args.min_area, 
        padding=args.padding
    )
    t_loc_elapsed = time.time() - t_start_loc
    print(f"[OK] Found {len(proposals)} candidate objects in {t_loc_elapsed*1000:.1f}ms")
    
    # Validate weights normalization
    w_sum = args.cnn_weight + args.ann_weight
    w_cnn = args.cnn_weight / w_sum
    w_ann = args.ann_weight / w_sum
    if w_sum != 1.0:
        print(f"[WARN] Normalizing voting weights from {args.cnn_weight}:{args.ann_weight} to {w_cnn:.2f}:{w_ann:.2f}")

    # 6. Process candidate crops through the Ensemble Pipeline
    valid_detections = 0
    t_start_infer = time.time()
    
    print("\n" + "-"*80)
    print(f"{'Crop ID':<8} | {'Coordinates':<16} | {'ANN Match (%)':<18} | {'CNN Match (%)':<18} | {'Ensemble Vote (%)'}")
    print("-"*80)
    
    for prop in proposals:
        crop_id = prop["id"]
        x, y, w, h = prop["bbox"]
        crop = prop["crop"]
        
        # -------------------------------------------------------------
        # 1. Keras CNN MobileNetV2 Inference
        # -------------------------------------------------------------
        # Preprocess: resize -> RGB converter -> normalize [-1.0, 1.0]
        crop_cnn = cv2.resize(crop, (128, 128))
        crop_cnn_rgb = cv2.cvtColor(crop_cnn, cv2.COLOR_BGR2RGB)
        crop_cnn_norm = (crop_cnn_rgb.astype(np.float32) / 127.5) - 1.0
        crop_cnn_input = np.expand_dims(crop_cnn_norm, axis=0)
        
        # Keras predict
        cnn_probs = cnn_model.predict(crop_cnn_input, verbose=0)[0]
        cnn_pred_idx = np.argmax(cnn_probs)
        cnn_class = TARGET_CLASSES[cnn_pred_idx]
        cnn_conf = cnn_probs[cnn_pred_idx]
        
        # -------------------------------------------------------------
        # 2. PyTorch ANN MLP Inference
        # -------------------------------------------------------------
        # Preprocess: Handcrafted feature extraction -> Standard scale -> Torch tensor
        try:
            feats = extract_637_features(crop)
            feats_scaled = ann_scaler.transform([feats])
            feats_tensor = torch.tensor(feats_scaled, dtype=torch.float32).to(device)
            
            with torch.no_grad():
                logits = ann_model(feats_tensor)
                ann_probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
                
            ann_pred_idx = np.argmax(ann_probs)
            ann_class = TARGET_CLASSES[ann_pred_idx]
            ann_conf = ann_probs[ann_pred_idx]
        except Exception as ex:
            # Fallback in case feature extraction experiences exceptions on highly degenerate crops
            print(f"[WARN] ANN feature extraction failed on Crop {crop_id}: {ex}")
            ann_probs = np.zeros(len(TARGET_CLASSES))
            ann_probs[-1] = 1.0  # Default to Background
            ann_class = "Background"
            ann_conf = 1.0
            
        # -------------------------------------------------------------
        # 3. Soft Voting Ensemble Fusion
        # -------------------------------------------------------------
        ensemble_probs = w_cnn * cnn_probs + w_ann * ann_probs
        ensemble_pred_idx = np.argmax(ensemble_probs)
        ensemble_class = TARGET_CLASSES[ensemble_pred_idx]
        ensemble_conf = ensemble_probs[ensemble_pred_idx]
        
        # Print diagnostic report
        ann_str = f"{ann_class.upper()} ({ann_conf*100:.1f}%)"
        cnn_str = f"{cnn_class.upper()} ({cnn_conf*100:.1f}%)"
        ens_str = f"{ensemble_class.upper()} ({ensemble_conf*100:.1f}%)"
        print(f"Crop #{crop_id:<4} | ({x},{y},{w},{h}) | {ann_str:<18} | {cnn_str:<18} | {ens_str}")
        
        # -------------------------------------------------------------
        # 4. Filter Background and Low Confidence Predictions
        # -------------------------------------------------------------
        if ensemble_class != "Background" and ensemble_conf >= args.confidence:
            valid_detections += 1
            color = CLASS_COLORS.get(ensemble_class, (0, 255, 0))
            
            # Draw primary bounding box on frame
            cv2.rectangle(annotated_frame, (x, y), (x + w, y + h), color, 3)
            
            # Draw micro design dot at top-left corner
            cv2.circle(annotated_frame, (x, y), 5, color, cv2.FILLED)
            
            # Formulate premium label card
            label_text = f"{ensemble_class.upper()} {ensemble_conf*100:.1f}%"
            (text_w, text_h), baseline = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
            
            # Render visual label card container
            cv2.rectangle(
                annotated_frame, 
                (x, y - text_h - 10), 
                (x + text_w + 10, y), 
                color, 
                cv2.FILLED
            )
            # Render readable text inside the card container
            cv2.putText(
                annotated_frame, 
                label_text, 
                (x + 5, y - 5), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                0.45, 
                (255, 255, 255) if color != (241, 196, 15) else (0, 0, 0),  # Black text for yellow card, white otherwise
                1, 
                cv2.LINE_AA
            )
            
    t_infer_elapsed = time.time() - t_start_infer
    print("-"*80)
    print(f"[OK] Total Inference completed in {t_infer_elapsed:.2f}s (Average {(t_infer_elapsed/max(1, len(proposals)))*1000:.1f}ms per crop).")
    print(f"[OK] Labeled {valid_detections} valid waste objects above threshold >= {args.confidence*100:.0f}%.")
    
    # 7. Merge original annotated image and threshold mask side-by-side
    debug_mask_bgr = cv2.cvtColor(debug_mask, cv2.COLOR_GRAY2BGR)
    # Check if we should resize the debug mask to match original frame size
    if debug_mask_bgr.shape != frame.shape:
        debug_mask_bgr = cv2.resize(debug_mask_bgr, (w_img, h_img))
        
    combined_display = np.hstack((annotated_frame, debug_mask_bgr))
    
    # Write output to disk
    out_img_name = f"ensemble_det_{selected_image_path.stem}.png"
    out_img_path = out_dir / out_img_name
    cv2.imwrite(str(out_img_path), combined_display)
    
    print("\n=====================================================================")
    print(f"[SUCCESS] Visualized side-by-side panel saved to:")
    print(f"  --> {out_img_path}")
    print("=====================================================================")

if __name__ == "__main__":
    main()
