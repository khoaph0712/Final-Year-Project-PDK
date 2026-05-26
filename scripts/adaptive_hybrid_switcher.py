#!/usr/bin/env python
import sys
import os
import time
import argparse
import random
import pickle
from pathlib import Path
import numpy as np
import cv2

# Quiet TensorFlow warnings
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms

# Include scripts directory to import local files
SCRIPTS_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPTS_DIR.parent
sys.path.append(str(SCRIPTS_DIR))
sys.path.append(str(SCRIPTS_DIR / "archive"))

from custom_feature_extractor import extract_637_features

# =====================================================================
# Configuration Paths
# =====================================================================
YOLO_WEIGHTS = ROOT_DIR / "runs" / "detect" / "yolov11_super_dataset" / "weights" / "best.pt"
CNN_EFFICIENTNET_WEIGHTS = ROOT_DIR / "runs" / "dl" / "cnn_efficientnet_tuned" / "best_efficientnet_tuned.h5"
CONVNEXT_ENSEMBLE_WEIGHTS = ROOT_DIR / "runs" / "dl" / "convnext_ensemble" / "best_convnext_ensemble.pth"
CONVNEXT_ENSEMBLE_TUNED_WEIGHTS = ROOT_DIR / "runs" / "dl" / "convnext_ensemble_tuned" / "best_convnext_ensemble_tuned.pth"
HANDCRAFTED_SCALER = ROOT_DIR / "runs" / "dl" / "convnext_ensemble" / "handcrafted_scaler.pkl"
OUT_DIR = ROOT_DIR / "runs" / "dl" / "adaptive_switcher_results"

TARGET_CLASSES = ["plastic", "glass", "metal", "paper", "cardboard", "organic", "Background"]

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
# Model Architecture Definitions
# =====================================================================
class ConvNeXtFeatureExtractor(nn.Module):
    def __init__(self):
        super(ConvNeXtFeatureExtractor, self).__init__()
        self.backbone = models.convnext_tiny(weights=models.ConvNeXt_Tiny_Weights.DEFAULT)
        self.backbone.classifier = nn.Identity()
        for param in self.backbone.parameters():
            param.requires_grad = False
            
    def forward(self, x):
        with torch.no_grad():
            features = self.backbone(x)
        return torch.flatten(features, 1)

class Stage3EnsembleClassifier(nn.Module):
    def __init__(self, num_classes=7, dropout_rate=0.3):
        super(Stage3EnsembleClassifier, self).__init__()
        self.convnext_extractor = ConvNeXtFeatureExtractor()
        self.input_dim = 768 + 637
        
        self.classifier = nn.Sequential(
            nn.Linear(self.input_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate),
            
            nn.Linear(256, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            
            nn.Linear(64, num_classes)
        )
        
    def forward(self, image_tensor, handcrafted_features_tensor):
        deep_features = self.convnext_extractor(image_tensor)
        fused_features = torch.cat((deep_features, handcrafted_features_tensor), dim=1)
        logits = self.classifier(fused_features)
        return logits

# =====================================================================
# Stage 1: OpenCV Contour Localization (4-Stage)
# =====================================================================
def detect_candidate_blobs(image, min_area=400, max_area=None, padding=12):
    h_img, w_img = image.shape[:2]
    target_w = 1024
    scale = target_w / w_img if w_img > target_w else 1.0
    proc_img = cv2.resize(image, (target_w, int(h_img * scale)), interpolation=cv2.INTER_AREA) if scale != 1.0 else image.copy()
    h_proc, w_proc = proc_img.shape[:2]
    
    min_area_proc = min_area * (scale * scale)
    max_area_proc = (h_proc * w_proc) * 0.25 if max_area is None else max_area * (scale * scale)
    min_box_dim = max(8, int(12 * scale))

    gray = cv2.cvtColor(proc_img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    
    _, thresh_otsu = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    edges = cv2.Canny(blur, 30, 100)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    dilated_edges = cv2.dilate(edges, kernel, iterations=1)
    
    combined_proc = cv2.bitwise_or(thresh_otsu, dilated_edges)
    combined_proc = cv2.morphologyEx(combined_proc, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(combined_proc, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    proposals = []
    for idx, cnt in enumerate(contours):
        area = cv2.contourArea(cnt)
        if min_area_proc < area < max_area_proc:
            x, y, w, h = cv2.boundingRect(cnt)
            if w > min_box_dim and h > min_box_dim:
                x_orig = int(x / scale)
                y_orig = int(y / scale)
                w_orig = int(w / scale)
                h_orig = int(h / scale)
                
                x1 = max(0, x_orig - padding)
                y1 = max(0, y_orig - padding)
                x2 = min(w_img, x_orig + w_orig + padding)
                y2 = min(h_img, y_orig + h_orig + padding)
                
                crop = image[y1:y2, x1:x2]
                if crop.size > 0:
                    proposals.append({
                        "id": idx + 1,
                        "bbox": (x1, y1, x2 - x1, y2 - y1),
                        "crop": crop
                    })
    return proposals

# =====================================================================
# Adaptive Switcher Complexity Analysis
# =====================================================================
def analyze_frame_complexity(image, laplacian_thresh=180.0, green_thresh=0.20):
    """
    Analyzes Laplacian Variance (Edge Complexity) and HSV Color space 
    to dynamically route the frame to the optimal pipeline.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Calculate Laplacian Variance (measures sharpness and high-frequency edge density)
    lap_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    
    # Calculate green pixel density (measures natural outdoor backgrounds)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower_green = np.array([35, 40, 40])
    upper_green = np.array([85, 255, 255])
    green_mask = cv2.inRange(hsv, lower_green, upper_green)
    green_pct = np.sum(green_mask > 0) / (image.shape[0] * image.shape[1])
    
    print("\n[HYBRID SWITCHER ENGINE] Complexity Diagnosis:")
    print(f"  * Laplacian Variance (Edge Clutter): {lap_var:.2f}")
    print(f"  * Green Background Density: {green_pct*100:.2f}%")
    
    # ROUTING RULE:
    # High Laplacian Variance indicates high edge clutter (like grass/leaves/branches) 
    # High Green background confirms an outdoor natural scene.
    # Route to 2-Stage YOLO if high clutter or high green percentage.
    if lap_var > laplacian_thresh or green_pct > green_thresh:
        print("[ROUTE MATCHED] -> OUTDOOR / HIGH COMPLEXITY -> Routing to 2-Stage YOLOv11 Pipeline")
        return "2-STAGE"
    else:
        print("[ROUTE MATCHED] -> INDOOR / STRUCTURED -> Routing to 4-Stage OpenCV-ConvNeXt Pipeline")
        return "4-STAGE"

# =====================================================================
# Fused Inference Executor (4-Stage Pipeline)
# =====================================================================
def run_4stage_pipeline(image, device, model, scaler, confidence=0.35):
    print("\n--- Running 4-Stage OpenCV + ConvNeXt + Handcrafted ANN Pipeline ---")
    start_time = time.time()
    
    # Stage 1: Localization
    proposals = detect_candidate_blobs(image)
    loc_time = time.time() - start_time
    print(f"[OK] Localization complete in {loc_time*1000:.1f}ms. Proposed {len(proposals)} crops.")
    
    vis_img = image.copy()
    preprocess = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    detections = 0
    for prop in proposals:
        crop = prop["crop"]
        x, y, w, h = prop["bbox"]
        
        # Stage 2: Preprocess Crop
        crop_resized = cv2.resize(crop, (224, 224), interpolation=cv2.INTER_CUBIC)
        crop_rgb = cv2.cvtColor(crop_resized, cv2.COLOR_BGR2RGB)
        
        # Stage 3: Extract Features & Classify
        feats = extract_637_features(crop)
        feats_scaled = scaler.transform(np.array([feats], dtype=np.float32))
        
        from PIL import Image
        img_pil = Image.fromarray(crop_rgb)
        img_tensor = preprocess(img_pil).unsqueeze(0).to(device)
        feats_tensor = torch.tensor(feats_scaled, dtype=torch.float32).to(device)
        
        with torch.no_grad():
            logits = model(img_tensor, feats_tensor)
            probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
            
        cls_idx = np.argmax(probs)
        conf = probs[cls_idx]
        cls_name = TARGET_CLASSES[cls_idx]
        
        if cls_name != "Background" and conf >= confidence:
            detections += 1
            print(f"  * [4-STAGE SUCCESS] Found object: {cls_name.upper()} ({conf*100:.1f}%)")
            color = CLASS_COLORS.get(cls_name, (255, 255, 255))
            cv2.rectangle(vis_img, (x, y), (x+w, y+h), color, 3)
            tag = f"[4S] {cls_name} {conf*100:.0f}%"
            cv2.putText(vis_img, tag, (x + 5, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)
            
    total_time = time.time() - start_time
    return vis_img, detections, total_time

# =====================================================================
# YOLO Inference Executor (2-Stage Pipeline)
# =====================================================================
def run_2stage_pipeline(image, cnn_model, confidence=0.20):
    print("\n--- Running 2-Stage YOLOv11 + EfficientNet Pipeline ---")
    start_time = time.time()
    
    # Stage 1: YOLO Object Detection
    from ultralytics import YOLO
    yolo = YOLO(str(YOLO_WEIGHTS))
    yolo_results = yolo.predict(image, conf=confidence, verbose=False)
    
    loc_time = time.time() - start_time
    print(f"[OK] YOLOv11 Localization complete in {loc_time*1000:.1f}ms.")
    
    vis_img = image.copy()
    detections = 0
    
    for result in yolo_results:
        boxes = result.boxes
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            w, h = x2 - x1, y2 - y1
            yolo_cls_idx = int(box.cls[0])
            yolo_conf = float(box.conf[0])
            
            # Crop proposal (Stage 2 Verification)
            crop = image[max(0, y1-8):min(image.shape[0], y2+8), max(0, x1-8):min(image.shape[1], x2+8)]
            if crop.size == 0:
                continue
                
            crop_resized = cv2.resize(crop, (224, 224), interpolation=cv2.INTER_LINEAR)
            crop_preprocessed = np.expand_dims(crop_resized / 255.0, axis=0) # Simple normalization
            
            # CNN verification
            cnn_probs = cnn_model.predict(crop_preprocessed, verbose=False)[0]
            cnn_cls_idx = np.argmax(cnn_probs)
            cnn_conf = cnn_probs[cnn_cls_idx]
            cnn_cls_name = TARGET_CLASSES[cnn_cls_idx]
            
            if cnn_cls_name != "Background" and cnn_conf >= confidence:
                detections += 1
                print(f"  * [2-STAGE SUCCESS] Verified object: {cnn_cls_name.upper()} ({cnn_conf*100:.1f}%)")
                color = CLASS_COLORS.get(cnn_cls_name, (255, 255, 255))
                cv2.rectangle(vis_img, (x1, y1), (x2, y2), color, 3)
                tag = f"[2S] {cnn_cls_name} {cnn_conf*100:.0f}%"
                cv2.putText(vis_img, tag, (x1 + 5, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)
                
    total_time = time.time() - start_time
    return vis_img, detections, total_time

# =====================================================================
# Main Orchestrator Flow
# =====================================================================
def main():
    parser = argparse.ArgumentParser(description="Adaptive Hybrid Pipeline Switcher Module")
    parser.add_argument("--image", type=str, required=True, help="Path to complex test image.")
    args = parser.parse_args()
    
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    img_path = Path(args.image)
    
    # 1. Load image
    image = cv2.imread(str(img_path))
    if image is None:
        print(f"[ERROR] Could not load image at {img_path}")
        return
        
    print("====================================================")
    print("Executing Adaptive Hybrid Pipeline Switcher Module")
    print("====================================================")
    
    # 2. Dynamic environment complexity routing
    route = analyze_frame_complexity(image)
    
    # 3. Route to optimal pipeline
    if route == "4-STAGE":
        # Load Stage 3 PyTorch ConvNeXt Ensemble (prefer progressive-tuned weights)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = Stage3EnsembleClassifier(num_classes=7)
        
        weights_to_load = CONVNEXT_ENSEMBLE_TUNED_WEIGHTS if CONVNEXT_ENSEMBLE_TUNED_WEIGHTS.exists() else CONVNEXT_ENSEMBLE_WEIGHTS
        print(f"[INFO] Loading Stage 3 Ensemble weights from: {weights_to_load}")
        model.load_state_dict(torch.load(weights_to_load, map_location=device))
        model.to(device)
        model.eval()
        
        with open(HANDCRAFTED_SCALER, "rb") as f:
            scaler = pickle.load(f)
            
        # Execute
        vis_img, detections, latency = run_4stage_pipeline(image, device, model, scaler)
        
    else: # 2-STAGE
        # Load Keras CNN model robustly using tf_keras
        try:
            import tf_keras as keras_fallback
        except ImportError:
            from tensorflow import keras as keras_fallback
        cnn_model = keras_fallback.models.load_model(str(CNN_EFFICIENTNET_WEIGHTS))
        
        # Execute
        vis_img, detections, latency = run_2stage_pipeline(image, cnn_model)
        
    # Save visual results
    save_path = OUT_DIR / f"switcher_{route}_{img_path.name}"
    cv2.imwrite(str(save_path), vis_img)
    
    print("\n====================================================")
    print(f"[SUCCESS] SWITCHER DECISION SUMMARY:")
    print(f"  - Selected Route: {route}")
    print(f"  - Verified Detections: {detections} item(s)")
    print(f"  - End-to-End Latency: {latency*1000:.1f}ms")
    print(f"  - Visual Report Saved to: {save_path}")
    print("====================================================")

if __name__ == "__main__":
    main()
