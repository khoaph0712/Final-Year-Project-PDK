#!/usr/bin/env python
import sys
import os
import argparse
import random
import pickle
import time
from pathlib import Path
import numpy as np
import cv2

import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms

# Include current directory in paths
SCRIPTS_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPTS_DIR.parent
sys.path.append(str(SCRIPTS_DIR))
sys.path.append(str(SCRIPTS_DIR / "archive"))

from custom_feature_extractor import extract_637_features

# =====================================================================
# Configuration Paths
# =====================================================================
ENSEMBLE_DIR = ROOT_DIR / "runs" / "dl" / "convnext_ensemble"
ENSEMBLE_TUNED_DIR = ROOT_DIR / "runs" / "dl" / "convnext_ensemble_tuned"
TACO_TEST_DIR = ROOT_DIR / "data" / "demo_images" / "beach" # Or demo images
DEFAULT_OUT_DIR = ROOT_DIR / "runs" / "dl" / "convnext_pipeline_results"

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
# OpenCV Contour Localization (Stage 1)
# =====================================================================
def detect_candidate_blobs(image, min_area=400, max_area=None, padding=12):
    """
    Stage 1: OpenCV Contour Localization.
    Uses Canny edge detection and Morphological closing to localize candidate blobs.
    """
    h_img, w_img = image.shape[:2]
    target_w = 1024
    scale = 1.0
    if w_img > target_w:
        scale = target_w / w_img
        target_h = int(h_img * scale)
        proc_img = cv2.resize(image, (target_w, target_h), interpolation=cv2.INTER_AREA)
    else:
        proc_img = image.copy()
        
    h_proc, w_proc = proc_img.shape[:2]
    min_area_proc = min_area * (scale * scale)
    max_area_proc = (h_proc * w_proc) * 0.25 if max_area is None else max_area * (scale * scale)
    min_box_dim = max(8, int(12 * scale))

    gray = cv2.cvtColor(proc_img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Otsu thresholding combined with Canny
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
                
                # Apply padding (Stage 2 preparation)
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
                        "area": area / (scale * scale)
                    })
                    
    return proposals, combined_proc

# =====================================================================
# Main 4-Stage Runner
# =====================================================================
def main():
    parser = argparse.ArgumentParser(description="4-Stage Stage 3 ConvNeXt Ensemble Waste Pipeline")
    parser.add_argument("--image", type=str, help="Path to complex test image.")
    parser.add_argument("--confidence", type=float, default=0.35, help="Confidence filter.")
    parser.add_argument("--out-dir", type=str, default=str(DEFAULT_OUT_DIR), help="Output folder.")
    args = parser.parse_args()

    args.out_dir = Path(args.out_dir)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load weights and scalers
    print("====================================================")
    print("Executing 4-Stage ConvNeXt-ANN Waste sorting Pipeline")
    print("====================================================")
    
    tuned_model_path = ENSEMBLE_TUNED_DIR / "best_convnext_ensemble_tuned.pth"
    model_path = tuned_model_path if tuned_model_path.exists() else (ENSEMBLE_DIR / "best_convnext_ensemble.pth")
    scaler_path = ENSEMBLE_DIR / "handcrafted_scaler.pkl"
    
    if not model_path.exists() or not scaler_path.exists():
        print(f"[ERROR] Model weights or scaler not found at: {model_path} or {scaler_path}. Please run training first.")
        return
        
    print(f"[INFO] Using weights file: {model_path}")

    # Load scaler
    with open(scaler_path, "rb") as f:
        scaler = pickle.load(f)

    # Load PyTorch Model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Stage3EnsembleClassifier(num_classes=7)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    print(f"[INFO] Fused model successfully loaded on: {device}")

    # Pick an image
    if args.image:
        img_path = Path(args.image)
    else:
        # Pick a random image from demo sets
        demo_images = list(ROOT_DIR.glob("data/demo_images/**/*.jpg")) + list(ROOT_DIR.glob("data/demo_images/**/*.png"))
        if len(demo_images) == 0:
            print("[ERROR] No demo images found under data/demo_images/. Please specify --image.")
            return
        img_path = random.choice(demo_images)
        
    print(f"[INFO] Loading target image: {img_path}")
    image = cv2.imread(str(img_path))
    if image is None:
        print("[ERROR] Image loaded is None. Verify path.")
        return
        
    # STAGE 1: OpenCV Contour Localization
    print("\n--- STAGE 1: LOCALIZATION GATING ---")
    start_time = time.time()
    proposals, _ = detect_candidate_blobs(image)
    print(f"[INFO] Proposed {len(proposals)} candidate objects in {float(time.time() - start_time)*1000:.1f}ms")

    # Image canvas for overlays
    vis_img = image.copy()
    
    # Preprocessing transform
    preprocess = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    detected_counts = {}
    
    # STAGE 2 & 3: Crop and Fused Inference
    print("\n--- STAGE 2 & 3: HYBRID FUSED INFERENCE ---")
    for prop in proposals:
        crop = prop["crop"]
        x, y, w, h = prop["bbox"]
        
        # Crop Preprocessing (Stage 2)
        crop_resized = cv2.resize(crop, (224, 224), interpolation=cv2.INTER_CUBIC)
        crop_rgb = cv2.cvtColor(crop_resized, cv2.COLOR_BGR2RGB)
        
        # Extract Handcrafted features
        feats = extract_637_features(crop)
        feats_scaled = scaler.transform(np.array([feats], dtype=np.float32))
        
        # Prepare tensors
        from PIL import Image
        img_pil = Image.fromarray(crop_rgb)
        img_tensor = preprocess(img_pil).unsqueeze(0).to(device)
        feats_tensor = torch.tensor(feats_scaled, dtype=torch.float32).to(device)
        
        # Run forward pass (Stage 3)
        with torch.no_grad():
            logits = model(img_tensor, feats_tensor)
            probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
            
        cls_idx = np.argmax(probs)
        conf = probs[cls_idx]
        cls_name = TARGET_CLASSES[cls_idx]
        
        # Gating threshold (Stage 4)
        if cls_name != "Background" and conf >= args.confidence:
            print(f"  * Detected Object ID {prop['id']}: [{cls_name.upper()}] with confidence: {conf*100:.1f}%")
            detected_counts[cls_name] = detected_counts.get(cls_name, 0) + 1
            
            # Premium annotation drawing
            color = CLASS_COLORS.get(cls_name, (255, 255, 255))
            
            # Bounding box
            cv2.rectangle(vis_img, (x, y), (x+w, y+h), color, 3)
            
            # Bounding Box Tag
            tag = f"{cls_name} {conf*100:.0f}%"
            (tw, th), baseline = cv2.getTextSize(tag, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(vis_img, (x, y - th - 8), (x + tw + 10, y), color, -1)
            cv2.putText(vis_img, tag, (x + 5, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)

    # Save visual result
    out_path = args.out_dir / f"pipeline_4stage_{img_path.name}"
    cv2.imwrite(str(out_path), vis_img)
    
    print("\n--- STAGE 4: DECISION ENGINE OVERVIEW ---")
    print(f"[SUCCESS] Visual report saved to: {out_path}")
    print("Summary of classifications:")
    for k, v in detected_counts.items():
        print(f"  - {k.upper()}: {v} item(s)")
    print("====================================================")

if __name__ == "__main__":
    main()
