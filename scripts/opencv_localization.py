import sys
import os
import random
from pathlib import Path
import numpy as np
import cv2
import tensorflow as tf

ROOT = Path(__file__).resolve().parent.parent
DATA_YAML = ROOT / "merged_dataset_v3" / "data.yaml"
TFLITE_PATH = ROOT / "runs" / "dl" / "cnn_mobilenet" / "best_mobilenet_quant.tflite"
OUT_DIR = ROOT / "runs" / "dl" / "cnn_mobilenet" / "detection_results"

def detect_trash_blobs(frame):
    """Propose candidate bounding boxes using OpenCV Canny Edges + Otsu Binarization."""
    # Convert to grayscale and apply mild Gaussian Blur
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # 1. Otsu thresholding to find high-contrast foreground objects
    _, thresh_otsu = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # 2. Canny Edge Detection to capture textured boundaries
    edges = cv2.Canny(blur, 30, 100)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    dilated_edges = cv2.dilate(edges, kernel, iterations=1)
    
    # Combine contrast thresholding and edge boundaries
    combined = cv2.bitwise_or(thresh_otsu, dilated_edges)
    combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel)
    
    # Find contours
    contours, _ = cv2.findContours(combined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    proposals = []
    h_img, w_img = frame.shape[:2]
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 450:  # Ignore tiny noisy specs
            x, y, w, h = cv2.boundingRect(cnt)
            # Filter out thin degenerate lines or tiny boxes
            if w > 12 and h > 12:
                # Add safe padding around bounding box
                pad = 4
                x1 = max(0, x - pad)
                y1 = max(0, y - pad)
                x2 = min(w_img, x + w + pad)
                y2 = min(h_img, y + h + pad)
                
                crop = frame[y1:y2, x1:x2]
                proposals.append((crop, (x1, y1, x2 - x1, y2 - y1)))
                
    return proposals, combined

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    
    print("====================================================")
    print("FYP Waste Hybrid Localization-Classification Pipeline")
    print("====================================================")
    
    target_classes = ["plastic", "glass", "metal", "paper", "cardboard", "organic", "Background"]
    
    # 1. Check TFLite model
    if not TFLITE_PATH.exists():
        raise FileNotFoundError(f"Quantized TFLite model not found at {TFLITE_PATH}! Run export_tflite.py first.")
        
    print(f"[INFO] Loading quantized TFLite classifier from {TFLITE_PATH}...")
    interpreter = tf.lite.Interpreter(model_path=str(TFLITE_PATH))
    interpreter.allocate_tensors()
    
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    # 2. Find real annotated test images from merged_dataset_v3
    img_dir = ROOT / "merged_dataset_v3" / "test" / "images"
    lbl_dir = ROOT / "merged_dataset_v3" / "test" / "labels"
    
    if not img_dir.exists():
        raise FileNotFoundError(f"Test split images directory not found at {img_dir}!")
        
    img_paths = list(img_dir.glob("*"))
    # Shuffle and find 5 images that contain active waste annotations (non-empty labels)
    random.seed(42)
    random.shuffle(img_paths)
    
    test_selected = []
    for p in img_paths:
        if p.suffix.lower() not in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}:
            continue
        lbl_p = lbl_dir / p.with_suffix(".txt").name
        if lbl_p.exists() and lbl_p.stat().st_size > 0:
            test_selected.append(p)
            if len(test_selected) >= 5:
                break
                
    if not test_selected:
        print("[WARN] No active annotated test images found, using random test images...")
        test_selected = [p for p in img_paths[:5] if p.suffix.lower() in {".jpg", ".jpeg", ".png"}]
        
    print(f"[INFO] Running hybrid pipeline on {len(test_selected)} real test images...")
    
    for idx, img_path in enumerate(test_selected):
        print(f"\nProcessing Image {idx+1}: {img_path.name}")
        frame = cv2.imread(str(img_path))
        if frame is None:
            continue
            
        annotated_frame = frame.copy()
        
        # Run CV Bounding Box proposals
        proposals, debug_mask = detect_trash_blobs(frame)
        print(f"  - OpenCV proposed {len(proposals)} candidate regions")
        
        valid_detections = 0
        for crop, (x, y, w, h) in proposals:
            if crop.size == 0:
                continue
                
            # Preprocess crop to 128x128x3 normal MobileNetV2 input
            crop_resized = cv2.resize(crop, (128, 128))
            crop_rgb = cv2.cvtColor(crop_resized, cv2.COLOR_BGR2RGB)
            crop_norm = (crop_rgb.astype(np.float32) / 127.5) - 1.0
            crop_input = np.expand_dims(crop_norm, axis=0)
            
            # TFLite inference
            interpreter.set_tensor(input_details[0]['index'], crop_input)
            interpreter.invoke()
            probs = interpreter.get_tensor(output_details[0]['index'])[0]
            
            pred_idx = np.argmax(probs)
            confidence = probs[pred_idx]
            pred_class = target_classes[pred_idx]
            
            # Suppress detections classified as Background (negative filter) or low confidence
            if pred_class != "Background" and confidence > 0.35:
                valid_detections += 1
                
                # Visual overlays: Draw bounding box
                cv2.rectangle(annotated_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                
                # Draw label card with semi-transparent background
                label_text = f"{pred_class.upper()} ({confidence*100:.1f}%)"
                (text_w, text_h), baseline = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
                
                # Bounding label background rectangle
                cv2.rectangle(
                    annotated_frame, 
                    (x, y - text_h - 6), 
                    (x + text_w + 6, y), 
                    (0, 255, 0), 
                    cv2.FILLED
                )
                # Label text
                cv2.putText(
                    annotated_frame, 
                    label_text, 
                    (x + 3, y - 4), 
                    cv2.FONT_HERSHEY_SIMPLEX, 
                    0.45, 
                    (0, 0, 0), 
                    1, 
                    cv2.LINE_AA
                )
                
        print(f"  - Pipeline detected and labeled {valid_detections} valid waste objects")
        
        # Save output visual frame and threshold debug mask side-by-side
        # Resize debug mask to match original frame size color channel format
        debug_mask_bgr = cv2.cvtColor(debug_mask, cv2.COLOR_GRAY2BGR)
        combined_display = np.hstack((annotated_frame, debug_mask_bgr))
        
        out_name = f"hybrid_det_{img_path.stem}.png"
        cv2.imwrite(str(OUT_DIR / out_name), combined_display)
        print(f"  [OK] Visualized side-by-side result saved to: {OUT_DIR / out_name}")
        
    print(f"\n[OK] Detection by Classification pipeline run completed successfully.")

if __name__ == "__main__":
    main()
