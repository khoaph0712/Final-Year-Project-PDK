#!/usr/bin/env python
"""
FYP Waste Sorting - Master Confusion Matrix Evaluator (Stage 2)
Evaluates Tuned EfficientNetB0 (Ours), MobileNetV2, and ResNet50.
Generates individual confusion matrices and compiles them into a beautiful 1x3 grid.
"""

import sys
import os
import time
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Robust Keras imports
try:
    import tf_keras as keras
except ImportError:
    from tensorflow import keras

import tensorflow as tf
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import cv2

# Add paths for local modules
SCRIPTS_DIR = Path(__file__).resolve().parent
ROOT = SCRIPTS_DIR.parent
sys.path.append(str(SCRIPTS_DIR))
sys.path.append(str(SCRIPTS_DIR / "archive"))

from ml_balanced_training import load_crops_and_balance

DATA_YAML = ROOT / "data" / "merged_dataset_v5" / "data.yaml"
OUT_DIR = ROOT / "runs" / "dl" / "comparison_models"

# Model Paths
EFFNET_PATH = ROOT / "models" / "trained" / "efficientnet_classifier" / "best_efficientnet_tuned.h5"
MOBILENET_PATH = ROOT / "models" / "trained" / "comparison_baselines" / "best_mobilenetv2.h5"
RESNET_PATH = ROOT / "models" / "trained" / "comparison_baselines" / "best_resnet50.h5"

def preprocess_crops(crops, preprocess_fn, target_size=(224, 224)):
    processed = []
    for crop in crops:
        resized = cv2.resize(crop, target_size, interpolation=cv2.INTER_LINEAR)
        resized_rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        processed.append(resized_rgb)
    processed_arr = np.array(processed, dtype=np.float32)
    return preprocess_fn(processed_arr)

def evaluate_model(model_path, preprocess_fn, test_crops, y_test, target_classes, model_name, cmap_name):
    print(f"\n--- Evaluating Model: {model_name} ---")
    if not model_path.exists():
        print(f"[ERROR] Weights not found for {model_name} at {model_path}!")
        return None, None
        
    model = keras.models.load_model(str(model_path))
    x_test = preprocess_crops(test_crops, preprocess_fn)
    
    # Predict
    probs = model.predict(x_test, batch_size=64, verbose=1)
    preds = np.argmax(probs, axis=1)
    
    # Generate confusion matrix
    cm = confusion_matrix(y_test, preds, labels=list(range(len(target_classes))))
    
    # Save individual confusion matrix plot
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=target_classes)
    disp.plot(ax=ax, cmap=cmap_name, xticks_rotation=35, colorbar=False)
    ax.set_title(f"Confusion Matrix: {model_name}", fontsize=11, fontweight='bold')
    fig.tight_layout()
    
    plot_path = OUT_DIR / f"confusion_{model_name.lower().replace(' ', '_').replace('(', '').replace(')', '')}.png"
    fig.savefig(plot_path, dpi=130)
    plt.close(fig)
    print(f"[OK] Saved {model_name} confusion matrix to {plot_path}")
    
    tf.keras.backend.clear_session()
    return preds, cm

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    
    target_classes = ["plastic", "glass", "metal", "paper", "cardboard", "organic", "Background"]
    
    print("[INFO] Loading balanced test crops for evaluation...")
    test_crops, y_test_list = load_crops_and_balance(
        DATA_YAML, target_classes, max_per_class=300, is_train=False, seed=42
    )
    y_test = np.array(y_test_list, dtype=np.int32)
    
    # Preprocessing functions
    effnet_preprocess = keras.applications.efficientnet.preprocess_input
    mobilenet_preprocess = keras.applications.mobilenet_v2.preprocess_input
    resnet_preprocess = keras.applications.resnet50.preprocess_input
    
    results = {}
    
    # Evaluate Tuned EfficientNetB0
    preds_eff, cm_eff = evaluate_model(
        EFFNET_PATH, effnet_preprocess, test_crops, y_test, target_classes, "EfficientNetB0 (Ours)", "Greens"
    )
    if cm_eff is not None:
        results["EfficientNetB0"] = cm_eff
        
    # Evaluate MobileNetV2
    preds_mob, cm_mob = evaluate_model(
        MOBILENET_PATH, mobilenet_preprocess, test_crops, y_test, target_classes, "MobileNetV2", "Blues"
    )
    if cm_mob is not None:
        results["MobileNetV2"] = cm_mob
        
    # Evaluate ResNet50
    preds_res, cm_res = evaluate_model(
        RESNET_PATH, resnet_preprocess, test_crops, y_test, target_classes, "ResNet50", "Oranges"
    )
    if cm_res is not None:
        results["ResNet50"] = cm_res
        
    # Generate 1x3 comparative grid plot
    if len(results) > 0:
        print("\n--- Generating Master Comparative Confusion Grid ---")
        fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
        
        cmaps = {
            "EfficientNetB0": ("Greens", "EfficientNetB0 (Ours)"),
            "MobileNetV2": ("Blues", "MobileNetV2 Baseline"),
            "ResNet50": ("Oranges", "ResNet50 Baseline")
        }
        
        for idx, (mname, cm) in enumerate(results.items()):
            cmap, title = cmaps[mname]
            ax = axes[idx]
            disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=target_classes)
            disp.plot(ax=ax, cmap=cmap, xticks_rotation=35, colorbar=False)
            ax.set_title(title, fontsize=12, fontweight='bold')
            
        fig.tight_layout()
        grid_path = OUT_DIR / "confusion_matrix_grid.png"
        fig.savefig(grid_path, dpi=150)
        plt.close(fig)
        print(f"[SUCCESS] Comparative confusion grid saved to: {grid_path}")

if __name__ == "__main__":
    main()
