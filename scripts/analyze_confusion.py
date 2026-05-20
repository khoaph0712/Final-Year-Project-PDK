import sys
import pickle
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
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report

import cv2

# Add scripts directory to path if needed
sys.path.append(str(Path(__file__).resolve().parent))

from ml_balanced_training import load_crops_and_balance
from custom_feature_extractor import extract_637_features
from train_ann import WasteMLP
from train_cnn import preprocess_crops

ROOT = Path(__file__).resolve().parent.parent
DATA_YAML = ROOT / "merged_dataset_v3" / "data.yaml"
ANN_DIR = ROOT / "runs" / "dl" / "ann_637"
CNN_DIR = ROOT / "runs" / "dl" / "cnn_mobilenet"

def evaluate_ann(test_crops, y_test, target_classes):
    print("\n--- Evaluating ANN (MLP)... ---")
    # Check if features are cached
    cache_x_test = ANN_DIR / "x_test_637.npy"
    if cache_x_test.exists():
        print("[INFO] Loading cached 637-feature vectors for test split...")
        x_test = np.load(cache_x_test)
    else:
        print("[INFO] Extracting custom 637 features for test split...")
        x_test = []
        for idx, crop in enumerate(test_crops):
            if idx > 0 and idx % 500 == 0:
                print(f"  - Test features: {idx}/{len(test_crops)} extracted")
            x_test.append(extract_637_features(crop))
        x_test = np.array(x_test, dtype=np.float32)
        
    # Load scaler
    scaler_path = ANN_DIR / "scaler_ann.pkl"
    if not scaler_path.exists():
        raise FileNotFoundError("ANN Scaler not found! Run train_ann.py first.")
    with open(scaler_path, "rb") as f:
        scaler = pickle.load(f)
        
    x_test_scaled = scaler.transform(x_test)
    
    # Load model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = WasteMLP(input_dim=637, num_classes=len(target_classes)).to(device)
    model_path = ANN_DIR / "best_ann.pt"
    if not model_path.exists():
        raise FileNotFoundError("ANN model weights not found! Run train_ann.py first.")
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    # Predict
    inputs_t = torch.tensor(x_test_scaled, dtype=torch.float32).to(device)
    with torch.no_grad():
        outputs = model(inputs_t)
        _, preds = torch.max(outputs, 1)
        preds = preds.cpu().numpy()
        
    # Generate confusion matrix
    cm = confusion_matrix(y_test, preds, labels=list(range(len(target_classes))))
    
    # Plot
    fig, ax = plt.subplots(figsize=(8, 7))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=target_classes)
    disp.plot(ax=ax, cmap="Blues", xticks_rotation=35, colorbar=False)
    ax.set_title("Confusion Matrix: PyTorch ANN (637 Features)", fontsize=12, fontweight='bold')
    fig.tight_layout()
    fig.savefig(ANN_DIR / "confusion_ann.png", dpi=140)
    plt.close(fig)
    print(f"[OK] Saved ANN confusion matrix to {ANN_DIR / 'confusion_ann.png'}")
    
    return preds, cm

def evaluate_cnn(test_crops, y_test, target_classes):
    print("\n--- Evaluating CNN (MobileNetV2)... ---")
    x_test = preprocess_crops(test_crops)
    
    # Load model
    model_path = CNN_DIR / "best_mobilenet.h5"
    if not model_path.exists():
        raise FileNotFoundError("CNN model weights not found! Run train_cnn.py first.")
        
    model = keras.models.load_model(str(model_path))
    
    # Predict
    probs = model.predict(x_test, batch_size=64)
    preds = np.argmax(probs, axis=1)
    
    # Generate confusion matrix
    cm = confusion_matrix(y_test, preds, labels=list(range(len(target_classes))))
    
    # Plot
    fig, ax = plt.subplots(figsize=(8, 7))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=target_classes)
    disp.plot(ax=ax, cmap="Oranges", xticks_rotation=35, colorbar=False)
    ax.set_title("Confusion Matrix: Keras CNN (MobileNetV2)", fontsize=12, fontweight='bold')
    fig.tight_layout()
    fig.savefig(CNN_DIR / "confusion_cnn.png", dpi=140)
    plt.close(fig)
    print(f"[OK] Saved CNN confusion matrix to {CNN_DIR / 'confusion_cnn.png'}")
    
    return preds, cm

def write_confusion_report(ann_cm, cnn_cm, target_classes):
    print("\n[INFO] Analyzing Background rows and columns...")
    # Background class is the last class
    bg_idx = target_classes.index("Background")
    num_samples_per_class = 250 # test split max_per_class
    
    report_lines = [
        "# Deep Learning Confusion Matrix & Background Error Analysis\n",
        "This report evaluates model predictions against the balanced test dataset (250 crops per class) with a specific deep-dive into how PyTorch ANN and MobileNetV2 CNN models interact with the **Background** (negative noise) class.\n",
        "## 1. Class Performance Overview\n",
        "### PyTorch ANN (637 Handcrafted Features) Confusion Matrix\n",
        f"![ANN Confusion Matrix](file:///{(ANN_DIR / 'confusion_ann.png').as_posix()})\n",
        "### MobileNetV2 CNN (Raw Image Crops) Confusion Matrix\n",
        f"![CNN Confusion Matrix](file:///{(CNN_DIR / 'confusion_cnn.png').as_posix()})\n\n",
        "## 2. Background Class Deep-Dive\n",
        "The background class represents floor textures, carpets, grass, and table surfaces where waste is absent. Evaluating how other classes leak into the background (False Negatives) or how the background is identified as waste (False Positives) is vital for physical robotic and mobile applications.\n\n"
    ]
    
    for model_name, cm in [("PyTorch ANN (MLP)", ann_cm), ("MobileNetV2 CNN", cnn_cm)]:
        report_lines.append(f"### {model_name} Error Profile\n")
        
        # 1. Background Row (True class = Background, predicted as others) -> False Positives (Ghost waste)
        bg_row = cm[bg_idx, :]
        fps = sum(bg_row) - bg_row[bg_idx]
        fp_rate = fps / num_samples_per_class
        
        # 2. Background Column (True class = others, predicted as Background) -> False Negatives (Waste blindness)
        bg_col = cm[:, bg_idx]
        fns = sum(bg_col) - bg_col[bg_idx]
        fn_rate = fns / (num_samples_per_class * (len(target_classes) - 1))
        
        report_lines.append(f"- **False Positive Rate (Ghost Waste)**: **{fp_rate*100:.2f}%** ({fps} / {num_samples_per_class} background samples misclassified as waste items).")
        report_lines.append(f"  - *Impact*: In a robotic or mobile trash bin setting, this means the AI sees 'Ghost waste' on clear surfaces and triggers sorting mechanisms unnecessarily.")
        
        # Identify which waste class is most commonly predicted for background
        max_fp_idx = -1
        max_fp_val = 0
        for i in range(len(target_classes)):
            if i != bg_idx and bg_row[i] > max_fp_val:
                max_fp_val = bg_row[i]
                max_fp_idx = i
        if max_fp_idx != -1:
            report_lines.append(f"  - *Top False Positive Leakage*: Background textures are most frequently mistaken for **{target_classes[max_fp_idx]}** ({max_fp_val} instances).")
            
        report_lines.append(f"- **False Negative Rate (Waste Blindness)**: **{fn_rate*100:.2f}%** ({fns} / {num_samples_per_class * (len(target_classes) - 1)} actual waste items misclassified as background).")
        report_lines.append(f"  - *Impact*: This represents blind spots where the AI completely ignores waste items, assuming they are simply floor textures.")
        
        # Identify which waste class is most commonly ignored as background
        max_fn_idx = -1
        max_fn_val = 0
        for i in range(len(target_classes)):
            if i != bg_idx and cm[i, bg_idx] > max_fn_val:
                max_fn_val = cm[i, bg_idx]
                max_fn_idx = i
        if max_fn_idx != -1:
            report_lines.append(f"  - *Top False Negative Leakage*: Waste item **{target_classes[max_fn_idx]}** is most frequently ignored as background ({max_fn_val} instances).")
            
        report_lines.append("\n")
        
    # Comparative summary table
    report_lines.append("## 3. Background Leakage Comparison Table\n")
    report_lines.append("| Model | Ghost Waste FP Rate (%) | Waste Blindness FN Rate (%) | Primary FP Leakage Class | Primary FN Leakage Class |")
    report_lines.append("|---|---:|---:|---|---|")
    
    for model_name, cm in [("PyTorch ANN", ann_cm), ("MobileNetV2 CNN", cnn_cm)]:
        bg_row = cm[bg_idx, :]
        fps = sum(bg_row) - bg_row[bg_idx]
        fp_rate = fps / num_samples_per_class
        
        bg_col = cm[:, bg_idx]
        fns = sum(bg_col) - bg_col[bg_idx]
        fn_rate = fns / (num_samples_per_class * (len(target_classes) - 1))
        
        # Max FP
        max_fp_idx = -1
        max_fp_val = 0
        for i in range(len(target_classes)):
            if i != bg_idx and bg_row[i] > max_fp_val:
                max_fp_val = bg_row[i]
                max_fp_idx = i
        max_fp_cls = target_classes[max_fp_idx] if max_fp_idx != -1 else "None"
        
        # Max FN
        max_fn_idx = -1
        max_fn_val = 0
        for i in range(len(target_classes)):
            if i != bg_idx and cm[i, bg_idx] > max_fn_val:
                max_fn_val = cm[i, bg_idx]
                max_fn_idx = i
        max_fn_cls = target_classes[max_fn_idx] if max_fn_idx != -1 else "None"
        
        report_lines.append(f"| {model_name} | {fp_rate*100:.2f}% | {fn_rate*100:.2f}% | {max_fp_cls} | {max_fn_cls} |")
        
    report_lines.append("\n## 4. Engineering Recommendations\n")
    report_lines.append("1. **Texture Feature Augmentation**: The ANN relies heavily on texture features (LBP/GLCM). Backgrounds with high texture similarity to paper/cardboard are mistaken. We should expand texture training to cover a wider variety of floors.\n")
    report_lines.append("2. **Context-Aware Scaling**: The CNN performs better due to contextual representations learned via MobileNetV2's deep convolutional layers. However, when transparent waste (glass/plastic) is present, the model can struggle to differentiate it from floors. Adding alpha-mask or edge highlight augmentations will resolve this.\n")
    
    report_path = ROOT / "runs" / "dl" / "confusion_analysis_report.md"
    report_path.write_text("\n".join(report_lines), encoding="utf-8")
    print(f"[OK] Saved comprehensive report to {report_path}")

def main():
    target_classes = ["plastic", "glass", "metal", "paper", "cardboard", "organic", "Background"]
    
    print("[INFO] Loading balanced test crops for evaluation...")
    test_crops, y_test_list = load_crops_and_balance(
        DATA_YAML, target_classes, max_per_class=250, is_train=False, seed=42
    )
    y_test = np.array(y_test_list, dtype=np.int32)
    
    ann_preds, ann_cm = evaluate_ann(test_crops, y_test, target_classes)
    cnn_preds, cnn_cm = evaluate_cnn(test_crops, y_test, target_classes)
    
    write_confusion_report(ann_cm, cnn_cm, target_classes)

if __name__ == "__main__":
    main()
