import sys
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
Model = keras.models.Model

import cv2

# Add scripts directory to path if needed
sys.path.append(str(Path(__file__).resolve().parent))

from ml_balanced_training import load_crops_and_balance
from train_cnn import preprocess_crops

ROOT = Path(__file__).resolve().parent.parent
DATA_YAML = ROOT / "merged_dataset_v3" / "data.yaml"
CNN_DIR = ROOT / "runs" / "dl" / "cnn_mobilenet"
OUT_DIR = CNN_DIR / "gradcam_results"

def compute_gradcam(grad_model, img_array, class_idx):
    """Compute Grad-CAM heatmap for a single image and class."""
    # Ensure img_array has batch dimension
    if len(img_array.shape) == 3:
        img_array = np.expand_dims(img_array, axis=0)
        
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        loss = predictions[:, class_idx]
        
    # Compute gradients of loss with respect to conv outputs
    grads = tape.gradient(loss, conv_outputs)
    
    # Global average pooling of gradients
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    
    # Weight the activation maps by pooled gradients
    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    
    # Apply ReLU to keep only positive features
    heatmap = tf.maximum(heatmap, 0.0)
    
    # Normalize heatmap
    max_val = tf.reduce_max(heatmap)
    if max_val == 0.0:
        max_val = 1e-10
    heatmap = heatmap / max_val
    return heatmap.numpy()

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    
    print("====================================================")
    print("FYP Waste Management: Generating Grad-CAM Heatmaps...")
    print("====================================================")
    
    target_classes = ["plastic", "glass", "metal", "paper", "cardboard", "organic", "Background"]
    
    # Load model
    model_path = CNN_DIR / "best_mobilenet.h5"
    if not model_path.exists():
        raise FileNotFoundError("CNN model weights not found! Run train_cnn.py first.")
    
    model = keras.models.load_model(str(model_path))
    
    # Find base model layer
    base_model = None
    for layer in model.layers:
        if isinstance(layer, keras.Model) or (hasattr(layer, 'layers') and len(layer.layers) > 0):
            base_model = layer
            break
            
    if base_model is None:
        raise ValueError("Could not locate the nested MobileNetV2 base model in the Keras model.")
        
    print(f"[INFO] Found base model: {base_model.name}")
    last_conv_layer = base_model.get_layer("out_relu")
    print(f"[INFO] Found target conv layer: {last_conv_layer.name} with shape {last_conv_layer.output.shape}")
    
    # Reconstruct the head layers applied to base_model.output
    x = base_model.output
    base_idx = model.layers.index(base_model)
    for layer in model.layers[base_idx + 1:]:
        x = layer(x)
        
    # Create submodel outputting intermediate conv activation and final prediction
    grad_model = Model(base_model.inputs, [last_conv_layer.output, x])
    
    # Load test crops
    print("[INFO] Loading test crops...")
    test_crops, y_test_list = load_crops_and_balance(
        DATA_YAML, target_classes, max_per_class=250, is_train=False, seed=42
    )
    y_test = np.array(y_test_list, dtype=np.int32)
    
    # Preprocess
    x_test = preprocess_crops(test_crops)
    
    # Predict to find correct samples for each class
    probs = model.predict(x_test, batch_size=64)
    preds = np.argmax(probs, axis=1)
    
    # Set up gallery plotting
    fig, axes = plt.subplots(len(target_classes), 3, figsize=(9, len(target_classes) * 3))
    
    print("\n--- Visualizing Attention Maps ---")
    for cls_idx, cname in enumerate(target_classes):
        # Find index of a crop belonging to this class that was correctly predicted
        correct_indices = np.where((y_test == cls_idx) & (preds == cls_idx))[0]
        
        # Fallback to any sample of this class if none correctly predicted
        if len(correct_indices) == 0:
            correct_indices = np.where(y_test == cls_idx)[0]
            
        if len(correct_indices) == 0:
            print(f"[WARN] No crops found for class '{cname}'. Skipping.")
            continue
            
        crop_idx = correct_indices[0]
        raw_crop = test_crops[crop_idx]
        preprocessed_img = x_test[crop_idx]
        
        # Compute Grad-CAM
        heatmap = compute_gradcam(grad_model, preprocessed_img, cls_idx)
        
        # Resize raw crop for consistent display (128x128)
        raw_display = cv2.resize(raw_crop, (128, 128))
        raw_display = cv2.cvtColor(raw_display, cv2.COLOR_BGR2RGB)
        
        # Resize heatmap to match image size
        heatmap_resized = cv2.resize(heatmap, (128, 128))
        
        # Create RGB Heatmap and overlay it
        heatmap_color = np.uint8(255 * heatmap_resized)
        heatmap_colored = cv2.applyColorMap(heatmap_color, cv2.COLORMAP_JET)
        heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
        
        # Superimpose
        overlay = cv2.addWeighted(raw_display, 0.6, heatmap_colored, 0.4, 0)
        
        # Save individual plot
        ind_fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(10, 3.5))
        ax1.imshow(raw_display)
        ax1.set_title("Raw Image Crop")
        ax1.axis("off")
        
        ax2.imshow(heatmap_resized, cmap="jet")
        ax2.set_title("Grad-CAM Heatmap")
        ax2.axis("off")
        
        ax3.imshow(overlay)
        ax3.set_title("Overlay (Focus)")
        ax3.axis("off")
        
        ind_fig.suptitle(f"Class: {cname.upper()} (Grad-CAM Saliency)", fontsize=12, fontweight='bold')
        ind_fig.tight_layout()
        ind_fig.savefig(OUT_DIR / f"gradcam_{cname}.png", dpi=120)
        plt.close(ind_fig)
        print(f"  * Class '{cname}' heatmap saved to {OUT_DIR / f'gradcam_{cname}.png'}")
        
        # Add to gallery
        ax_raw, ax_heat, ax_over = axes[cls_idx]
        
        ax_raw.imshow(raw_display)
        ax_raw.set_title(f"{cname.upper()}\nRaw Crop", fontsize=9, fontweight='bold')
        ax_raw.axis("off")
        
        ax_heat.imshow(heatmap_resized, cmap="jet")
        ax_heat.set_title("Heatmap", fontsize=9)
        ax_heat.axis("off")
        
        ax_over.imshow(overlay)
        ax_over.set_title("Focus Overlay", fontsize=9)
        ax_over.axis("off")
        
    fig.tight_layout()
    gallery_path = OUT_DIR / "gradcam_gallery.png"
    fig.savefig(gallery_path, dpi=150)
    plt.close(fig)
    print(f"\n[OK] Saliency gallery generated at: {gallery_path}")

if __name__ == "__main__":
    main()
