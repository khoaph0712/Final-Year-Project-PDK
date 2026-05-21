import sys
import os
import csv
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Robust Keras imports for TF 2.16+
try:
    import tf_keras as keras
except ImportError:
    from tensorflow import keras

import tensorflow as tf
MobileNetV2 = keras.applications.MobileNetV2
preprocess_input = keras.applications.mobilenet_v2.preprocess_input
Model = keras.models.Model
Dense = keras.layers.Dense
GlobalAveragePooling2D = keras.layers.GlobalAveragePooling2D
Dropout = keras.layers.Dropout
Input = keras.layers.Input
Adam = keras.optimizers.Adam
ModelCheckpoint = keras.callbacks.ModelCheckpoint
CSVLogger = keras.callbacks.CSVLogger
EarlyStopping = keras.callbacks.EarlyStopping

import cv2

# Add scripts directory to path if needed
sys.path.append(str(Path(__file__).resolve().parent))

from ml_balanced_training import load_crops_and_balance

ROOT = Path(__file__).resolve().parent.parent
DATA_YAML = ROOT / "merged_dataset_v5" / "data.yaml"
OUT_DIR = ROOT / "runs" / "dl" / "cnn_mobilenet"

def preprocess_crops(crops, target_size=(128, 128)):
    """Resize crops and convert to float numpy array preprocessed for MobileNetV2."""
    processed = []
    for crop in crops:
        # Resize using bilinear interpolation
        resized = cv2.resize(crop, target_size, interpolation=cv2.INTER_LINEAR)
        # Convert BGR to RGB
        resized_rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        processed.append(resized_rgb)
    
    # Convert to array and scale to [-1, 1] using MobileNetV2 preprocess
    processed_arr = np.array(processed, dtype=np.float32)
    return preprocess_input(processed_arr)

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    
    print("====================================================")
    print("FYP Waste Management: Training Keras MobileNetV2 CNN")
    print("====================================================")
    
    target_classes = ["plastic", "glass", "metal", "paper", "cardboard", "organic", "Background"]
    
    # 1. Load balanced crop splits
    print("[INFO] Loading balanced crops from dataset splits...")
    train_crops, y_train_list = load_crops_and_balance(
        DATA_YAML, target_classes, max_per_class=3500, is_train=True, seed=42
    )
    test_crops, y_test_list = load_crops_and_balance(
        DATA_YAML, target_classes, max_per_class=800, is_train=False, seed=42
    )
    
    # Convert labels to numpy arrays
    y_train = np.array(y_train_list, dtype=np.int32)
    y_test = np.array(y_test_list, dtype=np.int32)
    
    # 2. Preprocess images
    print("\n[INFO] Resizing and preprocessing image crops to 128x128...")
    x_train = preprocess_crops(train_crops)
    x_test = preprocess_crops(test_crops)
    
    print(f"Dataset shapes:")
    print(f"  - X_train: {x_train.shape}, y_train: {y_train.shape}")
    print(f"  - X_test: {x_test.shape}, y_test: {y_test.shape}")
    
    # 3. Build MobileNetV2 Model
    print("\n[INFO] Loading pre-trained MobileNetV2 base...")
    base_model = MobileNetV2(
        input_shape=(128, 128, 3),
        include_top=False,
        weights="imagenet"
    )
    
    # Freeze the base model weights initially
    base_model.trainable = False
    
    # Add custom classification head
    inputs = Input(shape=(128, 128, 3))
    x = base_model(inputs, training=False)
    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.35)(x)
    x = Dense(128, activation="relu")(x)
    x = Dropout(0.2)(x)
    outputs = Dense(len(target_classes), activation="softmax")(x)
    
    model = Model(inputs, outputs)
    
    # Phase 1: Warmup custom classification head
    print("\n--- Phase 1: Training Custom Classification Head (3 Epochs) ---")
    model.compile(
        optimizer=Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    
    model.fit(
        x_train, y_train,
        validation_data=(x_test, y_test),
        epochs=3,
        batch_size=64
    )
    
    # Phase 2: Fine-tune Upper MobileNetV2 Layers
    print("\n--- Phase 2: Fine-Tuning Top MobileNetV2 Layers (9 Epochs) ---")
    # Unfreeze the base model
    base_model.trainable = True
    # Freeze all layers except the last 30 layers
    for layer in base_model.layers[:-30]:
        layer.trainable = False
        
    # Re-compile with low learning rate
    model.compile(
        optimizer=Adam(learning_rate=1e-4),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    
    # Set up callbacks
    best_weights_path = OUT_DIR / "best_mobilenet.h5"
    history_csv_path = OUT_DIR / "training_history.csv"
    
    callbacks = [
        ModelCheckpoint(
            filepath=str(best_weights_path),
            monitor="val_accuracy",
            save_best_only=True,
            mode="max",
            verbose=1
        ),
        CSVLogger(
            filename=str(history_csv_path),
            append=True
        )
    ]
    
    # Train model
    history = model.fit(
        x_train, y_train,
        validation_data=(x_test, y_test),
        epochs=9,
        batch_size=64,
        callbacks=callbacks
    )
    
    print(f"\n[OK] Training complete. Best weights saved to: {best_weights_path}")
    
    # 4. Save training plots
    print("\n[INFO] Saving training history plots...")
    # Load and combine phase 1 history if possible, or plot phase 2 history
    history_dict = history.history
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    ax1.plot(history_dict["loss"], label="Train Loss", color="#1f77b4")
    ax1.plot(history_dict["val_loss"], label="Val Loss", color="#ff7f0e")
    ax1.set_title("CNN Loss History (Phase 2 Fine-Tuning)")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    ax2.plot(history_dict["accuracy"], label="Train Acc", color="#2ca02c")
    ax2.plot(history_dict["val_accuracy"], label="Val Acc", color="#d62728")
    ax2.set_title("CNN Accuracy History (Phase 2 Fine-Tuning)")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    fig.tight_layout()
    fig.savefig(OUT_DIR / "training_plots.png", dpi=120)
    plt.close(fig)
    print(f"[INFO] Training history plots saved to: {OUT_DIR / 'training_plots.png'}")
    
    # 5. Final Evaluation on Test Set
    best_model = keras.models.load_model(str(best_weights_path))
    test_loss, test_acc = best_model.evaluate(x_test, y_test, verbose=0)
    print(f"\nCNN Fine-Tuned Model Performance on Test Crops:")
    print(f"  - Test Loss: {test_loss:.4f}")
    print(f"  - Test Accuracy: {test_acc:.4f}")

if __name__ == "__main__":
    main()
