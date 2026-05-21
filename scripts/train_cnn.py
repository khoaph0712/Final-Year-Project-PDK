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
EfficientNetB0 = keras.applications.EfficientNetB0
preprocess_input = keras.applications.efficientnet.preprocess_input
Model = keras.models.Model
Dense = keras.layers.Dense
GlobalAveragePooling2D = keras.layers.GlobalAveragePooling2D
Dropout = keras.layers.Dropout
Input = keras.layers.Input
Adam = keras.optimizers.Adam
ModelCheckpoint = keras.callbacks.ModelCheckpoint
CSVLogger = keras.callbacks.CSVLogger
EarlyStopping = keras.callbacks.EarlyStopping
Sequence = keras.utils.Sequence

import cv2

# Add scripts and archive directory to path if needed
sys.path.append(str(Path(__file__).resolve().parent))
sys.path.append(str(Path(__file__).resolve().parent / "archive"))

from ml_balanced_training import load_crops_and_balance

ROOT = Path(__file__).resolve().parent.parent
DATA_YAML = ROOT / "data" / "merged_dataset_v5" / "data.yaml"
OUT_DIR = ROOT / "runs" / "dl" / "cnn_efficientnet"

def preprocess_crops(crops, target_size=(224, 224)):
    """Resize crops and convert to float numpy array preprocessed for EfficientNetB0 (used for validation/test evaluation if small)."""
    processed = []
    for crop in crops:
        resized = cv2.resize(crop, target_size, interpolation=cv2.INTER_LINEAR)
        resized_rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        processed.append(resized_rgb)
    processed_arr = np.array(processed, dtype=np.float32)
    return preprocess_input(processed_arr)

class CropSequence(Sequence):
    def __init__(self, crops, labels, batch_size=64, target_size=(224, 224)):
        self.crops = crops
        self.labels = labels
        self.batch_size = batch_size
        self.target_size = target_size
        self.indices = np.arange(len(self.crops))
        
    def __len__(self):
        return int(np.ceil(len(self.crops) / self.batch_size))
        
    def __getitem__(self, idx):
        batch_indices = self.indices[idx * self.batch_size : (idx + 1) * self.batch_size]
        batch_x = []
        batch_y = []
        for i in batch_indices:
            crop = self.crops[i]
            # Resize
            resized = cv2.resize(crop, self.target_size, interpolation=cv2.INTER_LINEAR)
            # BGR to RGB
            resized_rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
            batch_x.append(resized_rgb)
            batch_y.append(self.labels[i])
            
        batch_x_arr = np.array(batch_x, dtype=np.float32)
        batch_x_preprocessed = preprocess_input(batch_x_arr)
        return batch_x_preprocessed, np.array(batch_y, dtype=np.int32)
        
    def on_epoch_end(self):
        np.random.shuffle(self.indices)

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    
    print("====================================================")
    print("FYP Waste Management: Training Keras EfficientNetB0 CNN")
    print("====================================================")
    
    target_classes = ["plastic", "glass", "metal", "paper", "cardboard", "organic", "Background"]
    
    # 1. Load balanced crop splits
    print("[INFO] Loading balanced crops from dataset splits...")
    train_crops, y_train_list = load_crops_and_balance(
        DATA_YAML, target_classes, max_per_class=2000, is_train=True, seed=42
    )
    test_crops, y_test_list = load_crops_and_balance(
        DATA_YAML, target_classes, max_per_class=500, is_train=False, seed=42
    )
    
    # Convert labels to numpy arrays
    y_train = np.array(y_train_list, dtype=np.int32)
    y_test = np.array(y_test_list, dtype=np.int32)
    
    # 2. Setup memory-safe crop sequences
    print("\n[INFO] Initializing memory-safe CropSequence generators...")
    train_gen = CropSequence(train_crops, y_train, batch_size=64, target_size=(224, 224))
    test_gen = CropSequence(test_crops, y_test, batch_size=64, target_size=(224, 224))
    
    print(f"Dataset shapes:")
    print(f"  - Train crops count: {len(train_crops)}, Labels count: {len(y_train)}")
    print(f"  - Test crops count: {len(test_crops)}, Labels count: {len(y_test)}")
    
    # 3. Build EfficientNetB0 Model
    print("\n[INFO] Loading pre-trained EfficientNetB0 base...")
    base_model = EfficientNetB0(
        input_shape=(224, 224, 3),
        include_top=False,
        weights="imagenet"
    )
    
    # Freeze the base model weights initially
    base_model.trainable = False
    
    # Add custom classification head
    inputs = Input(shape=(224, 224, 3))
    x = base_model(inputs, training=False)
    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.35)(x)
    x = Dense(128, activation="relu")(x)
    x = Dropout(0.2)(x)
    outputs = Dense(len(target_classes), activation="softmax")(x)
    
    model = Model(inputs, outputs)
    
    # Phase 1: Warmup custom classification head
    print("\n--- Phase 1: Training Custom Classification Head (1 Epochs) ---")
    model.compile(
        optimizer=Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    
    model.fit(
        train_gen,
        validation_data=test_gen,
        epochs=1
    )
    
    # Phase 2: Fine-tune Upper EfficientNetB0 Layers
    print("\n--- Phase 2: Fine-Tuning Top EfficientNetB0 Layers (2 Epochs) ---")
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
    best_weights_path = OUT_DIR / "best_efficientnet.h5"
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
        train_gen,
        validation_data=test_gen,
        epochs=2,
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
    test_loss, test_acc = best_model.evaluate(test_gen, verbose=0)
    print(f"\nCNN Fine-Tuned Model Performance on Test Crops:")
    print(f"  - Test Loss: {test_loss:.4f}")
    print(f"  - Test Accuracy: {test_acc:.4f}")

if __name__ == "__main__":
    main()
