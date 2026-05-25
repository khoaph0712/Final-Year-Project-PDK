import sys
import os
import csv
import json
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
OUT_DIR = ROOT / "runs" / "dl" / "cnn_efficientnet_tuned"

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

def build_model(num_classes):
    """Builds the custom EfficientNetB0 model."""
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
    outputs = Dense(num_classes, activation="softmax")(x)
    
    model = Model(inputs, outputs)
    return model, base_model

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    
    print("====================================================")
    print("FYP Waste Management: Hyperparameter Tuning CNN")
    print("====================================================")
    
    target_classes = ["plastic", "glass", "metal", "paper", "cardboard", "organic", "Background"]
    num_classes = len(target_classes)
    
    # 1. Load balanced crop splits
    print("[INFO] Loading balanced crops from dataset splits...")
    train_crops, y_train_list = load_crops_and_balance(
        DATA_YAML, target_classes, max_per_class=2000, is_train=True, seed=42
    )
    test_crops, y_test_list = load_crops_and_balance(
        DATA_YAML, target_classes, max_per_class=500, is_train=False, seed=42
    )
    
    y_train = np.array(y_train_list, dtype=np.int32)
    y_test = np.array(y_test_list, dtype=np.int32)
    
    # 2. Hyperparameter Grid Setup
    learning_rates = [1e-3, 1e-4, 5e-5]
    batch_sizes = [32, 64]
    
    tuning_results = []
    
    print("\n--- PHASE 1: HYPERPARAMETER TUNING SWEEP (Grid Search) ---")
    for lr in learning_rates:
        for batch_size in batch_sizes:
            print(f"\n[EVALUATING] Learning Rate: {lr} | Batch Size: {batch_size}")
            
            # Setup crop sequences for this batch size
            train_gen = CropSequence(train_crops, y_train, batch_size=batch_size, target_size=(224, 224))
            test_gen = CropSequence(test_crops, y_test, batch_size=batch_size, target_size=(224, 224))
            
            # Build and compile model
            model, base_model = build_model(num_classes)
            
            # Warm up head (1 epoch)
            model.compile(
                optimizer=Adam(learning_rate=1e-3),
                loss="sparse_categorical_crossentropy",
                metrics=["accuracy"]
            )
            model.fit(train_gen, validation_data=test_gen, epochs=1, verbose=0)
            
            # Fine-tune Top 30 layers (3 epochs for quick tuning evaluation)
            base_model.trainable = True
            for layer in base_model.layers[:-30]:
                layer.trainable = False
                
            model.compile(
                optimizer=Adam(learning_rate=lr),
                loss="sparse_categorical_crossentropy",
                metrics=["accuracy"]
            )
            
            history = model.fit(
                train_gen,
                validation_data=test_gen,
                epochs=3,
                verbose=1
            )
            
            # Log results
            final_val_acc = history.history["val_accuracy"][-1]
            final_val_loss = history.history["val_loss"][-1]
            print(f"[RESULT] Val Accuracy: {final_val_acc:.4f} | Val Loss: {final_val_loss:.4f}")
            
            tuning_results.append({
                "learning_rate": lr,
                "batch_size": batch_size,
                "val_accuracy": float(final_val_acc),
                "val_loss": float(final_val_loss)
            })
            
            # Clear TF backend memory
            tf.keras.backend.clear_session()
            
    # Save tuning results to JSON
    tuning_log_path = OUT_DIR / "tuning_results.json"
    with open(tuning_log_path, "w") as f:
        json.dump(tuning_results, f, indent=4)
    print(f"\n[OK] Hyperparameter tuning complete. Logs saved to: {tuning_log_path}")
    
    # 3. Find the best hyperparameters
    best_config = max(tuning_results, key=lambda x: x["val_accuracy"])
    best_lr = best_config["learning_rate"]
    best_batch_size = best_config["batch_size"]
    
    print("\n====================================================")
    print(f"🏆 BEST CONFIGURATION FOUND:")
    print(f"  - Learning Rate: {best_lr}")
    print(f"  - Batch Size: {best_batch_size}")
    print(f"  - Validation Accuracy: {best_config['val_accuracy']:.4f}")
    print("====================================================")
    
    # 4. Final Deep Training Run with Best Hyperparameters
    print(f"\n--- PHASE 2: RUNNING DEEP CONVERGENCE TRAINING (Up to 15 Epochs) ---")
    print(f"[INFO] Using Learning Rate: {best_lr} | Batch Size: {best_batch_size}")
    
    train_gen = CropSequence(train_crops, y_train, batch_size=best_batch_size, target_size=(224, 224))
    test_gen = CropSequence(test_crops, y_test, batch_size=best_batch_size, target_size=(224, 224))
    
    model, base_model = build_model(num_classes)
    
    # Warm up head
    print("\n[STEP 1] Warming up Classification Head (1 Epoch)...")
    model.compile(
        optimizer=Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    model.fit(train_gen, validation_data=test_gen, epochs=1, verbose=1)
    
    # Fine-tune Top 30 layers
    print("\n[STEP 2] Fine-Tuning Top 30 Layers (Max 15 Epochs with Early Stopping)...")
    base_model.trainable = True
    for layer in base_model.layers[:-30]:
        layer.trainable = False
        
    model.compile(
        optimizer=Adam(learning_rate=best_lr),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    
    # Set up robust academic callbacks
    best_weights_path = OUT_DIR / "best_efficientnet_tuned.h5"
    history_csv_path = OUT_DIR / "training_history_tuned.csv"
    
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
            append=False
        ),
        EarlyStopping(
            monitor="val_loss",
            patience=3,
            restore_best_weights=True,
            verbose=1
        )
    ]
    
    history = model.fit(
        train_gen,
        validation_data=test_gen,
        epochs=15,
        callbacks=callbacks,
        verbose=1
    )
    
    print(f"\n[OK] Deep convergence training complete.")
    print(f"[OK] Best weights saved to: {best_weights_path}")
    print(f"[OK] Detailed epoch logs saved to: {history_csv_path}")
    
    # 5. Save final training plots
    print("\n[INFO] Saving final convergence history plots...")
    history_dict = history.history
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    ax1.plot(history_dict["loss"], label="Train Loss", color="#1f77b4", marker='o')
    ax1.plot(history_dict["val_loss"], label="Val Loss", color="#ff7f0e", marker='s')
    ax1.set_title("CNN Loss convergence")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    ax2.plot(history_dict["accuracy"], label="Train Acc", color="#2ca02c", marker='o')
    ax2.plot(history_dict["val_accuracy"], label="Val Acc", color="#d62728", marker='s')
    ax2.set_title("CNN Accuracy convergence")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    fig.tight_layout()
    fig.savefig(OUT_DIR / "training_plots_tuned.png", dpi=120)
    plt.close(fig)
    print(f"[INFO] Plot saved to: {OUT_DIR / 'training_plots_tuned.png'}")
    
    # Save a quick summary text file for reports
    summary_path = OUT_DIR / "tuning_summary.txt"
    with open(summary_path, "w") as f:
        f.write("=== HYPERPARAMETER TUNING & DEEP TRAINING SUMMARY ===\n")
        f.write(f"Search Space:\n")
        f.write(f"  - Learning Rates evaluated: {learning_rates}\n")
        f.write(f"  - Batch Sizes evaluated: {batch_sizes}\n\n")
        f.write(f"Best Configuration Found:\n")
        f.write(f"  - Learning Rate: {best_lr}\n")
        f.write(f"  - Batch Size: {best_batch_size}\n")
        f.write(f"  - Val Accuracy during sweep: {best_config['val_accuracy']:.4f}\n\n")
        f.write(f"Final Convergence Details:\n")
        f.write(f"  - Total epochs trained: {len(history_dict['loss'])}\n")
        f.write(f"  - Final Train Accuracy: {history_dict['accuracy'][-1]:.4f}\n")
        f.write(f"  - Final Val Accuracy: {history_dict['val_accuracy'][-1]:.4f}\n")
        f.write(f"  - Final Train Loss: {history_dict['loss'][-1]:.4f}\n")
        f.write(f"  - Final Val Loss: {history_dict['val_loss'][-1]:.4f}\n")
        
    print(f"[INFO] Summary report saved to: {summary_path}")

if __name__ == "__main__":
    main()
