#!/usr/bin/env python
"""
Final Year Project (FYP) - Waste Sorting & Classification System
YOLOv11 Training Script for High-Accuracy Super Dataset

Trains YOLOv11 on the merged super-dataset containing ~24,500 images.
Optimized for NVIDIA GeForce RTX 3060 (12GB VRAM).
"""

import sys
from pathlib import Path

# Setup paths
ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_YAML = ROOT_DIR / "external_datasets" / "super_yolo_dataset" / "data.yaml"

def main():
    print("=====================================================================")
    print("            YOLOv11 SUPER DATASET MODEL TRAINING ORCHESTRATOR        ")
    print("=====================================================================")
    
    if not DATA_YAML.exists():
        print(f"[ERROR] Super YOLO Dataset configuration not found at: {DATA_YAML}")
        print("Please build the super dataset first using 'scripts/build_super_yolo_dataset.py'.")
        sys.exit(1)
        
    from ultralytics import YOLO
    import torch
    
    # 1. Hardware acceleration check
    device = "cpu"
    if torch.cuda.is_available():
        device = 0
        print(f"[OK] CUDA GPU detected: {torch.cuda.get_device_name(0)}")
    else:
        print("[WARNING] CUDA GPU not found! Super dataset is extremely large; training on CPU is not feasible.")
        sys.exit(1)

    # 2. Load model config (yolo11n - Nano for quick training and edge deployment)
    model_name = "yolo11n.pt"
    print(f"[INFO] Initializing model with pre-trained weights: {model_name}...")
    model = YOLO(model_name)
    
    # 3. Configure training hyperparameters
    epochs = 30  # Optimized for super-dataset size
    batch_size = 32  # Optimized for RTX 3060 (12GB VRAM)
    img_size = 640
    project_dir = ROOT_DIR / "runs" / "detect"
    name = "yolov11_super_dataset"
    
    print("\n[INFO] Starting training with the following parameters:")
    print(f"  - Dataset Config: {DATA_YAML}")
    print(f"  - Epochs: {epochs}")
    print(f"  - Batch Size: {batch_size}")
    print(f"  - Image Resolution: {img_size}x{img_size}")
    print(f"  - Device: {device}")
    print(f"  - Output Project: {project_dir}/{name}")
    print("---------------------------------------------------------------------")
    
    try:
        results = model.train(
            data=str(DATA_YAML),
            epochs=epochs,
            batch=batch_size,
            imgsz=img_size,
            device=device,
            project=str(project_dir),
            name=name,
            exist_ok=True,
            val=True,
            cache=True,  # Cache images in RAM for maximum speed
            workers=8
        )
        print("\n=====================================================================")
        print("     [SUCCESS] YOLOv11 Super Model Training Completed Successfully!  ")
        print("=====================================================================")
        print(f"[INFO] Training weights and logs saved to: runs/detect/{name}/")
        print(f"[INFO] Best model weights: runs/detect/{name}/weights/best.pt")
        print("=====================================================================")
    except Exception as e:
        print(f"\n[ERROR] An error occurred during training: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
