#!/usr/bin/env python
"""
Final Year Project (FYP) - Waste Sorting & Classification System
YOLOv8 Training Script for TACO Object Detection & Classification

This script:
1. Installs/Imports the Ultralytics YOLOv8 library.
2. Loads a pre-trained YOLOv8-Nano (or Small) model.
3. Automatically detects hardware accelerators (CUDA/MPS/CPU).
4. Trains the model on the newly created mapped TACO dataset (taco_yolo/data.yaml).
5. Exports training curves and model weights under 'runs/detect/yolov8_taco/'.
"""

import os
import sys
from pathlib import Path

# Setup paths
ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_YAML = ROOT_DIR / "external_datasets" / "taco_yolo" / "data.yaml"

def check_ultralytics():
    """Verifies if ultralytics is installed in the active environment, prints guide if missing."""
    try:
        import ultralytics
        print(f"[OK] Ultralytics YOLO package is installed. Version: {ultralytics.__version__}")
        return True
    except ImportError:
        print("[WARNING] 'ultralytics' library is not installed in your active virtual environment.")
        print("To install it, please run:")
        print("  pip install ultralytics")
        print("\nWe will attempt to run training once installed.")
        return False

def main():
    print("=====================================================================")
    print("                YOLOv8 Model Training Orchestrator                   ")
    print("=====================================================================")
    
    if not DATA_YAML.exists():
        print(f"[ERROR] YOLO Dataset configuration file not found at: {DATA_YAML}")
        print("Please ensure you have run 'scripts/prepare_taco_yolo.py' first.")
        sys.exit(1)
        
    if not check_ultralytics():
        print("[INFO] Exiting preparation script. Please install 'ultralytics' to proceed.")
        sys.exit(1)
        
    from ultralytics import YOLO
    import torch
    
    # 1. Detect hardware accelerator
    device = "cpu"
    if torch.cuda.is_available():
        device = 0
        print(f"[OK] CUDA GPU detected: {torch.cuda.get_device_name(0)}")
    else:
        print("[INFO] CUDA GPU not found. Training will run on CPU (recommended for testing only).")

    # 2. Load model configuration (yolo11n - Nano for quick test and Edge efficiency)
    model_name = "yolo11n.pt"
    print(f"[INFO] Initializing model with pre-trained weights: {model_name}...")
    model = YOLO(model_name)
    
    # 3. Configure training hyperparameters
    epochs = 50
    batch_size = 16
    img_size = 640
    project_dir = ROOT_DIR / "runs" / "detect"
    name = "yolov11_taco"
    
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
            cache=True
        )
        print("\n=====================================================================")
        print("      [SUCCESS] YOLOv8 Model Training Completed Successfully!        ")
        print("=====================================================================")
        print(f"[INFO] Training weights and logs saved to: runs/detect/{name}/")
        print(f"[INFO] Best model weights: runs/detect/{name}/weights/best.pt")
        print("=====================================================================")
    except Exception as e:
        print(f"\n[ERROR] An error occurred during training: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
