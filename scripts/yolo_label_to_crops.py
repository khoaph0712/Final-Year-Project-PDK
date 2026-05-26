import os
from pathlib import Path
import cv2
import yaml
import numpy as np

def extract_crops_from_yolo(data_yaml_path, output_dir, margin_factor=0.10, target_size=(224, 224)):
    """
    Converts YOLOv11 bounding box annotations into dynamic padded high-resolution crops
    tailored specifically for ConvNeXt classifier training.
    
    Args:
        data_yaml_path (str or Path): Path to the dataset data.yaml file.
        output_dir (str or Path): Centralized folder to save the generated crops.
        margin_factor (float): Dynamic halo margin around the bounding box (default: 10%).
        target_size (tuple): Output dimension for resized crops.
    """
    data_yaml_path = Path(data_yaml_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Parse dataset configuration yaml
    with open(data_yaml_path, 'r') as f:
        data_cfg = yaml.safe_load(f)
        
    classes = data_cfg['names']
    print(f"[INFO] Dataset classes: {classes}")
    
    # Create subfolders for each class in the output directory
    for cls in classes:
        (output_dir / cls).mkdir(parents=True, exist_ok=True)
        
    dataset_root = data_yaml_path.parent
    train_images_dir = dataset_root / "images" / "train"
    train_labels_dir = dataset_root / "labels" / "train"
    
    if not train_images_dir.exists():
        # Fallback to direct path search if standard YOLO structure differs
        print(f"[WARNING] YOLO images train dir not found at {train_images_dir}. Searching under root...")
        return
        
    image_paths = list(train_images_dir.glob("*.jpg")) + list(train_images_dir.glob("*.png"))
    print(f"[INFO] Found {len(image_paths)} images to process...")
    
    crop_count = 0
    for img_path in image_paths:
        # Load corresponding label file
        label_path = train_labels_dir / f"{img_path.stem}.txt"
        if not label_path.exists():
            continue
            
        # Load image
        img = cv2.imread(str(img_path))
        if img is None:
            continue
            
        h_img, w_img = img.shape[:2]
        
        # Read YOLO labels
        with open(label_path, 'r') as lf:
            lines = lf.readlines()
            
        for line_idx, line in enumerate(lines):
            parts = line.strip().split()
            if len(parts) < 5:
                continue
                
            cls_idx = int(parts[0])
            x_c, y_c, w_b, h_b = map(float, parts[1:5])
            
            # Map normalized coordinates to absolute pixel space
            w_abs = w_b * w_img
            h_abs = h_b * h_img
            x_min = (x_c - w_b / 2) * w_img
            y_min = (y_c - h_b / 2) * h_img
            
            # Calculate padded margin (10% halo to capture light refraction edges for Glass/Plastic)
            pad_x = w_abs * margin_factor
            pad_y = h_abs * margin_factor
            
            x1 = int(max(0, x_min - pad_x))
            y1 = int(max(0, y_min - pad_y))
            x2 = int(min(w_img, x_min + w_abs + pad_x))
            y2 = int(min(h_img, y_min + h_abs + pad_y))
            
            # Extract component crop
            crop = img[y1:y2, x1:x2]
            if crop.size == 0:
                continue
                
            # Resize using Bicubic interpolation to preserve edge frequencies
            crop_resized = cv2.resize(crop, target_size, interpolation=cv2.INTER_CUBIC)
            
            # Save crop to class subfolder
            cls_name = classes[cls_idx]
            crop_filename = f"{img_path.stem}_crop_{line_idx}.jpg"
            save_path = output_dir / cls_name / crop_filename
            
            cv2.imwrite(str(save_path), crop_resized)
            crop_count += 1
            
    print(f"[SUCCESS] Extracted and saved {crop_count} high-fidelity crops to: {output_dir}")

if __name__ == "__main__":
    ROOT = Path(__file__).resolve().parent.parent
    DATA_YAML_PATH = ROOT / "data" / "merged_dataset_v5" / "data.yaml"
    OUTPUT_CROPS_DIR = ROOT / "data" / "convnext_training_crops"
    
    print("====================================================")
    print("YOLOv11 to ConvNeXt Crop Conversion Tool")
    print("====================================================")
    
    if DATA_YAML_PATH.exists():
        extract_crops_from_yolo(DATA_YAML_PATH, OUTPUT_CROPS_DIR)
    else:
        print(f"[ERROR] Could not locate data.yaml at {DATA_YAML_PATH}. Please verify your paths.")
