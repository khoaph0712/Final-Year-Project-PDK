import os
import sys
import pickle
import numpy as np
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report

# Setup paths
SCRIPTS_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPTS_DIR.parent
sys.path.append(str(SCRIPTS_DIR))
sys.path.append(str(SCRIPTS_DIR / "archive"))

from ml_balanced_training import load_crops_and_balance
from custom_feature_extractor import extract_637_features

ROOT = Path(__file__).resolve().parent.parent
DATA_YAML = ROOT / "data" / "merged_dataset_v5" / "data.yaml"
OUT_DIR = ROOT / "runs" / "dl" / "cross_dataset_validation"

class WasteMLP(nn.Module):
    def __init__(self, input_dim=637, num_classes=7):
        super(WasteMLP, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            nn.Linear(256, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            
            nn.Linear(64, num_classes)
        )
        
    def forward(self, x):
        return self.net(x)

def is_trashnet_file(filename):
    """
    Returns True if the crop filename belongs to the standard TrashNet dataset.
    TrashNet naming patterns:
    - contains 'white-glass', 'brown-glass', 'green-glass'
    - contains cardboard, glass, metal, paper, plastic, trash followed immediately by numbers
    """
    fn = filename.lower()
    # TrashNet Glass
    if 'glass' in fn and ('white' in fn or 'brown' in fn or 'green' in fn):
        return True
    # Other TrashNet categories (look for material name + digits)
    materials = ['paper', 'cardboard', 'plastic', 'metal', 'trash']
    for mat in materials:
        if mat in fn:
            # Check if there are digits in the filename, and not containing 'train' or 'rf'
            if any(char.isdigit() for char in fn) and '_train_' not in fn and 'rf_' not in fn:
                return True
    return False

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    
    print("====================================================")
    print("Cross-Dataset Domain Generalizability Validation")
    print("====================================================")
    
    target_classes = ["plastic", "glass", "metal", "paper", "cardboard", "organic", "Background"]
    num_classes = len(target_classes)
    
    # 1. Load balanced train and test crops
    print("[INFO] Loading balanced crops from splits...")
    train_crops, y_train_list = load_crops_and_balance(
        DATA_YAML, target_classes, max_per_class=2000, is_train=True, seed=42
    )
    test_crops, y_test_list = load_crops_and_balance(
        DATA_YAML, target_classes, max_per_class=600, is_train=False, seed=42
    )
    
    # We also need to map the filenames/source indices of balanced crops to split them.
    # Since load_crops_and_balance only returns BGR image arrays, let's reload them with filenames!
    # Let's inspect the files in training subfolders and extract features dynamically or load.
    # To be extremely efficient and accurate, let's load all crops and check their sources!
    
    print("\n[INFO] Separating Domain A (TrashNet) and Domain B (TACO) crops...")
    
    x_train_trashnet, y_train_trashnet = [], []
    x_test_taco, y_test_taco = [], []
    
    # Locate all crops in directory directly and check their filenames to build the splits
    train_root = DATA_YAML.parent / "train"
    test_root = DATA_YAML.parent / "test"
    
    # Extract features for TrashNet train set
    print("[INFO] Extracting features for Domain A (TrashNet Training Set)...")
    for cls_idx, cls_name in enumerate(target_classes):
        cls_dir = train_root / cls_name
        if not cls_dir.exists():
            continue
        files = list(cls_dir.glob("*.jpg")) + list(cls_dir.glob("*.png"))
        
        count = 0
        for f in files:
            if is_trashnet_file(f.name):
                img = cv2_read_img(f)
                if img is not None:
                    x_train_trashnet.append(extract_637_features(img))
                    y_train_trashnet.append(cls_idx)
                    count += 1
                    if count >= 800: # Limit to 800 per class to balance
                        break
        print(f"  * {cls_name.upper()}: Loaded {count} TrashNet crops")
        
    # Extract features for TACO test set
    print("\n[INFO] Extracting features for Domain B (TACO Test Set)...")
    for cls_idx, cls_name in enumerate(target_classes):
        cls_dir = test_root / cls_name
        if not cls_dir.exists():
            continue
        files = list(cls_dir.glob("*.jpg")) + list(cls_dir.glob("*.png"))
        
        count = 0
        for f in files:
            # If not trashnet, it belongs to TACO/Roboflow (Domain B)
            if not is_trashnet_file(f.name):
                img = cv2_read_img(f)
                if img is not None:
                    x_test_taco.append(extract_637_features(img))
                    y_test_taco.append(cls_idx)
                    count += 1
                    if count >= 200: # Limit to 200 per class for test
                        break
        print(f"  * {cls_name.upper()}: Loaded {count} TACO crops")
        
    x_train_trashnet = np.array(x_train_trashnet, dtype=np.float32)
    y_train_trashnet = np.array(y_train_trashnet, dtype=np.int64)
    x_test_taco = np.array(x_test_taco, dtype=np.float32)
    y_test_taco = np.array(y_test_taco, dtype=np.int64)
    
    print(f"\n[OK] Domain Splits Created:")
    print(f"  * Domain A (Train on TrashNet): {x_train_trashnet.shape}")
    print(f"  * Domain B (Test on TACO): {x_test_taco.shape}")
    
    if len(x_train_trashnet) == 0 or len(x_test_taco) == 0:
        print("[ERROR] Extracted splits are empty. Verify dataset naming conventions.")
        return
        
    # 2. Scale
    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train_trashnet)
    x_test_scaled = scaler.transform(x_test_taco)
    
    # 3. PyTorch Dataloaders
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using training device: {device}")
    
    train_dataset = TensorDataset(torch.tensor(x_train_scaled, dtype=torch.float32), torch.tensor(y_train_trashnet, dtype=torch.long))
    test_dataset = TensorDataset(torch.tensor(x_test_scaled, dtype=torch.float32), torch.tensor(y_test_taco, dtype=torch.long))
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    # 4. Train Model strictly on Domain A (TrashNet)
    model = WasteMLP(input_dim=637, num_classes=num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
    
    print("\n[INFO] Training MLP model strictly on Domain A (TrashNet)...")
    for epoch in range(1, 16):
        model.train()
        running_loss = 0.0
        for x_batch, y_batch in train_loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            
            optimizer.zero_grad()
            outputs = model(x_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * x_batch.size(0)
            
        epoch_loss = running_loss / len(x_train_trashnet)
        print(f"  * Epoch {epoch:02d}/15 | Loss: {epoch_loss:.4f}")
        
    # 5. Evaluate strictly on Domain B (TACO)
    model.eval()
    all_preds = []
    all_trues = []
    with torch.no_grad():
        for x_batch, y_batch in test_loader:
            x_batch = x_batch.to(device)
            outputs = model(x_batch)
            _, predicted = outputs.max(1)
            all_preds.extend(predicted.cpu().numpy())
            all_trues.extend(y_batch.numpy())
            
    acc = accuracy_score(all_trues, all_preds)
    print("\n====================================================")
    print(f"[SUCCESS] CROSS-DATASET GENERALIZATION RESULTS:")
    print(f"  - Trained on: TrashNet (Domain A - Laboratory Bins)")
    print(f"  - Tested on: TACO (Domain B - Real-world Outdoor)")
    print(f"  - Generalization Accuracy: {acc*100:.2f}%")
    print("====================================================")
    
    # Save Report
    report_path = OUT_DIR / "Cross_Dataset_Report.md"
    with open(report_path, "w") as f:
        f.write("# Cross-Dataset Generalizability Validation Report\n\n")
        f.write("This report details the domain shift evaluation to prove generalizability. We trained strictly on TrashNet and tested on complex TACO outdoor samples.\n\n")
        f.write(f"*   **Training Domain A (TrashNet):** {x_train_trashnet.shape[0]} samples\n")
        f.write(f"*   **Testing Domain B (TACO):** {x_test_taco.shape[0]} samples\n")
        f.write(f"*   **Domain-Shift Accuracy:** **{acc*100:.2f}%**\n\n")
        f.write("### Detailed Classification Report:\n")
        f.write(f"```text\n{classification_report(all_trues, all_preds, target_names=target_classes)}\n```\n")
        
    print(f"\n[SUCCESS] Cross-dataset evaluation report saved to: {report_path}")

def cv2_read_img(path):
    import cv2
    return cv2.imread(str(path))

if __name__ == "__main__":
    main()
