import sys
import os
import json
import time
import pickle
from pathlib import Path
import numpy as np
import cv2

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.models as models
import torchvision.transforms as transforms
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score

# Add scripts directory and its archive directory to path to import local modules
SCRIPTS_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPTS_DIR.parent
sys.path.append(str(SCRIPTS_DIR))
sys.path.append(str(SCRIPTS_DIR / "archive"))

from ml_balanced_training import load_crops_and_balance
from custom_feature_extractor import extract_637_features

DATA_YAML = ROOT_DIR / "data" / "merged_dataset_v5" / "data.yaml"
OUT_DIR = ROOT_DIR / "runs" / "dl" / "convnext_ensemble_tuned"

class ConvNeXtFeatureExtractor(nn.Module):
    """
    ConvNeXt-Tiny Feature Extractor with progressive unfreezing support.
    """
    def __init__(self, unfreeze_final=False):
        super(ConvNeXtFeatureExtractor, self).__init__()
        self.backbone = models.convnext_tiny(weights=models.ConvNeXt_Tiny_Weights.DEFAULT)
        self.backbone.classifier = nn.Identity()
        
        # 1. Freeze all parameters initially
        for param in self.backbone.parameters():
            param.requires_grad = False
            
        if unfreeze_final:
            # 2. Unfreeze the final Stage 3 block (features[7]) to allow fine-tuning
            print("[INFO] Unfreezing ConvNeXt final convolutional stage (features[7])...")
            for param in self.backbone.features[7].parameters():
                param.requires_grad = True
                
    def forward(self, x):
        # Do NOT use torch.no_grad() if we are fine-tuning the backbone!
        features = self.backbone(x)
        return torch.flatten(features, 1)

class Stage3EnsembleClassifier(nn.Module):
    def __init__(self, num_classes=7, dropout_rate=0.3, unfreeze_final=False):
        super(Stage3EnsembleClassifier, self).__init__()
        self.convnext_extractor = ConvNeXtFeatureExtractor(unfreeze_final=unfreeze_final)
        self.input_dim = 768 + 637
        
        self.classifier = nn.Sequential(
            nn.Linear(self.input_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate),
            
            nn.Linear(256, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            
            nn.Linear(64, num_classes)
        )
        
    def forward(self, image_tensor, handcrafted_features_tensor):
        deep_features = self.convnext_extractor(image_tensor)
        fused_features = torch.cat((deep_features, handcrafted_features_tensor), dim=1)
        logits = self.classifier(fused_features)
        return logits

class HybridWasteDataset(Dataset):
    def __init__(self, crops, handcrafted_features, labels, transform=None):
        self.crops = crops
        self.handcrafted_features = handcrafted_features
        self.labels = labels
        self.transform = transform
        
    def __len__(self):
        return len(self.crops)
        
    def __getitem__(self, idx):
        crop = self.crops[idx]
        resized = cv2.resize(crop, (224, 224), interpolation=cv2.INTER_CUBIC)
        resized_rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        
        from PIL import Image
        img_pil = Image.fromarray(resized_rgb)
        
        if self.transform:
            image_tensor = self.transform(img_pil)
        else:
            image_tensor = transforms.ToTensor()(img_pil)
            
        handcrafted_tensor = torch.tensor(self.handcrafted_features[idx], dtype=torch.float32)
        label_tensor = torch.tensor(self.labels[idx], dtype=torch.long)
        
        return image_tensor, handcrafted_tensor, label_tensor

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    
    print("====================================================")
    print("Stage 3: ConvNeXt + Handcrafted Progressive Tuning")
    print("====================================================")
    
    target_classes = ["plastic", "glass", "metal", "paper", "cardboard", "organic", "Background"]
    num_classes = len(target_classes)
    
    # 1. Load balanced crops
    print("[INFO] Loading balanced crops from splits...")
    train_crops, y_train_list = load_crops_and_balance(
        DATA_YAML, target_classes, max_per_class=1000, is_train=True, seed=42
    )
    test_crops, y_test_list = load_crops_and_balance(
        DATA_YAML, target_classes, max_per_class=300, is_train=False, seed=42
    )
    
    # 2. Load cached handcrafted features
    cache_dir = ROOT_DIR / "runs" / "dl" / "convnext_ensemble"
    cache_train_features = cache_dir / "train_handcrafted_637.npy"
    cache_test_features = cache_dir / "test_handcrafted_637.npy"
    
    if not cache_train_features.exists():
        print("[ERROR] Required caches not found. Run train_convnext_ensemble.py first.")
        return
        
    print("[INFO] Loading cached 637-feature vectors...")
    x_train_handcrafted = np.load(cache_train_features)
    x_test_handcrafted = np.load(cache_test_features)
    
    scaler = StandardScaler()
    x_train_handcrafted = scaler.fit_transform(x_train_handcrafted)
    x_test_handcrafted = scaler.transform(x_test_handcrafted)
    
    # DataLoaders
    preprocess = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    train_dataset = HybridWasteDataset(train_crops, x_train_handcrafted, y_train_list, transform=preprocess)
    test_dataset = HybridWasteDataset(test_crops, x_test_handcrafted, y_test_list, transform=preprocess)
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using training device: {device}")
    
    # ----------------------------------------------------
    # PHASE 1: Classification Head Warmup (Backbone Frozen)
    # ----------------------------------------------------
    print("\n--- PHASE 1: CLASSIFICATION HEAD WARMUP (3 Epochs) ---")
    model = Stage3EnsembleClassifier(num_classes=num_classes, unfreeze_final=False)
    model.to(device)
    
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.AdamW(model.classifier.parameters(), lr=1e-3, weight_decay=0.01)
    
    for epoch in range(1, 4):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for images, handcrafted, labels in train_loader:
            images = images.to(device)
            handcrafted = handcrafted.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images, handcrafted)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
        epoch_loss = running_loss / total
        epoch_acc = correct / total
        print(f"  * Warmup Epoch {epoch:02d}/03 | Loss: {epoch_loss:.4f} | Acc: {epoch_acc*100:.2f}%")
        
    # Save warmed-up classifier state
    torch.save(model.state_dict(), OUT_DIR / "warmed_up_model.pth")
    
    # ----------------------------------------------------
    # PHASE 2: Progressive Fine-Tuning (Unfreeze features[7])
    # ----------------------------------------------------
    print("\n--- PHASE 2: PROGRESSIVE FINE-TUNING (7 Epochs) ---")
    
    # Reinitialize model with unfreeze_final = True, and load warmed-up state
    model_tuned = Stage3EnsembleClassifier(num_classes=num_classes, unfreeze_final=True)
    model_tuned.load_state_dict(torch.load(OUT_DIR / "warmed_up_model.pth"))
    model_tuned.to(device)
    
    # Configure optimizer with differential learning rates:
    # Very low learning rate for ConvNeXt backbone (1e-5) to avoid feature disruption,
    # Standard learning rate for MLP classifier head (1e-4) to refine predictions.
    optimizer_tuned = optim.AdamW([
        {'params': model_tuned.convnext_extractor.backbone.features[7].parameters(), 'lr': 1e-5},
        {'params': model_tuned.classifier.parameters(), 'lr': 1e-4}
    ], weight_decay=0.05)
    
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer_tuned, T_max=7)
    
    best_acc = 0.0
    for epoch in range(1, 8):
        model_tuned.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for images, handcrafted, labels in train_loader:
            images = images.to(device)
            handcrafted = handcrafted.to(device)
            labels = labels.to(device)
            
            optimizer_tuned.zero_grad()
            outputs = model_tuned(images, handcrafted)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer_tuned.step()
            
            running_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
        scheduler.step()
        
        epoch_loss = running_loss / total
        epoch_acc = correct / total
        
        # Validation
        model_tuned.eval()
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for images, handcrafted, labels in test_loader:
                images = images.to(device)
                handcrafted = handcrafted.to(device)
                labels = labels.to(device)
                
                outputs = model_tuned(images, handcrafted)
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()
                
        val_acc = val_correct / val_total
        print(f"  * Fine-Tuning Epoch {epoch:02d}/07 | Train Loss: {epoch_loss:.4f} | Train Acc: {epoch_acc*100:.2f}% | Val Acc: {val_acc*100:.2f}%")
        
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model_tuned.state_dict(), OUT_DIR / "best_convnext_ensemble_tuned.pth")
            
    print("\n====================================================")
    print(f"[SUCCESS] Progressive unfreezing completed successfully!")
    print(f"  - Best Tuned Validation Accuracy: {best_acc*100:.2f}%")
    print(f"  - Model Saved to: {OUT_DIR / 'best_convnext_ensemble_tuned.pth'}")
    print("====================================================")

if __name__ == "__main__":
    main()
