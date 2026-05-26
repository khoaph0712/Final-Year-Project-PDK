import sys
import os
import json
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
from sklearn.metrics import accuracy_score, f1_score, classification_report

# Add scripts directory and its archive directory to path to import local modules
sys.path.append(str(Path(__file__).resolve().parent))
sys.path.append(str(Path(__file__).resolve().parent / "archive"))

from ml_balanced_training import load_crops_and_balance
from custom_feature_extractor import extract_637_features

ROOT = Path(__file__).resolve().parent.parent
DATA_YAML = ROOT / "data" / "merged_dataset_v5" / "data.yaml"
OUT_DIR = ROOT / "runs" / "dl" / "convnext_ensemble"

class ConvNeXtFeatureExtractor(nn.Module):
    """
    Extracts frozen 768-dimensional spatial representations from ConvNeXt-Tiny.
    """
    def __init__(self):
        super(ConvNeXtFeatureExtractor, self).__init__()
        # Load ConvNeXt-Tiny pre-trained on ImageNet-1k
        self.backbone = models.convnext_tiny(weights=models.ConvNeXt_Tiny_Weights.DEFAULT)
        # Remove the classification head, leaving the global average pooling output (768 dimensions)
        self.backbone.classifier = nn.Identity()
        
        # Freeze backbone parameters to adhere to Green AI guidelines
        for param in self.backbone.parameters():
            param.requires_grad = False
            
    def forward(self, x):
        with torch.no_grad():
            features = self.backbone(x)
        # Flatten the spatial dimensions from [Batch, 768, 1, 1] to [Batch, 768]
        return torch.flatten(features, 1)

class Stage3EnsembleClassifier(nn.Module):
    """
    Ensemble Classifier combining 768 ConvNeXt deep spatial features 
    with 637 handcrafted texture/edge features.
    Total input dimension: 768 + 637 = 1405.
    """
    def __init__(self, num_classes=7, dropout_rate=0.3):
        super(Stage3EnsembleClassifier, self).__init__()
        self.convnext_extractor = ConvNeXtFeatureExtractor()
        self.input_dim = 768 + 637
        
        # Multi-Layer Perceptron (MLP) for classification
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
        # Extract deep features (frozen)
        deep_features = self.convnext_extractor(image_tensor)
        # Concatenate with normalized handcrafted features
        fused_features = torch.cat((deep_features, handcrafted_features_tensor), dim=1)
        # Classify
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
        # Resize to 224x224 and convert to RGB for ConvNeXt
        resized = cv2.resize(crop, (224, 224), interpolation=cv2.INTER_CUBIC)
        resized_rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        
        # Apply transformation (e.g., standard ConvNeXt normalization)
        if self.transform:
            image_tensor = self.transform(Image_from_array(resized_rgb))
        else:
            image_tensor = transforms.ToTensor()(resized_rgb)
            
        handcrafted_tensor = torch.tensor(self.handcrafted_features[idx], dtype=torch.float32)
        label_tensor = torch.tensor(self.labels[idx], dtype=torch.long)
        
        return image_tensor, handcrafted_tensor, label_tensor

def Image_from_array(arr):
    # Quick helper to transform numpy to PIL Image for torchvision transforms
    from PIL import Image
    return Image.fromarray(arr)

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    
    print("====================================================")
    print("Stage 3: ConvNeXt + Handcrafted Ensemble Training")
    print("====================================================")
    
    target_classes = ["plastic", "glass", "metal", "paper", "cardboard", "organic", "Background"]
    num_classes = len(target_classes)
    
    # 1. Load balanced crops from dataset
    print("[INFO] Loading balanced crops from splits...")
    train_crops, y_train_list = load_crops_and_balance(
        DATA_YAML, target_classes, max_per_class=1000, is_train=True, seed=42
    )
    test_crops, y_test_list = load_crops_and_balance(
        DATA_YAML, target_classes, max_per_class=300, is_train=False, seed=42
    )
    
    # 2. Handcrafted features extraction / caching
    cache_train_features = OUT_DIR / "train_handcrafted_637.npy"
    cache_test_features = OUT_DIR / "test_handcrafted_637.npy"
    
    if cache_train_features.exists() and cache_test_features.exists():
        print("[INFO] Loading cached 637-feature vectors from disk...")
        x_train_handcrafted = np.load(cache_train_features)
        x_test_handcrafted = np.load(cache_test_features)
    else:
        print("[INFO] Extracting custom 637-features for train split...")
        x_train_handcrafted = []
        for idx, crop in enumerate(train_crops):
            if idx > 0 and idx % 500 == 0:
                print(f"  - Train features: {idx}/{len(train_crops)} extracted")
            x_train_handcrafted.append(extract_637_features(crop))
        x_train_handcrafted = np.array(x_train_handcrafted, dtype=np.float32)
        np.save(cache_train_features, x_train_handcrafted)
        
        print("[INFO] Extracting custom 637-features for test split...")
        x_test_handcrafted = []
        for idx, crop in enumerate(test_crops):
            if idx > 0 and idx % 200 == 0:
                print(f"  - Test features: {idx}/{len(test_crops)} extracted")
            x_test_handcrafted.append(extract_637_features(crop))
        x_test_handcrafted = np.array(x_test_handcrafted, dtype=np.float32)
        np.save(cache_test_features, x_test_handcrafted)
        
    # Scale the handcrafted features
    scaler = StandardScaler()
    x_train_handcrafted = scaler.fit_transform(x_train_handcrafted)
    x_test_handcrafted = scaler.transform(x_test_handcrafted)
    
    # Save the scaler for inference deployment
    import pickle
    with open(OUT_DIR / "handcrafted_scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)
        
    # 3. Setup PyTorch DataLoaders
    # ConvNeXt preprocessing transform
    preprocess = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    train_dataset = HybridWasteDataset(train_crops, x_train_handcrafted, y_train_list, transform=preprocess)
    test_dataset = HybridWasteDataset(test_crops, x_test_handcrafted, y_test_list, transform=preprocess)
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    # 4. Model, Loss, Optimizer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using training device: {device}")
    
    model = Stage3EnsembleClassifier(num_classes=num_classes)
    model.to(device)
    
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1) # 0.1 label smoothing as recommended
    optimizer = optim.AdamW(model.classifier.parameters(), lr=1e-3, weight_decay=0.05)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)
    
    # 5. Training Loop
    best_acc = 0.0
    epochs = 10
    
    print("\n[INFO] Starting Ensemble Training...")
    for epoch in range(1, epochs + 1):
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
            
        scheduler.step()
        
        epoch_loss = running_loss / total
        epoch_acc = correct / total
        
        # Validation
        model.eval()
        val_correct = 0
        val_total = 0
        val_preds = []
        val_trues = []
        
        with torch.no_grad():
            for images, handcrafted, labels in test_loader:
                images = images.to(device)
                handcrafted = handcrafted.to(device)
                labels = labels.to(device)
                
                outputs = model(images, handcrafted)
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()
                
                val_preds.extend(predicted.cpu().numpy())
                val_trues.extend(labels.cpu().numpy())
                
        val_acc = val_correct / val_total
        print(f"Epoch {epoch:02d}/{epochs:02d} | Train Loss: {epoch_loss:.4f} | Train Acc: {epoch_acc*100:.2f}% | Val Acc: {val_acc*100:.2f}%")
        
        # Save best model
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), OUT_DIR / "best_convnext_ensemble.pth")
            
    print("\n====================================================")
    print(f"[SUCCESS] Training Complete. Best Validation Accuracy: {best_acc*100:.2f}%")
    print(f"Model saved to: {OUT_DIR / 'best_convnext_ensemble.pth'}")
    print("====================================================")

if __name__ == "__main__":
    main()
