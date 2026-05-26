import sys
import time
import pickle
import numpy as np
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score

# Setup paths
SCRIPTS_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPTS_DIR.parent

# Add archive path to import local modules
sys.path.append(str(SCRIPTS_DIR))
sys.path.append(str(SCRIPTS_DIR / "archive"))
from ml_balanced_training import load_crops_and_balance

# Location of cached features
CACHE_DIR = ROOT_DIR / "runs" / "dl" / "convnext_ensemble"
OUT_DIR = ROOT_DIR / "runs" / "dl" / "pca_experiments"
DATA_YAML = ROOT_DIR / "data" / "merged_dataset_v5" / "data.yaml"

class WasteMLP(nn.Module):
    """
    Lightweight MLP optimized for PCA reduced features.
    """
    def __init__(self, input_dim, num_classes=7):
        super(WasteMLP, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            nn.Linear(128, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            
            nn.Linear(32, num_classes)
        )
        
    def forward(self, x):
        return self.net(x)

def train_eval_mlp(x_train_pca, y_train, x_test_pca, y_test, num_classes=7, epochs=15):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Scale
    scaler = StandardScaler()
    x_train_pca = scaler.fit_transform(x_train_pca)
    x_test_pca = scaler.transform(x_test_pca)
    
    # Dataloaders
    train_dataset = TensorDataset(torch.tensor(x_train_pca, dtype=torch.float32), torch.tensor(y_train, dtype=torch.long))
    test_dataset = TensorDataset(torch.tensor(x_test_pca, dtype=torch.float32), torch.tensor(y_test, dtype=torch.long))
    
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
    
    model = WasteMLP(input_dim=x_train_pca.shape[1], num_classes=num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
    
    # Training Loop
    start_time = time.time()
    for epoch in range(epochs):
        model.train()
        for x_batch, y_batch in train_loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            
            optimizer.zero_grad()
            outputs = model(x_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
    train_time = time.time() - start_time
    
    # Evaluation
    model.eval()
    all_preds = []
    all_trues = []
    
    # Measure Latency
    latency_start = time.time()
    with torch.no_grad():
        for x_batch, y_batch in test_loader:
            x_batch = x_batch.to(device)
            outputs = model(x_batch)
            _, predicted = outputs.max(1)
            all_preds.extend(predicted.cpu().numpy())
            all_trues.extend(y_batch.numpy())
    eval_latency = (time.time() - latency_start) / len(y_test) * 1000  # ms per sample
    
    acc = accuracy_score(all_trues, all_preds)
    f1 = f1_score(all_trues, all_preds, average='weighted')
    
    return acc, f1, train_time, eval_latency

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    
    print("====================================================")
    print("PCA Dimensionality Reduction Experiment Suite")
    print("====================================================")
    
    # 1. Load cached handcrafted features
    train_feat_path = CACHE_DIR / "train_handcrafted_637.npy"
    test_feat_path = CACHE_DIR / "test_handcrafted_637.npy"
    
    if not train_feat_path.exists():
        print(f"[ERROR] Required caches not found. Please run train_convnext_ensemble.py first.")
        return
        
    print("[INFO] Loading cached 637-feature vectors...")
    x_train = np.load(train_feat_path)
    x_test = np.load(test_feat_path)
    
    # Dynamically load balanced labels
    print("[INFO] Loading balanced crops to resolve labels...")
    target_classes = ["plastic", "glass", "metal", "paper", "cardboard", "organic", "Background"]
    _, y_train_list = load_crops_and_balance(
        DATA_YAML, target_classes, max_per_class=1000, is_train=True, seed=42
    )
    _, y_test_list = load_crops_and_balance(
        DATA_YAML, target_classes, max_per_class=300, is_train=False, seed=42
    )
    y_train = np.array(y_train_list, dtype=np.int64)
    y_test = np.array(y_test_list, dtype=np.int64)
    
    print(f"  * Loaded Train features: {x_train.shape} | Labels: {y_train.shape}")
    print(f"  * Loaded Test features: {x_test.shape} | Labels: {y_test.shape}")
    
    # We sweep over components count
    pca_dimensions = [32, 64, 128, 256, 512]
    results = []
    
    # Baseline (no PCA - full 637 dims)
    print("\n[EVALUATING] Baseline (Full 637 Handcrafted Dimensions)...")
    acc, f1, t_time, latency = train_eval_mlp(x_train, y_train, x_test, y_test)
    results.append({
        "components": 637,
        "accuracy": acc,
        "f1_score": f1,
        "train_time": t_time,
        "latency_ms": latency
    })
    print(f"  --> Accuracy: {acc*100:.2f}% | Latency: {latency:.4f}ms/sample")
    
    for dims in pca_dimensions:
        if dims >= x_train.shape[1]:
            continue
        print(f"\n[EVALUATING] Fitting PCA with {dims} Components...")
        
        # Fit PCA
        pca_start = time.time()
        pca = PCA(n_components=dims, random_state=42)
        x_train_pca = pca.fit_transform(x_train)
        x_test_pca = pca.transform(x_test)
        pca_time = time.time() - pca_start
        
        explained_variance = np.sum(pca.explained_variance_ratio_) * 100
        print(f"  * Explained Variance: {explained_variance:.2f}% (computed in {pca_time:.2f}s)")
        
        # Train MLP
        acc, f1, t_time, latency = train_eval_mlp(x_train_pca, y_train, x_test_pca, y_test)
        results.append({
            "components": dims,
            "accuracy": acc,
            "f1_score": f1,
            "train_time": t_time,
            "latency_ms": latency,
            "explained_variance": explained_variance
        })
        print(f"  --> Accuracy: {acc*100:.2f}% | Latency: {latency:.4f}ms/sample")
        
    # 2. Save Markdown Report
    report_path = OUT_DIR / "PCA_Dimensionality_Report.md"
    with open(report_path, "w") as f:
        f.write("# PCA Feature-Space Dimensionality Reduction Report\n\n")
        f.write("This report analyzes the impact of Principal Component Analysis (PCA) dimensionality reduction on our 637 handcrafted texture/edge feature classifier.\n\n")
        f.write("| Components Count | Explained Variance (%) | Validation Accuracy (%) | Weighted F1-Score | Inference Latency (ms) | Train Time (s) |\n")
        f.write("| :---: | :---: | :---: | :---: | :---: | :---: |\n")
        for res in results:
            exp_var = f"{res.get('explained_variance', 100.0):.2f}%"
            f.write(f"| **{res['components']}** | {exp_var} | {res['accuracy']*100:.2f}% | {res['f1_score']:.4f} | {res['latency_ms']:.4f} ms | {res['train_time']:.2f} s |\n")
            
    print(f"\n[SUCCESS] PCA Experiment report saved to: {report_path}")
    
    # 3. Save Summary Plot
    plt.figure(figsize=(10, 5))
    comps = [r["components"] for r in results]
    accs = [r["accuracy"] * 100 for r in results]
    lats = [r["latency_ms"] for r in results]
    
    fig, ax1 = plt.subplots(figsize=(8, 4))
    
    color = 'tab:blue'
    ax1.set_xlabel('Components Count (Dimensions)')
    ax1.set_ylabel('Validation Accuracy (%)', color=color)
    ax1.plot(comps, accs, marker='o', color=color, linewidth=2)
    ax1.tick_params(axis='y', labelcolor=color)
    
    ax2 = ax1.twinx()
    color = 'tab:red'
    ax2.set_ylabel('Inference Latency (ms)', color=color)
    ax2.plot(comps, lats, marker='s', color=color, linewidth=2, linestyle='dashed')
    ax2.tick_params(axis='y', labelcolor=color)
    
    plt.title("Dimensionality Count vs. Accuracy and Latency")
    fig.tight_layout()
    plot_path = OUT_DIR / "pca_dimensionality_chart.png"
    plt.savefig(plot_path)
    plt.close()
    print(f"[SUCCESS] PCA Summary plot saved to: {plot_path}")

if __name__ == "__main__":
    main()
