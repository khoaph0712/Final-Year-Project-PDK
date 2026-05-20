"""Balanced Feature ML Training and Evaluation Pipeline for Waste Management.

Academic Task 1, 2, and 3:
1) Perform statistical analysis of raw class distribution.
2) Add negative background samples as a specific "Background" class.
3) Balance dataset via Undersampling (majority classes) and Oversampling/Augmentation (minority classes).
4) Extract exactly 637 custom handcrafted features.
5) Train and compare 4 algorithms: Decision Tree, SVM, Random Forest, and XGBoost.
6) Generate detailed evaluation reports, confusion matrices, and feature importance analyses.
"""

from __future__ import annotations
import argparse
import csv
import json
import random
from pathlib import Path
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml

from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    f1_score,
)

# Optional XGBoost import
try:
    from xgboost import XGBClassifier
except ImportError:
    XGBClassifier = None

# Import our custom 637-feature extractor
from custom_feature_extractor import extract_637_features

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATA = ROOT / "merged_dataset_v3" / "data.yaml"
DEFAULT_OUT = ROOT / "runs" / "ml" / "balanced_637_ml"


def analyze_raw_class_distribution(data_yaml_path: Path) -> dict:
    """Analyze the statistical class distribution across dataset splits."""
    if not data_yaml_path.exists():
        print(f"[WARN] data.yaml not found at {data_yaml_path}. Skipping statistical report.")
        return {}

    cfg = yaml.safe_load(data_yaml_path.read_text(encoding="utf-8"))
    class_names = list(cfg["names"])
    ds_root = Path(cfg.get("path", data_yaml_path.parent))
    
    splits = ["train", "val", "test"]
    stats = {split: {name: 0 for name in class_names} for split in splits}
    
    for split in splits:
        split_key = "val" if split == "val" and "val" in cfg else ("test" if split == "test" and "test" in cfg else split)
        if split_key not in cfg:
            continue
            
        img_rel = cfg[split_key]
        img_dir = ds_root / img_rel
        if not img_dir.exists():
            # Fallback relative to data.yaml
            img_dir = data_yaml_path.parent / img_rel
            
        if not img_dir.exists():
            continue
            
        # Scan label directory
        lbl_dir = Path(str(img_dir).replace("/images", "/labels").replace("\\images", "\\labels"))
        if not lbl_dir.exists():
            continue
            
        for lbl_path in lbl_dir.glob("*.txt"):
            try:
                for line in lbl_path.read_text(encoding="utf-8").splitlines():
                    parts = line.strip().split()
                    if len(parts) >= 1:
                        cid = int(float(parts[0]))
                        if 0 <= cid < len(class_names):
                            stats[split][class_names[cid]] += 1
            except Exception:
                pass
                
    return stats


def load_background_images(seed: int = 42) -> list[Path]:
    """Scan the workspace for negative/background images."""
    bg_paths = []
    
    # 1. GINI backgrounds
    gini_dir = ROOT / "prepared_datasets" / "gini_binary"
    if gini_dir.exists():
        bg_paths.extend(list(gini_dir.rglob("background__*")))
        
    # 2. Main dataset empty labels
    merged_dir = ROOT / "merged_dataset_v3"
    if merged_dir.exists():
        for split in ("train", "valid", "test"):
            img_dir = merged_dir / split / "images"
            if img_dir.exists():
                for img_path in img_dir.glob("*"):
                    if img_path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}:
                        lbl_path = Path(str(img_path).replace("\\images\\", "\\labels\\").replace("/images/", "/labels/")).with_suffix(".txt")
                        if not lbl_path.exists() or lbl_path.stat().st_size == 0:
                            bg_paths.append(img_path)
                            
    random.Random(seed).shuffle(bg_paths)
    return bg_paths


def extract_background_crops(bg_paths: list[Path], num_needed: int, max_per_image: int = 15, seed: int = 42) -> list[np.ndarray]:
    """Extract random square patches from negative/background images to represent Background class."""
    crops = []
    rng = random.Random(seed)
    
    if bg_paths:
        for path in bg_paths:
            if len(crops) >= num_needed:
                break
            img = cv2.imread(str(path))
            if img is None:
                continue
            h, w = img.shape[:2]
            
            num_crops = min(max_per_image, num_needed - len(crops))
            for _ in range(num_crops):
                # Crop size between 64 and min(h, w, 256)
                c_size = rng.randint(64, min(h, w, 256))
                x = rng.randint(0, w - c_size)
                y = rng.randint(0, h - c_size)
                crop = img[y : y + c_size, x : x + c_size]
                crops.append(crop)
                
    # Fallback synthetic textures if insufficient real background images are found
    if len(crops) < num_needed:
        shortfall = num_needed - len(crops)
        print(f"[INFO] Generating {shortfall} synthetic background textures (grass/carpet/concrete) to meet balancing target.")
        for _ in range(shortfall):
            bg_type = rng.choice(["grass", "carpet", "concrete"])
            crop = np.zeros((128, 128, 3), dtype=np.uint8)
            if bg_type == "grass":
                crop[:, :, 1] = rng.randint(110, 180) # Green
                crop[:, :, 0] = rng.randint(30, 80)   # Blue
                crop[:, :, 2] = rng.randint(30, 80)   # Red
                noise = np.random.randint(-25, 25, (128, 128, 3)).astype(np.int16)
                crop = np.clip(crop.astype(np.int16) + noise, 0, 255).astype(np.uint8)
            elif bg_type == "carpet":
                crop[:, :, 0] = 170 # Beige (Blue)
                crop[:, :, 1] = 190 # Green
                crop[:, :, 2] = 210 # Red
                noise = np.random.randint(-15, 15, (128, 128, 3)).astype(np.int16)
                crop = np.clip(crop.astype(np.int16) + noise, 0, 255).astype(np.uint8)
            else:
                val = rng.randint(90, 150)
                crop[:, :, :] = val # Grey Concrete
                noise = np.random.randint(-12, 12, (128, 128, 3)).astype(np.int16)
                crop = np.clip(crop.astype(np.int16) + noise, 0, 255).astype(np.uint8)
            crops.append(crop)
            
    return crops


def augment_crop(crop: np.ndarray, rng: random.Random) -> np.ndarray:
    """Apply minor geometric and photometric image augmentation for minority oversampling."""
    augmented = crop.copy()
    if rng.random() > 0.5:
        augmented = cv2.flip(augmented, 1) # Horizontal flip
    if rng.random() > 0.5:
        augmented = cv2.flip(augmented, 0) # Vertical flip
        
    if rng.random() > 0.5:
        angle = rng.uniform(-15, 15)
        h, w = augmented.shape[:2]
        M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
        augmented = cv2.warpAffine(augmented, M, (w, h), borderMode=cv2.BORDER_REPLICATE)
        
    if rng.random() > 0.5:
        factor = rng.uniform(0.85, 1.15)
        augmented = np.clip(augmented * factor, 0, 255).astype(np.uint8)
        
    return augmented


def load_crops_and_balance(
    data_yaml_path: Path,
    target_classes: list[str],
    max_per_class: int,
    is_train: bool = True,
    seed: int = 42
) -> tuple[list[np.ndarray], list[int]]:
    """Load crops from labels, balance via undersampling/oversampling augmentation, and return crops + labels."""
    cfg = yaml.safe_load(data_yaml_path.read_text(encoding="utf-8"))
    class_names = list(cfg["names"])
    ds_root = Path(cfg.get("path", data_yaml_path.parent))
    
    split = "train" if is_train else "test"
    split_key = "train" if is_train else ("test" if "test" in cfg else "val")
    img_rel = cfg[split_key]
    img_dir = ds_root / img_rel
    if not img_dir.exists():
        img_dir = data_yaml_path.parent / img_rel
        
    lbl_dir = Path(str(img_dir).replace("/images", "/labels").replace("\\images", "\\labels"))
    
    # Bucket to collect crops per target class
    class_crops = {name: [] for name in target_classes}
    rng = random.Random(seed)
    
    # Load all real crops
    if img_dir.exists() and lbl_dir.exists():
        img_paths = list(img_dir.glob("*"))
        rng.shuffle(img_paths)
        
        for img_path in img_paths:
            if img_path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}:
                continue
            lbl_path = lbl_dir / img_path.with_suffix(".txt").name
            if not lbl_path.exists():
                continue
                
            img = cv2.imread(str(img_path))
            if img is None:
                continue
            h, w = img.shape[:2]
            
            try:
                for line in lbl_path.read_text(encoding="utf-8").splitlines():
                    parts = line.strip().split()
                    if len(parts) != 5:
                        continue
                    cid = int(float(parts[0]))
                    if cid < 0 or cid >= len(class_names):
                        continue
                    cname = class_names[cid]
                    
                    if cname in class_crops:
                        cx, cy, bw, bh = [float(x) for x in parts[1:]]
                        x1 = int((cx - bw / 2.0) * w)
                        y1 = int((cy - bh / 2.0) * h)
                        x2 = int((cx + bw / 2.0) * w)
                        y2 = int((cy + bh / 2.0) * h)
                        
                        # Clip boxes
                        x1 = max(0, min(x1, w - 1))
                        y1 = max(0, min(y1, h - 1))
                        x2 = max(x1 + 1, min(x2, w))
                        y2 = max(y1 + 1, min(y2, h))
                        
                        if (x2 - x1) >= 10 and (y2 - y1) >= 10:
                            class_crops[cname].append(img[y1:y2, x1:x2])
            except Exception:
                pass

    # Load Background class crops
    print(f"[INFO] Extracting Background crops for {split} split...")
    bg_paths = load_background_images(seed=seed)
    # Split backgrounds: 80% train, 20% test
    bg_split_idx = int(0.8 * len(bg_paths))
    bg_paths_split = bg_paths[:bg_split_idx] if is_train else bg_paths[bg_split_idx:]
    class_crops["Background"] = extract_background_crops(bg_paths_split, num_needed=max_per_class, seed=seed)
    
    # Balanced containers
    balanced_crops = []
    balanced_labels = []
    
    print(f"\n--- Balancing statistics ({split}) ---")
    for cls_idx, cname in enumerate(target_classes):
        crops = class_crops[cname]
        raw_count = len(crops)
        
        if raw_count >= max_per_class:
            # Undersample majority class
            rng.shuffle(crops)
            selected = crops[:max_per_class]
            print(f"  * {cname:<12}: Raw {raw_count:>5} -> Undersampled to {max_per_class:>5}")
        elif raw_count > 0:
            # Oversample/Augment minority class
            selected = list(crops)
            shortfall = max_per_class - raw_count
            for _ in range(shortfall):
                base_crop = rng.choice(crops)
                selected.append(augment_crop(base_crop, rng))
            print(f"  * {cname:<12}: Raw {raw_count:>5} -> Oversampled/Augmented to {max_per_class:>5}")
        else:
            # Absolute fallback if a class has 0 samples (e.g. organic in small sub-datasets)
            print(f"  * {cname:<12}: Raw {raw_count:>5} -> [WARN] 0 samples found! Generating synthetic placeholders.")
            selected = []
            for _ in range(max_per_class):
                placeholder = np.random.randint(0, 256, (64, 64, 3), dtype=np.uint8)
                selected.append(placeholder)
                
        balanced_crops.extend(selected)
        balanced_labels.extend([cls_idx] * max_per_class)
        
    return balanced_crops, balanced_labels


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--max-per-class-train", type=int, default=1000)
    parser.add_argument("--max-per-class-test", type=int, default=250)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    
    args.out.mkdir(parents=True, exist_ok=True)
    
    print("=================================================================")
    print("FYP AUTOMATED WASTE MANAGEMENT ML PIPELINE (BALANCED & 637-FEATS)")
    print("=================================================================")
    
    # 1. Statistical Analysis of raw class distribution
    raw_stats = analyze_raw_class_distribution(args.data)
    print("\n--- Raw Class Distribution Statistical Analysis (YOLO Boxes) ---")
    if raw_stats:
        print(f"{'class':<15}{'train':>10}{'valid':>10}{'test':>10}")
        print("-" * 48)
        for cname in raw_stats["train"]:
            print(f"{cname:<15}{raw_stats['train'].get(cname, 0):>10}{raw_stats['val'].get(cname, 0):>10}{raw_stats['test'].get(cname, 0):>10}")
    else:
        print("No raw statistical analysis available (dataset missing or invalid).")
        
    # Define target classes (excluding 'other' for main experiment, adding 'Background')
    target_classes = ["plastic", "glass", "metal", "paper", "cardboard", "organic", "Background"]
    print(f"\nTarget multi-class task classes: {target_classes}")
    
    # 2. Load and balance crops (Oversampling & Undersampling & Background inclusion)
    print("\n[1/5] Loading and balancing train crop split...")
    train_crops, y_train = load_crops_and_balance(
        args.data, target_classes, args.max_per_class_train, is_train=True, seed=args.seed
    )
    
    print("\n[2/5] Loading and balancing test crop split...")
    test_crops, y_test = load_crops_and_balance(
        args.data, target_classes, args.max_per_class_test, is_train=False, seed=args.seed
    )
    
    # 3. Handcrafted Feature Extraction (The 637 Features)
    print(f"\n[3/5] Extracting custom 637 features per crop using custom_feature_extractor...")
    x_train = []
    for idx, crop in enumerate(train_crops):
        if idx % 1000 == 0:
            print(f"  - Train features: {idx}/{len(train_crops)} extracted")
        x_train.append(extract_637_features(crop))
    x_train = np.array(x_train, dtype=np.float32)
    
    x_test = []
    for idx, crop in enumerate(test_crops):
        if idx % 500 == 0:
            print(f"  - Test features: {idx}/{len(test_crops)} extracted")
        x_test.append(extract_637_features(crop))
    x_test = np.array(x_test, dtype=np.float32)
    
    y_train = np.array(y_train, dtype=np.int64)
    y_test = np.array(y_test, dtype=np.int64)
    
    print(f"\nFeature vectors ready:")
    print(f"  - X_train shape: {x_train.shape}")
    print(f"  - X_test shape: {x_test.shape}")
    
    # 4. Fit and Evaluate models
    print("\n[4/5] Training ML Baseline Models on balanced 637 features...")
    models = {
        "decision_tree": DecisionTreeClassifier(max_depth=20, min_samples_leaf=4, random_state=args.seed),
        "linear_svm": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LinearSVC(C=0.5, max_iter=4000, random_state=args.seed))
        ]),
        "rf": RandomForestClassifier(n_estimators=200, random_state=args.seed, n_jobs=-1),
    }
    
    if XGBClassifier is not None:
        models["xgboost"] = XGBClassifier(
            n_estimators=200,
            learning_rate=0.1,
            max_depth=5,
            subsample=0.85,
            colsample_bytree=0.85,
            random_state=args.seed,
            n_jobs=-1,
            eval_metric="mlogloss"
        )
    else:
        print("[WARN] xgboost is not installed in the environment. Skipping XGBoost model training.")
        
    results = []
    detailed_reports = {}
    trained_models = {}
    
    for name, model in models.items():
        print(f"  * Training {name}...")
        model.fit(x_train, y_train)
        pred = model.predict(x_test)
        trained_models[name] = model
        
        acc = float(accuracy_score(y_test, pred))
        f1m = float(f1_score(y_test, pred, average="macro"))
        results.append({"model": name, "accuracy": acc, "f1_macro": f1m})
        
        detailed_reports[name] = classification_report(
            y_test,
            pred,
            labels=list(range(len(target_classes))),
            target_names=target_classes,
            output_dict=True,
            zero_division=0
        )
        
        # Save custom confusion matrix plot
        fig, ax = plt.subplots(figsize=(8, 7))
        cm = confusion_matrix(y_test, pred, labels=list(range(len(target_classes))))
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=target_classes)
        disp.plot(ax=ax, cmap="Blues", xticks_rotation=35, colorbar=False)
        ax.set_title(f"Confusion Matrix: {name} (637 features)")
        fig.tight_layout()
        fig.savefig(args.out / f"confusion_{name}.png", dpi=140)
        plt.close(fig)
        
    results.sort(key=lambda r: r["f1_macro"], reverse=True)
    
    # Save reports
    (args.out / "metrics_summary.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    (args.out / "classification_reports.json").write_text(json.dumps(detailed_reports, indent=2), encoding="utf-8")
    
    # 5. Extract Feature Importance and generate Domain Contribution Chart
    print("\n[5/5] Performing feature importance and domain contribution analysis...")
    rf_model = trained_models.get("rf")
    if rf_model is not None:
        importances = rf_model.feature_importances_
        # Feature offsets: Color (0..255), Texture (256..302), Shape (303..312), Edge/HOG (313..636)
        color_imp = float(np.sum(importances[0:256]))
        texture_imp = float(np.sum(importances[256:303]))
        shape_imp = float(np.sum(importances[303:313]))
        edge_imp = float(np.sum(importances[313:637]))
        
        domain_imps = {
            "Color Features (256)": color_imp * 100,
            "Texture Features (47)": texture_imp * 100,
            "Shape/Geometric (10)": shape_imp * 100,
            "Edge/HOG Features (324)": edge_imp * 100
        }
        
        # Plot domain contribution
        fig, ax = plt.subplots(figsize=(7, 4.5))
        colors = ["#4E79A7", "#F28E2B", "#59A14F", "#E15759"]
        bars = ax.bar(list(domain_imps.keys()), list(domain_imps.values()), color=colors)
        ax.set_title("Handcrafted Feature Domain Contribution (RF Importance)")
        ax.set_ylabel("Contribution (%)")
        ax.set_ylim(0.0, 100.0)
        ax.grid(True, axis="y", alpha=0.3)
        for bar in bars:
            y = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2.0, y + 1.0, f"{y:.2f}%", ha="center", va="bottom", fontsize=10)
        fig.tight_layout()
        fig.savefig(args.out / "chart_domain_importance.png", dpi=140)
        plt.close(fig)
    else:
        domain_imps = {}

    # Write REPORT.md
    write_final_markdown_report(args.out, target_classes, results, domain_imps, raw_stats)
    
    print("\n=================================================================")
    print(f"[OK] Pipeline completed. Results written to: {args.out}")
    print("=================================================================")


def write_final_markdown_report(out_dir: Path, target_classes: list[str], results: list[dict], domain_imps: dict[str, float], raw_stats: dict) -> None:
    lines = []
    lines.append("# Balanced 637-Feature Machine Learning Report\n")
    
    lines.append("## 1. Raw Dataset Class Distribution (Statistical Analysis)\n")
    if raw_stats:
        lines.append("| Class | Train Bboxes | Valid Bboxes | Test Bboxes | Total Bboxes |")
        lines.append("|---|---:|---:|---:|---:|")
        for name in raw_stats["train"]:
            tr = raw_stats["train"].get(name, 0)
            va = raw_stats["val"].get(name, 0)
            te = raw_stats["test"].get(name, 0)
            tot = tr + va + te
            lines.append(f"| {name} | {tr:,} | {va:,} | {te:,} | {tot:,} |")
    else:
        lines.append("Dataset details not loaded.\n")
    lines.append("\n")
    
    lines.append("## 2. Dataset Balancing Strategy\n")
    lines.append("- **Undersampling**: Capped major classes to create a uniform training distribution.\n")
    lines.append("- **Oversampling & Augmentation**: For minority classes, applied flips, rotations, and intensity shifts to achieve a fully balanced set.\n")
    lines.append("- **Negative Samples (Background Class)**: Added real-world empty images and synthetic textures (grass, concrete, carpet) to represent environmental noise, preventing false positives in real-world scenarios.\n\n")
    
    lines.append("## 3. Handcrafted Feature Extractor Layout (Exactly 637 Features)\n")
    lines.append("- **Color (256 Features)**: 144-bin RGB Histograms (3 channels * 48 bins) + 112-bin HSV Histograms (H:48, S:32, V:32).\n")
    lines.append("- **Texture (47 Features)**: 10-bin Uniform LBP Histogram + 37-bin Gray-Level Co-occurrence Matrix (GLCM/Haralick) Descriptors.\n")
    lines.append("- **Shape/Geometric (10 Features)**: 7 log-transformed scale/rotation-invariant Hu Moments + Area, Perimeter, and Circularity.\n")
    lines.append("- **Edge/HOG (324 Features)**: Gradient orientation histogram (64x64 window, 32x32 block, 16x16 cell/stride, 9 bins).\n\n")
    
    if domain_imps:
        lines.append("### Feature Group Importance Contributions\n")
        lines.append("| Feature Group | Random Forest Contribution (%) |")
        lines.append("|---|---:|")
        for group, val in domain_imps.items():
            lines.append(f"| {group} | {val:.2f}% |")
        lines.append("\n")
        
    lines.append("## 4. Machine Learning Baseline Comparison\n")
    lines.append("| Model | Accuracy | Macro F1-score |")
    lines.append("|---|---:|---:|")
    for r in results:
        lines.append(f"| {r['model']} | {r['accuracy']:.4f} | {r['f1_macro']:.4f} |")
    lines.append("\n")
    
    lines.append("## 5. Noise and Background Performance Assessment\n")
    lines.append("Please inspect the saved confusion matrix files (`confusion_*.png`) to see how the models separate the `Background` class from actual waste items (`plastic`, `glass`, `metal`, etc.). Training models with a high-fidelity `Background` class prevents false positive classifications on floors, tables, and grass in edge deployments.\n")
    
    (out_dir / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
