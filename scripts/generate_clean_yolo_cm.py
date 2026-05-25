import sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from ultralytics import YOLO

def main():
    ROOT = Path("C:/FYP")
    model_path = ROOT / "runs" / "detect" / "yolov11_super_dataset" / "weights" / "best.pt"
    data_config = ROOT / "external_datasets" / "super_yolo_dataset" / "data.yaml"
    output_dir = ROOT / "runs" / "detect" / "yolov11_super_dataset_validation_plots"
    artifact_dir = Path(r"C:\Users\PC\.gemini\antigravity\brain\593e7fc3-a808-4784-8d6f-901aa978bf93")

    print("Loading model...")
    model = YOLO(model_path)

    print("Running validation to extract raw confusion matrix...")
    # Wrap in try-except to get detailed logs
    try:
        results = model.val(data=str(data_config), device=0, plots=True, workers=0) # workers=0 avoids multiprocessing issues on Windows!
        
        # Get the raw confusion matrix matrix from the validator
        # In Ultralytics, the confusion matrix is in results.confusion_matrix.matrix
        if hasattr(results, "confusion_matrix") and hasattr(results.confusion_matrix, "matrix"):
            cm = results.confusion_matrix.matrix
            print("Successfully retrieved confusion matrix from results. Shape:", cm.shape)
        else:
            # Fallback to accessing validator directly
            print("confusion_matrix attribute not found on results. Checking validator...")
            validator = model.predictor.validator if hasattr(model.predictor, "validator") else None
            if validator and hasattr(validator, "confusion_matrix"):
                cm = validator.confusion_matrix.matrix
                print("Successfully retrieved confusion matrix from validator. Shape:", cm.shape)
            else:
                raise AttributeError("Could not find confusion matrix matrix in YOLO validation results!")

        # The names of the classes (excluding background)
        class_names = ["plastic", "glass", "metal", "paper", "cardboard", "organic"]
        nc = len(class_names)

        # In Ultralytics YOLO, the confusion matrix is of shape (nc+1, nc+1) where the last row/column is background.
        # Let's crop the matrix to nc x nc (index 0 to 5) to remove the background class!
        clean_cm = cm[0:nc, 0:nc]
        print("Cropped clean confusion matrix (6x6):")
        print(clean_cm)

        # 1. Plot Raw Counts Confusion Matrix
        plt.figure(figsize=(8, 7))
        sns.heatmap(
            clean_cm, 
            annot=True, 
            fmt=".0f", 
            cmap="Blues", 
            xticklabels=class_names, 
            yticklabels=class_names,
            cbar=True,
            square=True,
            annot_kws={"size": 10, "weight": "bold"}
        )
        plt.title("YOLOv11 Confusion Matrix (Without Background - Raw Counts)", fontsize=12, fontweight="bold", pad=15)
        plt.xlabel("Predicted Class", fontsize=10, fontweight="bold", labelpad=10)
        plt.ylabel("True Class", fontsize=10, fontweight="bold", labelpad=10)
        plt.xticks(rotation=35, ha="right")
        plt.yticks(rotation=0)
        plt.tight_layout()

        # Save raw matrix
        raw_path = output_dir / "confusion_matrix_no_background.png"
        raw_artifact_path = artifact_dir / "confusion_matrix_no_background.png"
        plt.savefig(raw_path, dpi=150)
        plt.savefig(raw_artifact_path, dpi=150)
        plt.close()
        print(f"[OK] Saved raw confusion matrix to {raw_path} and artifact folder.")

        # 2. Plot Normalized Confusion Matrix
        # Row-normalize: divide each row by its sum (with epsilon to avoid division by zero)
        row_sums = clean_cm.sum(axis=1, keepdims=True)
        normalized_cm = clean_cm / (row_sums + 1e-9)

        plt.figure(figsize=(8, 7))
        sns.heatmap(
            normalized_cm, 
            annot=True, 
            fmt=".2f", 
            cmap="Blues", 
            xticklabels=class_names, 
            yticklabels=class_names,
            cbar=True,
            square=True,
            annot_kws={"size": 10, "weight": "bold"},
            vmin=0,
            vmax=1
        )
        plt.title("YOLOv11 Confusion Matrix (Without Background - Normalized)", fontsize=12, fontweight="bold", pad=15)
        plt.xlabel("Predicted Class", fontsize=10, fontweight="bold", labelpad=10)
        plt.ylabel("True Class", fontsize=10, fontweight="bold", labelpad=10)
        plt.xticks(rotation=35, ha="right")
        plt.yticks(rotation=0)
        plt.tight_layout()

        # Save normalized matrix
        norm_path = output_dir / "confusion_matrix_no_background_normalized.png"
        norm_artifact_path = artifact_dir / "confusion_matrix_no_background_normalized.png"
        plt.savefig(norm_path, dpi=150)
        plt.savefig(norm_artifact_path, dpi=150)
        plt.close()
        print(f"[OK] Saved normalized confusion matrix to {norm_path} and artifact folder.")

        print("\n========================================================")
        print("🟢 CUSTOM 6X6 CONFUSION MATRIX GENERATION COMPLETE!")
        print("========================================================")
    except Exception as e:
        print(f"Error during validation: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
