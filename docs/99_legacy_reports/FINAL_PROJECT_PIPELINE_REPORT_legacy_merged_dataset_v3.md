# Final Project Pipeline Report

This document freezes the current project direction so the work can move from dataset iteration into reproducible results, demo testing, and final write-up.

## 1. Locked Dataset And Evidence

The final dataset is `merged_dataset_v3`, referenced by `merged_dataset_v3\data.yaml`.

Raw YOLO dataset classes:

| Class | Train boxes | Valid boxes | Test boxes | Total boxes |
|---|---:|---:|---:|---:|
| plastic | 17,786 | 1,792 | 2,214 | 21,792 |
| glass | 7,020 | 2,520 | 203 | 9,743 |
| metal | 12,292 | 2,144 | 770 | 15,206 |
| paper | 7,286 | 1,012 | 2,145 | 10,443 |
| cardboard | 17,377 | 3,386 | 1,042 | 21,805 |
| organic | 22,785 | 2,766 | 2,895 | 28,446 |
| other | 5,361 | 1,099 | 325 | 6,785 |
| **Total boxes** | **89,907** | **14,719** | **9,594** | **114,220** |

The raw data is imbalanced. For the lecturer-required classical ML experiment, the project uses a balanced capped sample:

- Dataset: `merged_dataset_v3`
- Classes used: `plastic`, `glass`, `metal`, `paper`, `cardboard`, `organic`
- Excluded class: `other`
- Training cap: `4000` object crops per class
- Total train crops: `24000`
- Output folder: `runs\ml\feature_ml_enhanced_6class_4k`

Current evidence to report honestly:

| System | Metric | Result |
|---|---|---:|
| Lecturer ML best model (`xgboost`) | Accuracy | 0.6742 |
| Lecturer ML best model (`xgboost`) | F1-macro | 0.6506 |
| ANN baseline on tuned dataset | Accuracy | 0.4057 |
| CNN baseline on tuned dataset | Accuracy | 0.4413 |
| YOLOv8n detector | Test mAP@0.5 | 0.7559 |
| YOLOv8n detector | Test mAP@0.5:0.95 | 0.5754 |

Conclusion for the current lecturer checkpoint: explain the handcrafted features + classical ML first, show ANN/CNN baselines with saved logs, and keep YOLO as previous/final detector evidence unless asked.

## 2. Reproducible Pipeline

Run these commands from `C:\FYP_v2` using the Python 3.11 virtual environment.

To run the main stages through one script:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\run_final_pipeline.ps1
```

For a faster refresh that reuses existing heavy outputs:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\run_final_pipeline.ps1 -SkipFeatureMl -SkipTinyCnn -SkipYoloEval -SkipExport
```

### 2.1 Check Class Balance

```powershell
.\.venv311\Scripts\python.exe scripts\check_class_balance.py merged_dataset_v3
```

Expected output: class counts for train, valid, and test splits.

### 2.2 Feature Extraction And Classical ML

```powershell
.\.venv311\Scripts\python.exe scripts\feature_ml_analysis.py `
  --data merged_dataset_v3\data.yaml `
  --out runs\ml\feature_ml_lecturer_6class_4k `
  --exclude-classes other `
  --max-per-class-train 4000 `
  --max-per-class-test 800 `
  --domain-out ml\frequency_analysis
```

Expected output folder: `runs\ml\feature_ml_lecturer_6class_4k`

Important artifacts:

- `REPORT.md`
- `class_support.json`
- `metrics_summary.json`
- `classification_reports.json`
- `chart_model_comparison.png`
- `chart_domain_importance.png`
- `confusion_decision_tree.png`
- `confusion_linear_svm.png`
- `confusion_rf.png`
- `confusion_xgboost.png`

### 2.3 ANN/CNN Baselines

```powershell
.\.venv311\Scripts\python.exe scripts\deep_learning_baseline.py `
  --data merged_dataset_v3\data.yaml `
  --out runs\dl\ann_cnn_merged_dataset_v3 `
  --model both `
  --max-train-objects 8000 `
  --max-val-objects 2000 `
  --max-test-objects 3000 `
  --epochs 8

.\.venv311\Scripts\python.exe scripts\deep_learning_baseline.py `
  --data tuned_dataset_v1\data.yaml `
  --out runs\dl\ann_cnn_tuned_dataset_v1 `
  --model both `
  --max-train-objects 8000 `
  --max-val-objects 2000 `
  --max-test-objects 3000 `
  --epochs 8
```

Expected output folders:

- `runs\dl\ann_cnn_merged_dataset_v3`
- `runs\dl\ann_cnn_tuned_dataset_v1`

Important artifacts:

- `metrics_summary.json`
- `REPORT.md`
- `ann\ann.pt`, `ann\ann_full.pt`
- `cnn\cnn.pt`, `cnn\cnn_full.pt`
- `training_log.csv` and `training_log.json`
- `training_curves_ann.png` / `training_curves_cnn.png`
- `confusion_ann.png` / `confusion_cnn.png`
- `classification_report.json`

### 2.4 ML-vs-DL Comparison

```powershell
.\.venv311\Scripts\python.exe scripts\compare_ml_dl.py `
  --ml-metrics runs\ml\feature_ml_enhanced_6class_4k\metrics_summary.json `
  --out runs\comparisons\model_comparison
```

Expected output folder: `runs\comparisons\model_comparison`

Important artifacts:

- `REPORT.md`
- `comparison_metrics.json`
- `chart_ml_vs_dl.png`

### 2.5 YOLO Quality Check

```powershell
.\.venv311\Scripts\python.exe scripts\evaluate.py `
  --weights runs\dl\trash_yolov8n_v3\weights\best.pt `
  --data merged_dataset_v3\data.yaml `
  --split both
```

Expected output folder: `runs\dl\trash_yolov8n_v3\quality_check`

Important artifacts:

- `REPORT.md`
- `val_metrics.json`
- `test_metrics.json`
- `confusion_matrix.png`
- `PR_curve.png`
- `F1_curve.png`

### 2.6 Export Model For Mobile

```powershell
.\.venv311\Scripts\python.exe scripts\export_model.py --imgsz 640
```

Expected output folder: `runs\dl\trash_yolov8n_v3\weights`

Important artifacts:

- `best.onnx`
- `best_float32.tflite`
- `best_float16.tflite`
- `best_metadata.json`

### 2.7 Manual No-UI Prediction Test

```powershell
.\.venv311\Scripts\python.exe scripts\predict_images.py `
  --source C:\path\to\test_images `
  --conf 0.10
```

Expected output folder: `runs\manual_tests\yolo_predictions`

Important artifacts:

- annotated prediction images
- `predictions_summary.json`
- `sorting_output` values such as `recyclable`, `organic`, `general waste`, or `no detection`

### 2.8 Mobile App Check

The mobile app is an Expo Dev Client app in `mobile\`. It includes live camera inference using Vision Camera, `react-native-fast-tflite`, YOLO post-processing, scan history, settings, and sorting guidance.

```powershell
cd mobile
C:\Users\PC\.cache\codex-runtimes\codex-primary-runtime\dependencies\node\bin\node.exe .\node_modules\typescript\bin\tsc --noEmit
npx expo prebuild --clean
npm run android
```

Model assets currently bundled:

- `mobile\assets\model\best_float16.tflite`
- `mobile\assets\model\best_float32.tflite`
- `mobile\assets\model\best_metadata.json`

The app setting is **Use Float16 model**:

- On: faster live preview with `best_float16.tflite`
- Off: maximum accuracy with `best_float32.tflite`

Final mobile validation still requires a physical Android device or emulator because camera/TFLite runtime behavior cannot be fully verified from static checks alone.

## 3. Feature And Model Summary

The enhanced classical ML run uses 637 handcrafted features extracted from YOLO object crops:

| Feature group | Count | Purpose |
|---|---:|---|
| Spatial | 8 | Intensity, gradient, and edge-density descriptors |
| Frequency / FFT | 9 | Radial FFT band energy plus high-frequency summary |
| Color | 44 | HSV histograms plus BGR/HSV mean and standard deviation |
| HOG | 576 | Texture and shape descriptor from each 64x64 crop |

Models trained on extracted features:

- Decision Tree
- Linear SVM
- Random Forest
- XGBoost
- Logistic Regression
- Extra Trees

Model comparison:

| Model | Accuracy | F1-macro |
|---|---:|---:|
| xgboost | 0.6742 | 0.6506 |
| extra_trees | 0.6312 | 0.6113 |
| rf | 0.6317 | 0.6111 |
| linear_svm | 0.5960 | 0.5642 |
| logreg | 0.5864 | 0.5558 |
| decision_tree | 0.5115 | 0.4883 |

Why 637 features:

- `8` spatial features describe intensity, gradients, and edges.
- `9` frequency/FFT features describe radial frequency energy and high-frequency texture.
- `44` color features describe HSV histograms plus BGR/HSV mean and standard deviation.
- `576` HOG features describe local gradient orientation/shape.
- Total: `8 + 9 + 44 + 576 = 637`.

## 4. ANN/CNN Baseline Results

YOLO is paused for this checkpoint. ANN/CNN are trained as image-crop classifiers to satisfy the deep-learning baseline requirement.

| Dataset | Model | Test Accuracy | Test F1-macro | Best Val Accuracy |
|---|---|---:|---:|---:|
| `merged_dataset_v3` | ANN | 0.3507 | 0.3278 | 0.3890 |
| `merged_dataset_v3` | CNN | 0.4730 | 0.4144 | 0.4195 |
| `tuned_dataset_v1` | ANN | 0.4057 | 0.3885 | 0.4245 |
| `tuned_dataset_v1` | CNN | 0.4413 | 0.4138 | 0.4730 |

Saved output evidence:

- `runs\dl\ann_cnn_merged_dataset_v3\REPORT.md`
- `runs\dl\ann_cnn_tuned_dataset_v1\REPORT.md`
- saved models: `ann.pt`, `ann_full.pt`, `cnn.pt`, `cnn_full.pt`
- logs: `training_log.csv`, `training_log.json`
- figures: train/val loss + accuracy curves, confusion matrices
- reports: `classification_report.json`

## 5. Confusion Matrix Background Explanation

For ANN/CNN crop classification, there is no `background` class because every crop is already assigned to one dataset class.

For YOLO detection confusion matrices, `background` is not an eighth trash category. It represents detection matching errors:

- True class to background: a real object was missed.
- Background to predicted class: the detector predicted an object where no ground-truth box matched.
- Normal class-to-class cells: the detector found an object but predicted the wrong class.

Feature-group importance from Random Forest:

| Group | Importance |
|---|---:|
| HOG | 59.5808% |
| Color | 29.2090% |
| Frequency | 5.6582% |
| Spatial | 5.5520% |

## 4. Figures To Include In Report

Classical ML:

- `runs\ml\feature_ml_enhanced_6class_4k\chart_model_comparison.png`
- `runs\ml\feature_ml_enhanced_6class_4k\chart_domain_importance.png`
- `runs\ml\feature_ml_enhanced_6class_4k\confusion_extra_trees.png`

YOLO / DL:

- `runs\dl\trash_yolov8n_v3\results.png`
- `runs\dl\trash_yolov8n_v3\quality_check\confusion_matrix.png`
- `runs\dl\trash_yolov8n_v3\quality_check\PR_curve.png`

Comparison:

- `runs\comparisons\model_comparison\chart_ml_vs_dl.png`

Manual demo:

| Image | Detected class | Confidence | Sorting output | Note |
|---|---|---:|---|---|
| `assets\manual_test_images\87514.jpg` | plastic | 0.1658 | recyclable | Weak confidence; use as demo of threshold sensitivity |
| `assets\manual_test_images\test.jpg` | none | n/a | no detection | Document as limitation / unfamiliar image case |

## 5. Improvement Direction

Do not randomly collect more data. Only add or clean data for observed weak cases:

- `organic` has the weakest YOLO AP@0.5:0.95.
- `paper` has lower YOLO recall than stronger classes.
- Real phone-camera images can fail when they differ from the training distribution.
- Manual demo images with weak or missing detections should be added only if they represent realistic target usage.

If higher accuracy is required, prioritize:

```powershell
yolo detect train `
  model=runs\dl\trash_yolov8n_v3\weights\best.pt `
  data=merged_dataset_v3\data.yaml `
  epochs=50 imgsz=800 batch=16 `
  project=runs\dl name=trash_yolov8n_v4 exist_ok=True
```

Then evaluate:

```powershell
.\.venv311\Scripts\python.exe scripts\evaluate.py `
  --weights runs\dl\trash_yolov8n_v4\weights\best.pt `
  --data merged_dataset_v3\data.yaml `
  --split both
```

Optional research direction, only if allowed by the lecturer: use pretrained image embeddings as features and train classical ML on those embeddings. Report that separately because it is no longer purely handcrafted feature extraction.

## 6. Final Statement For Report

This project first runs classical ML on handcrafted object-crop features to satisfy the explainable feature-analysis requirement. The balanced ML setup uses 6 classes with 4000 train crops per class, producing a best classical ML accuracy of 0.6312. YOLOv8n is kept as the final deployment detector because it achieves stronger object-detection performance on the same dataset, with test mAP@0.5 of 0.7559 and mAP@0.5:0.95 of 0.5754.
