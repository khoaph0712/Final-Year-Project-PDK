# WasteWise — YOLOv8n Waste Sorting (FYP)

Real-time waste classification using **YOLOv8n (Ultralytics)** trained on a merged
Roboflow dataset, exported to **TFLite / ONNX**, and deployed on a **React Native
(Expo) mobile app** with on-device inference.

## Final project status

- Final dataset: `merged_dataset_v3\data.yaml`.
- Final pipeline/report guide: `docs\FINAL_PROJECT_PIPELINE_REPORT.md`.
- Classical ML baseline: enhanced 6-class capped run in `runs\ml\feature_ml_enhanced_6class_4k\`.
- Main deployment model: YOLOv8n run in `runs\dl\trash_yolov8n_v3\`.
- Mobile app: Expo Dev Client app in `mobile\`, using bundled Float16/Float32 TFLite models.
- No-UI manual tester: `scripts\predict_images.py`.

Current headline results:

| System | Metric | Result |
|---|---|---:|
| Classical ML best model (`extra_trees`) | Accuracy | 0.6312 |
| Classical ML best model (`extra_trees`) | F1-macro | 0.6113 |
| YOLOv8n detector | Test mAP@0.5 | 0.7559 |
| YOLOv8n detector | Test mAP@0.5:0.95 | 0.5754 |

```
C:\FYP_v2
├── merged_dataset_v2\             # legacy 7-class dataset (optional; v3 is default in scripts)
├── merged_dataset_v3\             # canonical 7-class dataset (train/valid/test)
├── ml\
│   └── frequency_analysis\        # spatial/frequency CSVs + plots (from feature_ml_analysis.py)
├── runs\
│   ├── ml\                        # classical ML + feature reports (LogReg, SVM, RF, …)
│   ├── dl\                        # YOLO training runs + dl_baseline (tiny CNN)
│   └── comparisons\             # ML vs DL charts + REPORT (from compare_ml_dl.py)
├── scripts\                       # CLI tools (dataset, train helpers, eval, export, ML)
├── mobile\                        # Expo (Dev Client) React Native app
└── requirements.txt
```

## 0 · Install Python deps

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

> `tensorflow==2.16.1` is required by Ultralytics for TFLite export. Installation
> can be slow on Windows — expect 3–5 min.

## 1 · Quality check (are we mobile-ready?)

The v2 run (historical) at `runs/dl/trash_yolov8n_v2/weights/best.pt` reached:


| Metric                         | Value |
| ------------------------------ | ----- |
| Precision                      | 0.730 |
| Recall                         | 0.581 |
| [mAP@0.5](mailto:mAP@0.5)      | 0.654 |
| [mAP@0.5](mailto:mAP@0.5):0.95 | 0.476 |


Run the quality report:

```powershell
python scripts\evaluate.py --split both
python scripts\plot_training.py
```

`evaluate.py` defaults to v3 and writes under `runs/dl/trash_yolov8n_v3/quality_check/`. (Older v2 output lives in `runs/dl/trash_yolov8n_v2/quality_check/`.)

- `REPORT.md` — overall + per-class table
- `val_metrics.json`, `test_metrics.json`
- `confusion_matrix.png`, `PR_curve.png`, `F1_curve.png`
- `predictions/` — annotated sample images
- `training_curves.png`

## 1.5 · Feature extraction + ML comparison (for analysis/report)

Run handcrafted feature analysis first and classic ML baselines. The final enhanced setup uses
6 classes, excludes `other`, caps train crops at `4000` per class, and extracts 637 features:
spatial, frequency/FFT, color, and HOG.

```powershell
.\.venv311\Scripts\python.exe scripts\feature_ml_analysis.py `
  --data merged_dataset_v3\data.yaml `
  --out runs\ml\feature_ml_enhanced_6class_4k `
  --exclude-classes other `
  --max-per-class-train 4000 `
  --max-per-class-test 800
```

Outputs are written to `runs/ml/feature_ml_enhanced_6class_4k/`:

- `metrics_summary.json` — Accuracy/F1 summary by model
- `classification_reports.json` — full per-class precision/recall/F1
- `object_difference.json` — class-wise distinct feature comments
- `class_support.json` — train/test sample count per class (to detect imbalance)
- `confusion_*.png` — confusion matrix per model
- `chart_domain_importance.png` — spatial/frequency/color/HOG contribution chart
- `chart_model_comparison.png` — model comparison chart
- `REPORT.md` — rationale, chart comments, and conclusions

Also exports domain summaries to `ml/frequency_analysis/`:

- `spatial_summary.csv` — spatial features by class
- `frequency_summary.csv` — frequency features by class
- `color_summary.csv` — color features by class
- `domain_comparison.csv` — feature-group comparison metrics

## 1.6 · Deep-learning baseline + ML-vs-DL comparison

Train a lightweight CNN baseline on the same object-crop setup:

```powershell
python scripts\deep_learning_baseline.py --data merged_dataset_v3\data.yaml
```

Then generate a unified comparison report:

```powershell
.\.venv311\Scripts\python.exe scripts\compare_ml_dl.py `
  --ml-metrics runs\ml\feature_ml_enhanced_6class_4k\metrics_summary.json `
  --out runs\comparisons\model_comparison
```

Outputs:

- `runs/dl/dl_baseline/metrics.json`, `confusion_tiny_cnn.png`, `training_loss.png`
- `runs/comparisons/model_comparison/REPORT.md`, `comparison_metrics.json`, `chart_ml_vs_dl.png`

### How to interpret


| You see…                                     | Likely cause                  | Fix                                          |
| -------------------------------------------- | ----------------------------- | -------------------------------------------- |
| [mAP@0.5](mailto:mAP@0.5):0.95 < 0.4 on test | Underfitting / too few epochs | `epochs=50`, `imgsz=800`, consider `yolov8s` |
| One class with very low AP                   | Dataset imbalance             | Oversample or add data for that class        |
| High precision, low recall                   | Threshold too strict          | Lower `conf` in the app (Settings)           |
| `plastic ↔ glass` bleed                      | Visually similar              | Add diverse angles/lighting samples          |


## 2 · Export for mobile

```powershell
python scripts\export_model.py --imgsz 640
```

Produces (next to `best.pt`):

- `best.onnx`
- `best_float32.tflite`
- `best_float16.tflite`
- `best_int8.tflite` ← use on phone
- `best_metadata.json`

Copy the two TFLite files into `mobile/assets/model/`:

```powershell
Copy-Item runs\dl\trash_yolov8n_v3\weights\best_int8.tflite mobile\assets\model\
Copy-Item runs\dl\trash_yolov8n_v3\weights\best_float16.tflite mobile\assets\model\
```

## 3 · Run the mobile app

See `mobile/README.md` for full instructions.

```powershell
cd mobile
npm install
npx expo prebuild --clean
npm run android     # or: npm run ios
```

## 4 · Reproduce the final project pipeline

Run all final pipeline stages from one PowerShell script:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\run_final_pipeline.ps1
```

The full pipeline can be slow because it runs feature extraction, ML training, the CNN baseline,
YOLO evaluation, and export. To run a faster report-refresh pass:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\run_final_pipeline.ps1 -SkipFeatureMl -SkipTinyCnn -SkipYoloEval -SkipExport
```

To test your own images without a UI:

```powershell
.\.venv311\Scripts\python.exe scripts\predict_images.py --source C:\path\to\test_images --conf 0.10
```

This writes annotated images and `predictions_summary.json` under `runs\manual_tests\yolo_predictions\`,
including sorting output such as `recyclable`, `organic`, `general waste`, or `no detection`.

---

## Retraining tips (if the quality check is underwhelming)

```powershell
# Longer training + bigger image size
yolo detect train `
  model=runs\dl\trash_yolov8n\weights\best.pt `
  data=merged_dataset_v3\data.yaml `
  epochs=50 imgsz=800 batch=16 `
  project=runs\dl name=trash_yolov8n_v4 exist_ok=True
```
