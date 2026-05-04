# WasteWise — YOLOv8n Waste Sorting (FYP)

Real-time waste classification using **YOLOv8n (Ultralytics)** trained on a merged
Roboflow dataset, exported to **TFLite / ONNX**, and deployed on a **React Native
(Expo) mobile app** with on-device inference.

```
C:\FYP_v2
├── merged_dataset_v2\             # 7-class merged dataset (v2, baseline)
├── merged_dataset_v3\             # 7-class merged dataset (v3, cleaner organic + re-split glass)
├── runs\
│   ├── trash_yolov8n\             # initial run (weights/best.pt)
│   ├── trash_yolov8n_v2\          # fine-tuned run (20 epochs on v2 data)
│   │   └── quality_check\         # ← produced by scripts/evaluate.py
│   └── trash_yolov8n_v3\          # fine-tuned on v3 data
├── scripts\
│   ├── download_datasets.py       # original 5 Roboflow datasets → rf_* folders
│   ├── download_extra.py          # extra Roboflow datasets for organic/other
│   ├── build_merged_dataset.py    # merge rf_* folders → merged_dataset_v3
│   ├── check_class_balance.py     # per-class / per-split box counts
│   ├── evaluate.py                # val + test metrics, confusion matrix, sample preds, REPORT.md
│   ├── plot_training.py           # loss / mAP curves from results.csv
│   └── export_model.py            # ONNX + TFLite (fp32 / fp16 / int8) + metadata.json
├── mobile\                        # Expo (Dev Client) React Native app
└── requirements.txt               # Python deps for training/eval/export
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

The model at `runs/trash_yolov8n_v2/weights/best.pt` reached:


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

This writes to `runs/trash_yolov8n_v2/quality_check/`:

- `REPORT.md` — overall + per-class table
- `val_metrics.json`, `test_metrics.json`
- `confusion_matrix.png`, `PR_curve.png`, `F1_curve.png`
- `predictions/` — annotated sample images
- `training_curves.png`

## 1.5 · Feature extraction + ML comparison (for analysis/report)

Run handcrafted feature analysis first (spatial + frequency domains) and classic ML baselines:

```powershell
python scripts\feature_ml_analysis.py --data merged_dataset_v3\data.yaml
```

Outputs are written to `runs/feature_ml_analysis/`:

- `metrics_summary.json` — Accuracy/F1 summary by model
- `classification_reports.json` — full per-class precision/recall/F1
- `object_difference.json` — class-wise distinct feature comments
- `class_support.json` — train/test sample count per class (to detect imbalance)
- `confusion_*.png` — confusion matrix per model
- `chart_domain_importance.png` — spatial vs frequency contribution chart (importance-based)
- `chart_model_comparison.png` — model comparison chart
- `REPORT.md` — rationale, chart comments, and conclusions

Also exports domain summaries to `frequency_analysis/`:

- `spatial_summary.csv` — spatial features by class
- `frequency_summary.csv` — frequency features by class
- `domain_comparison.csv` — spatial vs frequency comparison metrics

## 1.6 · Deep-learning baseline + ML-vs-DL comparison

Train a lightweight CNN baseline on the same object-crop setup:

```powershell
python scripts\deep_learning_baseline.py --data merged_dataset_v3\data.yaml
```

Then generate a unified comparison report:

```powershell
python scripts\compare_ml_dl.py
```

Outputs:

- `runs/dl_baseline/metrics.json`, `confusion_tiny_cnn.png`, `training_loss.png`
- `runs/model_comparison/REPORT.md`, `comparison_metrics.json`, `chart_ml_vs_dl.png`

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
Copy-Item runs\trash_yolov8n_v2\weights\best_int8.tflite mobile\assets\model\
Copy-Item runs\trash_yolov8n_v2\weights\best_float16.tflite mobile\assets\model\
```

## 3 · Run the mobile app

See `mobile/README.md` for full instructions.

```powershell
cd mobile
npm install
npx expo prebuild --clean
npm run android     # or: npm run ios
```

---

## Retraining tips (if the quality check is underwhelming)

```powershell
# Longer training + bigger image size
yolo detect train `
  model=runs\trash_yolov8n\weights\best.pt `
  data=merged_dataset_v2\data.yaml `
  epochs=50 imgsz=800 batch=16 `
  project=runs name=trash_yolov8n_v3 exist_ok=True
```

