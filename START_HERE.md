# WasteWise Project Map

Open this file first when you need to find project evidence quickly.

## What To Show Lecturer First

1. Final pipeline/report:
   - `docs\01_final_report\FINAL_PROJECT_PIPELINE_REPORT.md`
2. Dataset cleaning and EDA proof:
   - `docs\02_dataset_training\DATASET_EDA_AND_TUNING_REPORT.md`
   - `runs\dataset_eda\merged_dataset_v3\EDA_REPORT.md`
   - `runs\dataset_eda\tuned_dataset_v1\EDA_REPORT.md`
3. Tuned YOLO result:
   - `runs\dl\trash_yolov8n_tuned_v1\quality_check\REPORT.md`
   - `runs\dl\trash_yolov8n_tuned_v1\quality_check\test_metrics.json`
   - `runs\dl\trash_yolov8n_tuned_v1\results.png`
4. Classical ML baseline:
   - `runs\ml\feature_ml_enhanced_6class_4k\metrics_summary.json`
   - `runs\ml\feature_ml_enhanced_6class_4k\chart_model_comparison.png`
5. Manual prediction demo:
   - `runs\manual_tests\`

## Main Result

- Classical ML best model: Extra Trees, accuracy `0.6312`, F1-macro `0.6113`.
- Tuned YOLOv8n: test mAP@0.5 `0.810`, test mAP@0.5:0.95 `0.622`.
- Conclusion: classical ML is the explainable baseline; YOLOv8n is the stronger final detector.

## Folder Guide

- `assets\` - local demo images and project media.
- `docs\01_final_report\` - final report material and reproducible pipeline.
- `docs\02_dataset_training\` - dataset EDA, tuning, and training progress notes.
- `docs\03_lecturer_notes\` - weekly/lecturer-facing notes.
- `merged_dataset_v3\` - locked original dataset baseline.
- `tuned_dataset_v1\` - cleaned/tuned dataset used for improved YOLO training.
- `models\pretrained\` - downloaded/pretrained model files.
- `ml\` - handcrafted feature analysis outputs.
- `mobile\` - Expo React Native app.
- `runs\` - generated experiment evidence, plots, metrics, and weights.
- `scripts\` - command-line tools for EDA, training support, evaluation, export, and prediction.

## Key Commands

```powershell
.\.venv311\Scripts\python.exe scripts\dataset_eda.py --data merged_dataset_v3\data.yaml --out runs\dataset_eda\merged_dataset_v3 --hash-duplicates
.\.venv311\Scripts\python.exe scripts\dataset_eda.py --data tuned_dataset_v1\data.yaml --out runs\dataset_eda\tuned_dataset_v1 --hash-duplicates
.\.venv311\Scripts\python.exe scripts\evaluate.py --weights runs\dl\trash_yolov8n_tuned_v1\weights\best.pt --data tuned_dataset_v1\data.yaml --out runs\dl\trash_yolov8n_tuned_v1\quality_check --split both
.\.venv311\Scripts\python.exe scripts\predict_images.py --source assets\manual_test_images --weights runs\dl\trash_yolov8n_tuned_v1\weights\best.pt --conf 0.10
```
