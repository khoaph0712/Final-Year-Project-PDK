# WasteWise Project Map

Open this file first when you need to find project evidence quickly.

## What To Show Lecturer First

1. Final pipeline/report:
   - `docs\01_final_report\FINAL_PROJECT_PIPELINE_REPORT.md`
2. Dataset cleaning and EDA proof:
   - `docs\02_dataset_training\DATASET_EDA_AND_TUNING_REPORT.md`
   - `docs\02_dataset_training\SOURCE_LEVEL_DATASET_ANALYSIS.md`
   - `docs\02_dataset_training\BALANCED_ML_ANN_CNN_RESULTS.md`
   - `docs\02_dataset_training\EXTERNAL_ONLY_ML_ANN_CNN_RESULTS.md`
   - `docs\02_dataset_training\PHASE2_EXPERIMENT_AUDIT.md`
   - `runs\dataset_eda\merged_dataset_v3\EDA_REPORT.md`
   - `runs\dataset_eda\tuned_dataset_v1\EDA_REPORT.md`
   - `runs\source_analysis\merged_dataset_v3\DATASET_SOURCE_COMPARISON.md`
   - `runs\raw_source_analysis\rf_taco_trash\DATASET_SOURCE_COMPARISON.md`
   - `runs\raw_source_analysis\rf_garbage_cls\DATASET_SOURCE_COMPARISON.md`
   - `runs\raw_source_analysis\rf_food_waste\DATASET_SOURCE_COMPARISON.md`
   - `docs\02_dataset_training\EXTERNAL_DATASET_RESEARCH_PLAN.md`
   - `runs\external_dataset_registry\DATASET_CANDIDATE_SUMMARY.md`
   - `prepared_datasets\trashnet\PREPARE_REPORT.md`
   - `prepared_datasets\taco_official_partial\PREPARE_REPORT.md`
   - `runs\ml\by_dataset\trashnet\REPORT.md`
   - `runs\ml\by_dataset\taco_official_partial\REPORT.md`
3. Tuned YOLO result:
   - `runs\dl\trash_yolov8n_tuned_v1\quality_check\REPORT.md`
   - `runs\dl\trash_yolov8n_tuned_v1\quality_check\test_metrics.json`
   - `runs\dl\trash_yolov8n_tuned_v1\results.png`
4. Lecturer-requested classical ML baseline:
   - `runs\ml\feature_ml_lecturer_6class_4k\REPORT.md`
   - `runs\ml\feature_ml_lecturer_6class_4k\metrics_summary.json`
   - `runs\ml\feature_ml_lecturer_6class_4k\classification_reports.json`
   - `runs\ml\feature_ml_lecturer_6class_4k\confusion_decision_tree.png`
   - `runs\ml\feature_ml_lecturer_6class_4k\confusion_linear_svm.png`
   - `runs\ml\feature_ml_lecturer_6class_4k\confusion_rf.png`
   - `runs\ml\feature_ml_lecturer_6class_4k\confusion_xgboost.png`
5. Lecturer-requested ANN/CNN baselines:
   - `runs\dl\ann_cnn_merged_dataset_v3\REPORT.md`
   - `runs\dl\ann_cnn_tuned_dataset_v1\REPORT.md`
   - `runs\dl\source_ann_cnn\rf_taco_trash\REPORT.md`
   - `runs\dl\source_ann_cnn\rf_waste_sorting\REPORT.md`
   - `runs\dl\raw_ann_cnn\rf_taco_trash\REPORT.md`
   - `runs\dl\raw_ann_cnn\rf_garbage_cls\REPORT.md`
   - `runs\dl\raw_ann_cnn\rf_food_waste\REPORT.md`
   - each model folder contains saved `.pt`, training logs, curves, confusion matrix, and classification report
6. Manual prediction demo:
   - `runs\manual_tests\`

## Main Result

- Lecturer ML best model: XGBoost, accuracy `0.6742`, F1-macro `0.6506`.
- ANN/CNN crop baselines are saved under `runs\dl\ann_cnn_*`; CNN is better than ANN but still below YOLO.
- Source-level analysis confirms the merged dataset is imbalanced by source; `rf_waste_sorting` has severe class imbalance and single-class sources are not valid standalone classifiers.
- Raw source folders were downloaded again: `rf_taco_trash`, `rf_garbage_cls`, `rf_food_waste`, `rf_trash_detection`, `rf_cigarettes`.
- The current Roboflow TACO-style raw folder has 12 classes; official TACO 60-class work would need a COCO-to-YOLO conversion step.
- External TACO-like dataset registry now covers TACO official, MJU-Waste, ZeroWaste, TrashCan, TrashNet, and GINI.
- External ML now has per-dataset results: TrashNet XGBoost accuracy `0.8103`; official TACO partial XGBoost accuracy `0.1399`, proving TACO-style raw 60-class data is much harder and needs mapping/capping.
- External-only run: TrashNet ML `0.8103`, GINI binary ML `0.8837`, TACO partial ML `0.0705`; external-only CNN: TrashNet `0.4292`, GINI binary `0.7895`, TACO partial `0.0586`.
- Balanced runs are now available: merged 6-class ML XGBoost accuracy `0.6497`; fair merged 6-class CNN accuracy `0.4614`; YOLO was not run.
- Phase 2 audit is saved at `docs\02_dataset_training\PHASE2_EXPERIMENT_AUDIT.md`; master comparison CSV is saved at `runs\comparisons\phase2_model_comparison.csv`.
- Tuned YOLOv8n: test mAP@0.5 `0.810`, test mAP@0.5:0.95 `0.622`.
- Current lecture focus: explain features + ML, pause YOLO discussion unless asked, then show ANN/CNN baselines and saved logs.

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
.\.venv311\Scripts\python.exe scripts\analyze_dataset_sources.py --data merged_dataset_v3\data.yaml --out runs\source_analysis\merged_dataset_v3 --max-feature-objects-per-source 1500
.\.venv311\Scripts\python.exe scripts\download_datasets.py
.\.venv311\Scripts\python.exe scripts\download_extra.py
.\.venv311\Scripts\python.exe scripts\summarize_external_dataset_registry.py
.\.venv311\Scripts\python.exe scripts\analyze_dataset_sources.py --data rf_taco_trash\data.yaml --out runs\raw_source_analysis\rf_taco_trash --max-feature-objects-per-source 1500
.\.venv311\Scripts\python.exe scripts\prepare_classification_dataset.py --source external_datasets\trashnet\data\dataset-resized --out prepared_datasets\trashnet --overwrite
.\.venv311\Scripts\python.exe scripts\prepare_coco_dataset.py --annotations external_datasets\taco_official\data\annotations.json --images-root external_datasets\taco_official\data --out prepared_datasets\taco_official_partial --overwrite
.\.venv311\Scripts\python.exe scripts\feature_ml_analysis.py --data prepared_datasets\trashnet\data.yaml --out runs\ml\by_dataset\trashnet --exclude-classes= --max-per-class-train 400 --max-per-class-test 120 --domain-out runs\ml\features_by_dataset\trashnet
.\.venv311\Scripts\python.exe scripts\feature_ml_analysis.py --data merged_dataset_v3\data.yaml --out runs\ml\feature_ml_lecturer_6class_4k --exclude-classes other --max-per-class-train 4000 --max-per-class-test 800 --domain-out ml\frequency_analysis
.\.venv311\Scripts\python.exe scripts\feature_ml_analysis.py --data merged_dataset_v3\data.yaml --out runs\ml\balanced_by_dataset\merged_6class --exclude-classes other --max-per-class-train 1000 --max-per-class-test 250 --domain-out runs\ml\features_by_dataset\merged_6class
.\.venv311\Scripts\python.exe scripts\deep_learning_baseline.py --data merged_dataset_v3\data.yaml --out runs\dl\ann_cnn_merged_dataset_v3 --model both --max-train-objects 8000 --max-val-objects 2000 --max-test-objects 3000 --epochs 8
.\.venv311\Scripts\python.exe scripts\deep_learning_baseline.py --data merged_dataset_v3\data.yaml --out runs\dl\balanced_ann_cnn\merged_6class --model both --source-filter rf_taco_trash,rf_garbage_cls,rf_waste_sorting,rf_uca_recyclable --max-per-class-train 1000 --max-per-class-val 250 --max-per-class-test 250 --epochs 5
.\.venv311\Scripts\python.exe scripts\deep_learning_baseline.py --data merged_dataset_v3\data.yaml --out runs\dl\balanced_ann_cnn\merged_6class_fair --model both --source-filter rf_taco_trash,rf_garbage_cls,rf_waste_sorting,rf_uca_recyclable,rf_food_waste --max-per-class-train 1000 --max-per-class-val 250 --max-per-class-test 250 --epochs 5 --batch-size 64 --image-size 64
.\.venv311\Scripts\python.exe scripts\deep_learning_baseline.py --data tuned_dataset_v1\data.yaml --out runs\dl\ann_cnn_tuned_dataset_v1 --model both --max-train-objects 8000 --max-val-objects 2000 --max-test-objects 3000 --epochs 8
.\.venv311\Scripts\python.exe scripts\deep_learning_baseline.py --data merged_dataset_v3\data.yaml --out runs\dl\source_ann_cnn\rf_taco_trash --model both --source-filter rf_taco_trash --max-train-objects 3000 --max-val-objects 800 --max-test-objects 800 --epochs 5
.\.venv311\Scripts\python.exe scripts\deep_learning_baseline.py --data rf_taco_trash\data.yaml --out runs\dl\raw_ann_cnn\rf_taco_trash --model both --max-train-objects 3000 --max-val-objects 800 --max-test-objects 800 --epochs 5 --batch-size 64
.\.venv311\Scripts\python.exe scripts\evaluate.py --weights runs\dl\trash_yolov8n_tuned_v1\weights\best.pt --data tuned_dataset_v1\data.yaml --out runs\dl\trash_yolov8n_tuned_v1\quality_check --split both
.\.venv311\Scripts\python.exe scripts\predict_images.py --source assets\manual_test_images --weights runs\dl\trash_yolov8n_tuned_v1\weights\best.pt --conf 0.10
```
