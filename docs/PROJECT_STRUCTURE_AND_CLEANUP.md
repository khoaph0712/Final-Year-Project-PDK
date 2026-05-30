# Project Structure And Cleanup Notes

Last updated: 2026-05-30

This workspace now separates active project material from legacy experiments and temporary outputs. The cleanup keeps current datasets, models, scripts, reports, and mobile code in place. Old demos, rejected image sets, scratch scripts, and redundant Stage 2 localization trial runs are moved to `_archive/legacy_cleanup_2026_05_30/` instead of being permanently deleted.

## Active Top-Level Folders

| Folder | Purpose | Keep / Notes |
|---|---|---|
| `data/` | Current classification dataset and small demo inputs. | Active. Main classification dataset is `data/merged_dataset_v5`. |
| `external_datasets/` | Current YOLO localization dataset. | Active. Main localization dataset is `external_datasets/super_yolo_dataset`. |
| `models/` | Stable model artifacts copied out of experiment runs. | Active deliverables for app/report use. |
| `scripts/` | Active training, evaluation, reporting, and utility scripts. | Active. Older scripts already live under `scripts/archive/`. |
| `runs/` | Experiment outputs and final evidence artifacts. | Active results stay here; old temporary outputs are archived. |
| `docs/` | Final report, workflow notes, dataset/training notes, lecturer notes. | Active reporting source. |
| `assets/` | Curated/manual test images used for demos and report evidence. | Active curated assets only. Raw/rejected internet sets are archived. |
| `mobile/` | React Native / Expo mobile application. | Active app code. |
| `.venv311/` | Local Python environment. | Local dependency folder; do not commit. |
| `_archive/` | Local quarantine for old/unrelated material removed from active project view. | Ignored by git. Review before permanent deletion. |

## Canonical Current Datasets

| Dataset | Path | Role |
|---|---|---|
| YOLO localization dataset | `external_datasets/super_yolo_dataset` | Stage 2 localization evidence, 6 classes. |
| Classification dataset | `data/merged_dataset_v5` | Stage 1 classification evidence, 7 classes including Background. |

## Key Result Folders

| Result | Path |
|---|---|
| Final Stage 2 localization, YOLO conf=0.35 | `runs/dl/localization_rework/yolo_conf035_stratified60_final` |
| Stage 2 localization ablation, YOLO conf=0.25 | `runs/dl/localization_rework/yolo_conf025_stratified60` |
| Stage 2 Grad-CAM baseline | `runs/dl/localization_rework/gradcam_baseline_stratified60` |
| PCA dimensionality sweep | `runs/dl/pca_experiments` |
| DL architecture comparison | `runs/dl/comparison_models` |
| ML model comparison | `runs/ml/feature_ml_lecturer_6class_4k` |
| ML vs DL comparison | `runs/comparisons/model_comparison` |
| YOLOv11 detector training | `runs/detect/yolov11_super_dataset` |
| YOLOv11 validation plots | `runs/detect/yolov11_super_dataset_validation_plots` |

## Archived During Cleanup

These were moved out of the active project tree:

| Old location | Archive location | Reason |
|---|---|---|
| `scratch/` | `_archive/legacy_cleanup_2026_05_30/scratch` | One-off debug and search scripts. |
| `assets/internet_test_images/` | `_archive/legacy_cleanup_2026_05_30/assets/internet_test_images_raw` | Raw internet image collection; curated set remains active. |
| `assets/internet_test_images_rejected/` | `_archive/legacy_cleanup_2026_05_30/assets/internet_test_images_rejected` | Rejected images, not active evidence. |
| `runs/detect/demo_beach_and_grass/` | `_archive/legacy_cleanup_2026_05_30/runs/detect/demo_beach_and_grass` | Old demo output folder. |
| `runs/manual_tests/` | `_archive/legacy_cleanup_2026_05_30/runs/manual_tests` | Old manual prediction outputs. |
| `runs/dl/classification_to_localization/` | `_archive/legacy_cleanup_2026_05_30/runs/dl/stage2_localization_first_trial` | Superseded first localization trial. |
| `runs/dl/classification_to_localization_yolo_smoke/` | `_archive/legacy_cleanup_2026_05_30/runs/dl/stage2_localization_yolo_smoke` | Smoke-test output. |
| `runs/dl/classification_to_localization_yolo_stratified60/` | `_archive/legacy_cleanup_2026_05_30/runs/dl/stage2_localization_yolo_conf025_early` | Superseded by clearer ablation folder. |

Also moved `docs/01_final_report/FINAL_PROJECT_PIPELINE_REPORT.md` to `docs/99_legacy_reports/FINAL_PROJECT_PIPELINE_REPORT_legacy_merged_dataset_v3.md` because it describes `merged_dataset_v3` as final. The current final tracking report is `docs/01_final_report/WasteWise_Project_Tracking_Report.docx`.

## Removed During Cleanup

Only empty or generated cache folders are permanently removed:

- `.antigravitycli/`
- `data/external_datasets/`
- `data/convnext_training_crops/`
- `docs/01_final_report/rendered_tracking_report/`
- `scripts/__pycache__/`
- `scripts/archive/__pycache__/`
- `runs/dl/localization_rework/yolo_conf035_stratified60_final/visuals/` (empty)
- `runs/raw_source_analysis/rf_trash_detection/features_by_source/` (empty)
- `mobile/android/.gradle/`
- `mobile/android/.idea/`

## Cleanup Command

The organization pass is scripted and guarded so paths cannot escape `C:\FYP`:

```powershell
.\scripts\organize_project_workspace.ps1
```

Preview without moving/removing:

```powershell
.\scripts\organize_project_workspace.ps1 -WhatIfOnly
```
