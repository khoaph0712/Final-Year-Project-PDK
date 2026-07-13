# Project Structure And Cleanup Notes

Last updated: 2026-06-12

This workspace now separates active project material from legacy experiments and temporary outputs. The cleanup keeps current datasets, models, scripts, reports, and mobile code in place. Old demos, rejected image sets, scratch scripts, and redundant Stage 2 localization trial runs were reviewed, moved out of the active tree, and then permanently deleted on request.

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
| `web/` | Flask-served web demo deployed to Hugging Face Spaces. | Active demo app code. |
| `.venv311/` | Local Python environment. | Local dependency folder; do not commit. |

## Canonical Current Datasets

| Dataset | Path | Role |
|---|---|---|
| YOLO localization dataset | `external_datasets/super_yolo_dataset` | Stage 2 localization evidence, 6 classes. |
| Classification dataset | `data/merged_dataset_v5` | Stage 1 classification evidence, 7 classes including Background. |
| Hard-case classifier dataset | `data/hard_case_classifier_v1` | Current hard-case Stage 1 classifier train/val/test export. |
| YOLO26 hard-case dataset | `external_datasets/yolo26_hardcase_dataset_v1` | Current hard-case YOLO26 localization training export. |
| Balanced real-world YOLO test dataset | `external_datasets/yolo26_balanced_realworld_v2` | Current real-life false-positive benchmark dataset. |

## Key Result Folders

| Result | Path |
|---|---|
| Final Stage 2 localization, YOLO conf=0.30 | `runs/dl/localization_rework/yolo_conf030_stratified300_final` |
| Stage 2 threshold sweep, 300 images | `runs/dl/localization_rework/THRESHOLD_SWEEP_300.md` |
| Stage 2 localization sweep, YOLO conf=0.35 | `runs/dl/localization_rework/yolo_conf035_stratified300_final` |
| Stage 2 localization sweep, YOLO conf=0.40 | `runs/dl/localization_rework/yolo_conf040_stratified300_sweep` |
| Stage 2 localization quick check, YOLO conf=0.35 | `runs/dl/localization_rework/yolo_conf035_stratified60_final` |
| Stage 2 localization ablation, YOLO conf=0.25 | `runs/dl/localization_rework/yolo_conf025_stratified60` |
| Stage 2 Grad-CAM baseline | `runs/dl/localization_rework/gradcam_baseline_stratified60` |
| Controlled PCA model sweep | `runs/ml/pca_feature_model_sweep` |
| Legacy ANN PCA dimensionality sweep | `runs/dl/pca_experiments` |
| DL architecture comparison | `runs/dl/comparison_models` |
| Current ML model comparison | `runs/ml/feature_ml_super_yolo_6class_4k` |
| ML model comparison | `runs/ml/feature_ml_lecturer_6class_4k` |
| ML vs DL comparison | `runs/comparisons/model_comparison` |
| YOLOv11 detector training | `runs/detect/yolov11_super_dataset` |
| YOLOv11 validation plots | `runs/detect/yolov11_super_dataset_validation_plots` |

## Permanently Deleted After Review

These were moved out of the active project tree during cleanup, reviewed, and then permanently deleted:

| Old location | Deleted material | Reason |
|---|---|---|
| `scratch/` | One-off debug and search scripts | Not active project code. |
| `assets/internet_test_images/` | Raw internet image collection | Curated set remains active. |
| `assets/internet_test_images_rejected/` | Rejected images | Not active evidence. |
| `runs/detect/demo_beach_and_grass/` | Old demo output folder | Superseded demo result. |
| `runs/manual_tests/` | Old manual prediction outputs | Superseded generated outputs. |
| `runs/dl/classification_to_localization/` | First localization trial | Superseded by final localization rework folder. |
| `runs/dl/classification_to_localization_yolo_smoke/` | Smoke-test output | Not final evidence. |
| `runs/dl/classification_to_localization_yolo_stratified60/` | Early YOLO conf=0.25 run | Superseded by clearer ablation folder. |

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

## 2026-06-12 Cleanup Additions

Removed unrelated or generated workspace clutter:

- `open-design/` separate repository accidentally inside the FYP workspace.
- `mobile/node_modules/` and `mobile/.expo/` generated mobile dependency/cache folders.
- `runs/detect/runs/` nested YOLO output folder duplicated under the wrong root.
- `runs/dl/convnextv2_material_stage1_smoke/` superseded smoke checkpoint.
- `runs/dl/convnextv2_material_stage1_candidate/` empty candidate folder.
- `external_datasets/yolo26_balanced_realworld_v1/` superseded by `external_datasets/yolo26_balanced_realworld_v2/`.
- `docs/figma_pipelines/` empty folder.
- `web/__pycache__/`, `scripts/__pycache__/`, and `scripts/archive/__pycache__/`.

Pretrained root clutter was normalized:

- Moved `yolo26n.pt` to `models/pretrained/yolo26n.pt`; future duplicate root copies are removed only when hashes match.
- Removed root `yolo11n.pt` only when it matched `models/pretrained/yolo11n.pt`.

## Cleanup Command

The organization pass is scripted and guarded so paths cannot escape `C:\FYP`:

```powershell
.\scripts\organize_project_workspace.ps1
```

Preview without moving/removing:

```powershell
.\scripts\organize_project_workspace.ps1 -WhatIfOnly
```
