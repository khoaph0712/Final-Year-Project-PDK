# WasteWise: Waste Classification and Localization

WasteWise is a Final Year Project for automated waste understanding. The project
combines a classical machine-learning branch with a redesigned deep-learning
branch for classification-first localization.

The current repository is organized around two active tracks:

- **ML track:** explainable handcrafted features, model comparison, and PCA
  compression experiments.
- **DL track:** image/classification gate first, localization second. YOLO is used
  as a localization module, not as the final class decision.

## Current Project Position

Earlier experiments used a YOLO-first two-stage pipeline:

```text
image -> YOLO localization -> crop -> EfficientNet classification
```

That pipeline is kept as experiment evidence, but it is no longer the main final
workflow. The current DL direction is:

```text
image -> classification/gate -> localization evidence -> boxes/heatmaps
```

This lets the DL branch report localization metrics directly: precision, recall,
IoU, and detection evidence quality.

## Active Datasets

| Dataset | Path | Role |
|---|---|---|
| Classification dataset | `data/merged_dataset_v5` | 7-class image/crop classification, including Background |
| YOLO localization dataset | `external_datasets/super_yolo_dataset` | 6-class localization evidence and box labels |

Large datasets are not expected to be fully tracked by Git. Keep local dataset
copies in the paths above.

## Main Results

### Classical ML Branch

The ML branch uses 637 handcrafted features per crop:

| Feature group | Count |
|---|---:|
| Spatial / edge features | 8 |
| FFT / frequency features | 9 |
| Color statistics and histograms | 44 |
| HOG descriptors | 576 |
| **Total** | **637** |

Saved lecturer-facing ML results:

| Model | Accuracy | F1-macro |
|---|---:|---:|
| XGBoost | 0.6742 | 0.6506 |
| ExtraTrees | 0.6312 | 0.6113 |
| Random Forest | 0.6317 | 0.6111 |
| Linear SVM | 0.5960 | 0.5642 |
| Logistic Regression | 0.5864 | 0.5558 |
| Decision Tree | 0.5115 | 0.4883 |

PCA evidence shows the handcrafted feature space can be compressed while keeping
most variance:

| Components | Explained variance | Accuracy | Weighted F1 | Latency |
|---:|---:|---:|---:|---:|
| 637 | 100.00% | 73.24% | 0.7319 | 0.0533 ms |
| 128 | 99.90% | 68.71% | 0.6863 | 0.0314 ms |
| 64 | 99.78% | 67.48% | 0.6736 | 0.0284 ms |

### Deep Learning Localization Branch

The current DL branch evaluates localization after a classifier/gate stage.
Quick-check localization results:

| Stage 2 localizer | Precision | Recall | Mean matched IoU | TP | FP | FN |
|---|---:|---:|---:|---:|---:|---:|
| Grad-CAM baseline | 0.2568 | 0.0728 | 0.7127 | 19 | 55 | 242 |
| YOLO localization-only, conf=0.25 | 0.6352 | 0.5670 | 0.9012 | 148 | 85 | 113 |
| YOLO localization-only, conf=0.35 | 0.7614 | 0.5134 | 0.9004 | 134 | 42 | 127 |

Recommended current setting:

```powershell
.\.venv311\Scripts\python.exe scripts\classification_to_localization_pipeline.py `
  --max-images 60 `
  --max-visuals 18 `
  --sample-mode stratified `
  --seed 42 `
  --localizer yolo `
  --yolo-conf 0.35 `
  --out-dir runs\dl\localization_rework\yolo_conf035_stratified60_final
```

## Repository Layout

```text
C:\FYP
|-- assets/                  Curated images for demos and evidence
|-- data/                    Classification datasets
|-- docs/                    Reports, workflow notes, and project tracking
|-- external_datasets/       YOLO-format localization datasets
|-- mobile/                  React Native / Expo mobile app
|-- models/                  Stable model artifacts for app/report use
|-- runs/                    Experiment outputs and evidence artifacts
|-- scripts/                 Training, evaluation, reporting, and cleanup scripts
|-- _archive/                Local legacy quarantine, ignored by Git
|-- requirements.txt         Python dependencies
`-- README.md                Project overview
```

More structure details:

- `docs/PROJECT_STRUCTURE_AND_CLEANUP.md`
- `docs/01_final_report/WORKFLOW_APPROACHES_AND_DL_REWORK.md`
- `docs/diagrams/wastewise_professional_pipelines.svg`

## Setup

Use Python 3.11.

```powershell
python -m venv .venv311
.\.venv311\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Common Commands

Train YOLO localization model:

```powershell
python scripts\train_super_yolo.py
```

Run current classification-to-localization evaluation:

```powershell
python scripts\classification_to_localization_pipeline.py --localizer yolo --yolo-conf 0.35
```

Train classical ML / ANN / CNN baselines:

```powershell
python scripts\train_ann.py
python scripts\train_cnn.py
python scripts\train_comparison_models.py
```

Export edge/mobile model formats:

```powershell
python scripts\export_tflite.py
python scripts\export_ensemble_onnx.py
```

Regenerate the project tracking document:

```powershell
python scripts\build_project_tracking_docx.py
```

Clean or reorganize workspace outputs:

```powershell
.\scripts\organize_project_workspace.ps1 -WhatIfOnly
.\scripts\organize_project_workspace.ps1
```

## Mobile App

The mobile app lives in `mobile/` and uses React Native / Expo.

```powershell
cd mobile
npm install
npm run android
```

Model files should be copied into the mobile asset location only when needed for
local testing or packaging.

## Large Artifacts

GitHub rejects files over 100 MB. Large model binaries such as `*.pth`, `*.pt`,
`*.onnx`, `*.h5`, `*.tflite`, datasets, and generated caches should stay local or
be handled through a release artifact / external storage workflow.

The repository intentionally ignores large model binaries and dataset folders to
keep Git history usable.

## Final Report Guidance

Use the ML branch as the explainable finalized pipeline. Use the DL branch as the
classification-to-localization rework, evaluated with localization metrics rather
than classification accuracy.

Current final tracking document:

```text
docs/01_final_report/WasteWise_Project_Tracking_Report.docx
```
