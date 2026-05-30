# Runs

This folder stores experiment evidence and generated outputs.

## Current Evidence

| Path | Purpose |
|---|---|
| `detect/yolov11_super_dataset/` | YOLOv11 detector training run and active weights. |
| `detect/yolov11_super_dataset_validation_plots/` | YOLO validation plots. |
| `dl/localization_rework/yolo_conf035_stratified60_final/` | Current improved Stage 2 localization result. |
| `dl/localization_rework/yolo_conf025_stratified60/` | Stage 2 localization ablation result. |
| `dl/localization_rework/gradcam_baseline_stratified60/` | Stage 2 Grad-CAM baseline. |
| `dl/pca_experiments/` | PCA dimensionality sweep artifacts. |
| `dl/comparison_models/` | DL architecture comparison. |
| `ml/feature_ml_lecturer_6class_4k/` | Main 637-feature classical ML evidence. |
| `comparisons/model_comparison/` | ML vs DL comparison report/chart. |

## Legacy Evidence Kept In Place

Some older experiment folders remain here because scripts still reference them directly. Treat them as historical evidence, not the final project direction.

| Path | Note |
|---|---|
| `detect/yolo_efficientnet_pipeline/` | Old YOLO-first, EfficientNet-second pipeline outputs. |
| `dl/cnn_efficientnet/` | Earlier EfficientNet training/export artifacts. |
| `dl/convnext_ensemble*/` | Older ConvNeXt ensemble experiments referenced by legacy scripts. |

Old demos, raw manual-test outputs, scratch files, and redundant localization trials were moved to `_archive/legacy_cleanup_2026_05_30/`.
