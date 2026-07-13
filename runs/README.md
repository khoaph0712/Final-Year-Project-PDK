# Runs

This folder stores experiment evidence and generated outputs.

## Current Evidence

| Path | Purpose |
|---|---|
| `detect/yolov11_super_dataset/` | YOLOv11 detector training run and active weights. |
| `detect/yolov11_super_dataset_validation_plots/` | YOLO validation plots. |
| `dl/localization_rework/yolo_conf030_stratified300_final/` | Current improved Stage 2 localization result; promoted balanced threshold. |
| `dl/localization_rework/THRESHOLD_SWEEP_300.md` | 300-image threshold sweep comparing conf=0.30, 0.35, and 0.40. |
| `dl/localization_rework/yolo_conf035_stratified300_final/` | Stage 2 threshold sweep evidence; balanced precision setting. |
| `dl/localization_rework/yolo_conf040_stratified300_sweep/` | Stage 2 threshold sweep evidence; high-precision setting. |
| `dl/localization_rework/yolo_conf035_stratified60_final/` | Earlier 60-image quick check. |
| `dl/localization_rework/yolo_conf025_stratified60/` | Stage 2 localization ablation result. |
| `dl/localization_rework/gradcam_baseline_stratified60/` | Stage 2 Grad-CAM baseline. |
| `ml/pca_feature_model_sweep/` | Controlled PCA sweep across classical ML models; use for the 637-to-128 about-2% claim. |
| `dl/pca_experiments/` | Legacy ANN-only PCA dimensionality sweep artifacts. |
| `dl/comparison_models/` | DL architecture comparison. |
| `ml/feature_ml_super_yolo_6class_4k/` | Current newest-dataset 637-feature classical ML rerun. |
| `ml/feature_ml_lecturer_6class_4k/` | Legacy lecturer-facing 637-feature classical ML evidence. |
| `comparisons/model_comparison/` | ML vs DL comparison report/chart. |

## Legacy Evidence Kept In Place

Some older experiment folders remain here because scripts still reference them directly. Treat them as historical evidence, not the final project direction.

| Path | Note |
|---|---|
| `detect/yolo_efficientnet_pipeline/` | Old YOLO-first, EfficientNet-second pipeline outputs. |
| `dl/cnn_efficientnet/` | Earlier EfficientNet training/export artifacts. |
| `dl/convnext_ensemble*/` | Older ConvNeXt ensemble experiments referenced by legacy scripts. |

Old demos, raw manual-test outputs, scratch files, and redundant localization trials were reviewed and permanently deleted after cleanup.
