# Localization-First Crop-Verification Report

This run implements the 2-stage hierarchical DL workflow: YOLO localization first, followed by crop classification verification.

## Configuration

- Data: `C:\FYP\external_datasets\super_yolo_dataset\data.yaml`
- Split: `test`
- Model: `C:\FYP\models\trained\efficientnet_classifier\best_efficientnet_tuned.h5`
- Stage 2 localizer: `yolo`
- Grad-CAM feature layer: `top_activation`
- Heatmap threshold: `0.45`
- YOLO weights: `C:\FYP\models\trained\yolov11_detector\best.pt`
- YOLO confidence: `0.35`
- IoU threshold: `0.5`
- Sample mode: `stratified`
- Seed: `42`

## Localization Metrics

| Metric | Value |
|---|---:|
| Images evaluated | 20 |
| Ground-truth boxes | 89 |
| Predicted boxes | 45 |
| True positives | 35 |
| False positives | 10 |
| False negatives | 54 |
| Precision | 0.7778 |
| Recall | 0.3933 |
| Mean matched IoU | 0.8623 |
| Classification gate hit-rate | 0.6500 |

## Notes

- Visual overlay uses a `YOLO verified objectness map`.
- Green boxes in visual outputs are ground truth.
- Red boxes are `YOLO + verified crop boxes`.
- Classification gate hit-rate is diagnostic only; final DL evaluation is localization-first.

## Artifacts

- `predictions.csv`
- `summary.json`
- `visuals/*.jpg`
