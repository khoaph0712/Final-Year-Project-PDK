# Classification-First Localization Report

This run implements the revised DL workflow: Stage 1 classification, Stage 2 localization.

## Configuration

- Data: `C:\FYP\external_datasets\super_yolo_dataset\data.yaml`
- Split: `test`
- Model: `C:\FYP\models\trained\efficientnet_classifier\best_efficientnet_tuned.h5`
- Stage 2 localizer: `yolo`
- Grad-CAM feature layer: `top_activation`
- Heatmap threshold: `0.45`
- YOLO weights: `C:\FYP\models\trained\yolov11_detector\best.pt`
- YOLO confidence: `0.25`
- IoU threshold: `0.5`
- Sample mode: `stratified`
- Seed: `42`

## Localization Metrics

| Metric | Value |
|---|---:|
| Images evaluated | 60 |
| Ground-truth boxes | 261 |
| Predicted boxes | 233 |
| True positives | 148 |
| False positives | 85 |
| False negatives | 113 |
| Precision | 0.6352 |
| Recall | 0.5670 |
| Mean matched IoU | 0.9012 |
| Classification gate hit-rate | 0.7667 |

## Notes

- Visual overlay uses a `YOLO objectness map`.
- Green boxes in visual outputs are ground truth.
- Red boxes are `YOLO localization boxes`.
- Classification gate hit-rate is diagnostic only; final DL evaluation is localization-first.

## Artifacts

- `predictions.csv`
- `summary.json`
- `visuals/*.jpg`
