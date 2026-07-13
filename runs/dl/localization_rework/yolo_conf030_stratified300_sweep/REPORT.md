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
- YOLO confidence: `0.3`
- IoU threshold: `0.5`
- Sample mode: `stratified`
- Seed: `42`

## Localization Metrics

| Metric | Value |
|---|---:|
| Images evaluated | 300 |
| Ground-truth boxes | 1152 |
| Predicted boxes | 943 |
| True positives | 660 |
| False positives | 283 |
| False negatives | 492 |
| Precision | 0.6999 |
| Recall | 0.5729 |
| Mean matched IoU | 0.9057 |
| Classification gate hit-rate | 0.8533 |

## Notes

- Visual overlay uses a `YOLO objectness map`.
- Green boxes in visual outputs are ground truth.
- Red boxes are `YOLO localization boxes`.
- Classification gate hit-rate is diagnostic only; final DL evaluation is localization-first.

## Artifacts

- `predictions.csv`
- `summary.json`
- `visuals/*.jpg`
