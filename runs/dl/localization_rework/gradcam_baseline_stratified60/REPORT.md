# Classification-First Localization Report

This run implements the revised DL workflow: Stage 1 classification, Stage 2 localization from classifier evidence.

## Configuration

- Data: `C:\FYP\external_datasets\super_yolo_dataset\data.yaml`
- Split: `test`
- Model: `C:\FYP\models\trained\efficientnet_classifier\best_efficientnet_tuned.h5`
- Grad-CAM feature layer: `top_activation`
- Heatmap threshold: `0.45`
- IoU threshold: `0.5`
- Sample mode: `stratified`
- Seed: `42`

## Localization Metrics

| Metric | Value |
|---|---:|
| Images evaluated | 60 |
| Ground-truth boxes | 261 |
| Predicted boxes | 74 |
| True positives | 19 |
| False positives | 55 |
| False negatives | 242 |
| Precision | 0.2568 |
| Recall | 0.0728 |
| Mean matched IoU | 0.7127 |
| Classification gate hit-rate | 0.7667 |

## Notes

- Green boxes in visual outputs are ground truth.
- Red boxes are Grad-CAM localization predictions.
- Classification gate hit-rate is diagnostic only; final DL evaluation is localization-first.

## Artifacts

- `predictions.csv`
- `summary.json`
- `visuals/*.jpg`
