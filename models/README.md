# Models

This folder stores stable model artifacts copied out of experiment runs.

| Path | Purpose |
|---|---|
| `pretrained/yolo11n.pt` | Base YOLO model. |
| `trained/yolov11_detector/best.pt` | Current trained localization model copy. |
| `trained/efficientnet_classifier/best_efficientnet_tuned.h5` | Current trained classifier copy. |
| `trained/comparison_baselines/` | Baseline classifier models used for architecture comparison. |

Training logs and experiment-specific outputs remain under `runs/`.
