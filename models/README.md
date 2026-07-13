# Models

This folder stores stable model artifacts copied out of experiment runs.

| Path | Purpose |
|---|---|
| `pretrained/yolo11n.pt` | Base YOLO model. |
| `pretrained/yolo26n.pt` | Base YOLO26 model used for newer hard-case localization training. |
| `trained/yolov11_detector/best.pt` | Current trained localization model copy. |
| `trained/efficientnet_classifier/best_efficientnet_tuned.h5` | Current trained classifier copy. |
| `trained/comparison_baselines/` | Baseline classifier models used for architecture comparison. |

Training logs and experiment-specific outputs remain under `runs/`.
