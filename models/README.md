# Models

This folder stores stable model artifacts copied out of experiment runs.

| Path | Purpose |
|---|---|
| `pretrained/yolo11n.pt` | Base YOLO model. |
| `pretrained/yolo26n.pt` | Base YOLO26 model used for newer hard-case localization training. |
| `trained/yolov11_detector/best.pt` | Current trained localization model copy (promoted in place: now the YOLO26m hard-case detector; directory name is historical). `best_before_*.pt` are pre-promotion backups. |
| `trained/efficientnet_classifier/best_efficientnet_tuned.h5` | Current trained classifier copy. |
| `trained/comparison_baselines/` | Baseline classifier models used for architecture comparison. |

Training logs and experiment-specific outputs remain under `runs/`.
