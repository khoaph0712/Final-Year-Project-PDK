# YOLO Precision Threshold Sweep (no retrain)

- Weights: `runs\detect\yolo26n_hardcase_dataset_v1\weights\best.pt`
- Data: `external_datasets\yolo26_hardcase_dataset_v1\data.yaml` (split=val), NMS IoU=0.7
- Target precision: 0.85

| conf | Precision | Recall | F1 | mAP50 | mAP50-95 |
|---:|---:|---:|---:|---:|---:|
| 0.001 | 0.759 | 0.592 | 0.665 | 0.672 | 0.505 |
| 0.1 | 0.759 | 0.592 | 0.665 | 0.621 | 0.475 |
| 0.2 | 0.759 | 0.592 | 0.665 | 0.586 | 0.456 |
| 0.25 | 0.759 | 0.592 | 0.665 | 0.570 | 0.446 |
| 0.3 | 0.764 | 0.589 | 0.665 | 0.552 | 0.437 |
| 0.4 | 0.818 | 0.554 | 0.660 | 0.526 | 0.421 |
| 0.5 | 0.861 | 0.517 | 0.646 | 0.496 | 0.403 |
| 0.6 | 0.891 | 0.474 | 0.619 | 0.458 | 0.379 |
| 0.7 | 0.922 | 0.424 | 0.581 | 0.412 | 0.348 |

- **Best F1 operating point:** conf=0.3 (P=0.764, R=0.589).
- **Lowest conf reaching precision >= 0.85:** conf=0.5 (P=0.861, R=0.517) - use this to raise precision with minimal recall loss.