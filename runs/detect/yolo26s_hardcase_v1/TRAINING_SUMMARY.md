# YOLO26s Training Summary Report

Completed training YOLO26s on the hard-case dataset.

## Summary Table

| Metric | Best Epoch (80) | Final Epoch (100) |
| :--- | :---: | :---: |
| **Precision** | 0.7942 | 0.8069 |
| **Recall** | 0.6805 | 0.6721 |
| **mAP@0.5** | **0.7446** | 0.7403 |
| **mAP@0.5:0.95** | 0.5766 | 0.5738 |
| **Train Box Loss** | 1.0067 | 0.9283 |
| **Train Class Loss** | 0.7544 | 0.6036 |
| **Val Box Loss** | 1.2384 | 1.2421 |
| **Val Class Loss** | 1.2033 | 1.2400 |

## Recent Training History (Last 5 Epochs)

| Epoch | Train Box Loss | Train Cls Loss | Precision | Recall | mAP50 | mAP50-95 |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 96 | 0.9350 | 0.6020 | 0.8014 | 0.6768 | 0.7415 | 0.5746 |
| 97 | 0.9336 | 0.6002 | 0.8015 | 0.6768 | 0.7417 | 0.5746 |
| 98 | 0.9293 | 0.6023 | 0.8045 | 0.6754 | 0.7414 | 0.5746 |
| 99 | 0.9352 | 0.6044 | 0.8048 | 0.6751 | 0.7407 | 0.5744 |
| 100 | 0.9283 | 0.6036 | 0.8069 | 0.6721 | 0.7403 | 0.5738 |

*Log outputs saved to `runs/detect/yolo26s_hardcase_v1/monitor_run.log`*
