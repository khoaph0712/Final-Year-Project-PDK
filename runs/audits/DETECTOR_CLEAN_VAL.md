# Detector: original vs leakage-quarantined eval (yolo26n_hardcase_v2_long best.pt)

| eval set | split | Precision | Recall | mAP50 | mAP50-95 |
|---|---|---:|---:|---:|---:|
| original | val | 0.788 | 0.634 | 0.707 | 0.536 |
| original | test | 0.603 | 0.428 | 0.461 | 0.353 |
| clean | val | 0.777 | 0.650 | 0.721 | 0.542 |
| clean | test | 0.590 | 0.456 | 0.474 | 0.348 |

The clean rows exclude eval images that near-duplicate training images (see runs/audits/quarantine_manifest.csv); they are the honest deployment numbers.