# Detector: original vs leakage-quarantined eval (runs\detect\yolo26n_hardcase_v3_960\weights\best.pt @ imgsz 960)

| eval set | split | Precision | Recall | mAP50 | mAP50-95 |
|---|---|---:|---:|---:|---:|
| original | val | 0.783 | 0.630 | 0.708 | 0.532 |
| original | test | 0.517 | 0.484 | 0.472 | 0.358 |
| clean | val | 0.768 | 0.639 | 0.718 | 0.533 |
| clean | test | 0.539 | 0.495 | 0.476 | 0.347 |

The clean rows exclude eval images that near-duplicate training images (see runs/audits/quarantine_manifest.csv); they are the honest deployment numbers.