# Detector: original vs leakage-quarantined eval (runs\detect\yolo26s_hardcase_v1\weights\best.pt @ imgsz 640)

| eval set | split | Precision | Recall | mAP50 | mAP50-95 |
|---|---|---:|---:|---:|---:|
| original | val | 0.805 | 0.658 | 0.737 | 0.567 |
| original | test | 0.558 | 0.488 | 0.482 | 0.375 |
| clean | val | 0.806 | 0.662 | 0.743 | 0.566 |
| clean | test | 0.567 | 0.495 | 0.491 | 0.369 |

The clean rows exclude eval images that near-duplicate training images (see runs/audits/quarantine_manifest.csv); they are the honest deployment numbers.