# Detector: original vs leakage-quarantined eval (runs\detect\yolo26s_hardcase_v1\weights\best.pt @ imgsz 640)

| eval set | split | Precision | Recall | mAP50 | mAP50-95 |
|---|---|---:|---:|---:|---:|
| original | val | 0.789 | 0.684 | 0.744 | 0.576 |
| original | test | 0.599 | 0.502 | 0.505 | 0.390 |
| clean | val | 0.781 | 0.690 | 0.748 | 0.572 |
| clean | test | 0.690 | 0.470 | 0.502 | 0.374 |

The clean rows exclude eval images that near-duplicate training images (see runs/audits/quarantine_manifest.csv); they are the honest deployment numbers.