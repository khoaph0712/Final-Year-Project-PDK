# Detector: original vs leakage-quarantined eval (C:\kaggle\working\runs\yolo26m_hardcase_v1\weights\last.pt @ imgsz 640)

| eval set | split | Precision | Recall | mAP50 | mAP50-95 |
|---|---|---:|---:|---:|---:|
| original | val | 0.834 | 0.685 | 0.757 | 0.586 |
| original | test | 0.622 | 0.529 | 0.529 | 0.417 |
| clean | val | 0.834 | 0.672 | 0.749 | 0.570 |
| clean | test | 0.605 | 0.479 | 0.482 | 0.366 |

The clean rows exclude eval images that near-duplicate training images (see runs/audits/quarantine_manifest.csv); they are the honest deployment numbers.