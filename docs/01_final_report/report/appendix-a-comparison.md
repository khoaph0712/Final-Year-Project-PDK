# Appendix A. Master Model Comparison

This appendix consolidates every model evaluated in the project into a single
table, so that selection decisions can be checked against the full field rather
than against the subset quoted in the section that made each decision.

## A.1 Detection Models

| Model | Config | Clean val mAP50 | Clean test mAP50 | Status | Source |
|---|---|---:|---:|---|---|
| YOLOv11 | baseline benchmark | 0.669 | — | superseded | `runs/detect/yolo11_vs_yolo26_benchmark.json` |
| YOLO26n | v2-long, 640 px | 0.721 | 0.474 | superseded | `runs/audits/detector_clean_val.json` |
| YOLO26n | v3, 960 px | 0.718 | 0.476 | rejected (F9) | `runs/audits/detector_clean_val_v3_960.json` |
| YOLO26n | tiled training | 0.449* | — | rejected (F4c) | `runs/audits/TILED_TRAINING_v1.md` |
| YOLO26s | check run | 0.743 | 0.491 | superseded | `runs/audits/detector_clean_val_yolo26s_check.json` |
| YOLO26s | 100 epochs | 0.748 | 0.502 | superseded | `runs/audits/detector_clean_val_yolo26s_final100.json` |
| **YOLO26m** | **100 ep, 640 px** | **0.749** | **0.482** | **deployed** | `runs/audits/detector_clean_val_yolo26m_final100.json` |

<!-- src: runs/audits/detector_clean_val*.json; runs/detect/yolo11_vs_yolo26_benchmark.json; runs/audits/TILED_TRAINING_v1.md -->

*Table A.1 — Detection models. \*The tiled figure is clean-test mAP50 against a
0.474 baseline; see §9.4. Note that YOLO26m is not uniformly best: it trails
YOLO26s by 2.0 points on clean test.*

## A.2 Controlled Cross-Architecture Comparison

Shared budget: 30 epochs, 512 px, batch 16, COCO-pretrained initialisation,
identical split, augmentation and seed. Only mAP50 and mAP50-95 are comparable
across the two implementation groups (§7.4).

| # | Model | Group | mAP50 | mAP50-95 | Train (h) | Notes |
|---|---|---|---:|---:|---:|---|
| 1 | YOLO26m | ultralytics | `TBD` | `TBD` | `TBD` | |
| 2 | YOLOv8m | ultralytics | `TBD` | `TBD` | `TBD` | |
| 3 | YOLO11m | ultralytics | `TBD` | `TBD` | `TBD` | |
| 4 | RT-DETR-l | ultralytics | `TBD` | `TBD` | `TBD` | AdamW @ lr0 1e-4 (disclosed deviation) |
| 5 | Faster R-CNN-R50-FPN | torchvision | `TBD` | `TBD` | `TBD` | P/R at fixed conf 0.5 |
| 6 | RetinaNet | torchvision | `TBD` | `TBD` | `TBD` | P/R at fixed conf 0.5 |
| 7 | FCOS | torchvision | `TBD` | `TBD` | `TBD` | P/R at fixed conf 0.5 |

<!-- BLOCKED: runs/detect/vast_comparison/ does not exist; fetch with scripts/vast/fetch_results.sh, then read each run's resolved optimizer from its args.yaml -->

*Table A.2 — Controlled comparison. Unfilled pending retrieval of the sweep.*

## A.3 Classification Models

| Model | Evaluation | Accuracy | Macro-F1 | Status | Source |
|---|---|---:|---:|---|---|
| ExtraTrees, 637-D handcrafted | classical test | 73.76% | 0.7381 | branch A best | `runs/ml/pca_feature_model_sweep/` |
| XGBoost, 637-D | merged 6-class balanced | 64.97% | 0.6422 | superseded | `runs/comparisons/phase2_model_comparison.csv` |
| ExtraTrees, PCA-128 | classical test | 67.81% | 0.6780 | superseded | `runs/ml/pca_feature_model_sweep/` |
| EfficientNetV2-S | stage 2 candidate | 80.23% | 0.7989 | rejected | `docs/01_final_report/stage2_backbone_reproduced_metrics.json` |
| Swin-Tiny | stage 2 candidate | 94.58% | 0.9464 | rejected on cost | `docs/01_final_report/stage2_backbone_reproduced_metrics.json` |
| ConvNeXtV2-Tiny | stage 2 candidate | 94.55% | 0.9459 | **selected** | `docs/01_final_report/stage2_backbone_reproduced_metrics.json` |
| EfficientNetB0 | merged_v5 clean test | 91.77% | 0.9114 | superseded | `runs/audits/classifier_clean_test_eval.json` |
| ConvNeXt ensemble | clean GT crops | 92.93% | 0.9290 | baseline | `runs/audits/convnext_clean_eval.json` |
| **ConvNeXt ensemble, FT** | **clean GT crops** | **93.77%** | — | **deployed** | `runs/dl/convnext_detector_crops_ft/finetune_result.json` |
| ConvNeXt ensemble, FT | **detector crops** | **88.88%** | 0.7989 | **deployed** | `runs/dl/convnext_detector_crops_ft/finetune_result.json` |

*Table A.3 — Classification models. Rows measure different datasets and splits;
compare only within an evaluation column.*

## A.4 Pipeline Configurations

All nine evaluated against the same 1,042 independent images.

| Configuration | Material acc | Bin acc | Status |
|---|---:|---:|---|
| **no_gate** | **94.53%** | **97.41%** | **deployed** |
| improved | 94.63% | 95.68% | rejected |
| margin20 | 94.53% | 96.16% | rejected (F16) |
| baseline | 94.43% | 96.26% | superseded |
| final3 | 94.43% | 96.26% | superseded |
| final | 94.43% | 95.97% | superseded |
| no_prior_damping | 94.43% | 95.97% | ablation |
| final2 | 94.43% | 94.63% | superseded |
| dominant_fix | 93.67% | 96.16% | rejected |
| *Constant "Recycling"* | — | *99.62%* | *trivial baseline* |

<!-- src: runs/audits/pipeline_bin_decision_eval_*.json -->

*Table A.4 — Pipeline configurations with the majority-class baseline. Per
§8.4.1, the bin-accuracy column is degenerate on this evaluation set and no
configuration beats the trivial baseline.*
