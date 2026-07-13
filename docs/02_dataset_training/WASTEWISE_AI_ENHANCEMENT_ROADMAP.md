# WasteWise AI Enhancement Roadmap

Last updated: 2026-06-18

## Objective

Improve the robustness, accuracy, and academic contribution of the WasteWise AI system while preserving the current deployment-ready pipeline:

- Hard-case ConvNeXt material classifier
- YOLO localizer
- Crop verification
- Conservative review routing for uncertain scans
- HTML/Python web demo and Expo mobile demo

The roadmap prioritizes evidence-backed optimization over architectural replacement.

## Current Baseline

| Component | Metric | Current result | Target |
|---|---:|---:|---:|
| ConvNeXt classifier | Test accuracy | 93.88% | 94-96% |
| ConvNeXt classifier | Test macro F1 | 93.98% | >=94% |
| YOLO localizer | Recall | 59.20% | >70% |
| YOLO localizer | mAP50 | 67.22% | >75% |
| Domain shift | TrashNet to TACO accuracy | 17.83% | Evidence only |

Key evidence:

- `runs/dl/convnext_hardcase_tuned/RESULT.json`
- `runs/detect/yolo11_vs_yolo26_benchmark.json`
- `runs/dl/cross_dataset_validation/Cross_Dataset_Report.md`
- `runs/dl/localization_rework/yolo_conf030_stratified300_final/summary.json`
- `runs/dl/localization_rework/gradcam_baseline_stratified60/REPORT.md`

## Phase Status

| Phase | Roadmap item | Status | Notes |
|---|---|---|---|
| 1 | Error analysis over 100+ samples | Partial | A 300-image localization rework exists, but failure categories still need a dissertation-ready summary table. |
| 1 | Hard example mining | Mostly done | RealWaste, TACO, and Outerview hard-case sources were downloaded and split. YOLO hard-case dataset v1 exists. |
| 1 | Augmentation ablation | Not complete | Needs controlled runs for mosaic, MixUp, copy-paste, rotation, scaling, and translation. |
| 2 | YOLO hyperparameter tuning | Partial | Confidence threshold sweeps exist; training hyperparameter ablation still needs a small controlled matrix. |
| 2 | YOLO26n/s/m comparison | Partial | YOLO26n was benchmarked against older YOLO11. YOLO26s and YOLO26m remain open. |
| 3 | RT-DETR benchmark | Not started | Highest academic-value detector comparison still missing. |
| 4 | Grad-CAM explainability | Partial | Grad-CAM visual outputs exist; final report should frame it as explainability, not a replacement localizer. |
| 5 | Cross-dataset generalization | Done | TrashNet-to-TACO result is severe domain shift at 17.83%, which is useful research evidence. |
| 6 | Confidence display | Done | Web result panel displays material confidence and possible material confidence. |
| 6 | Top-K predictions | Done | Web API now returns `topPredictions`; web UI renders a top-3 list plus all class bars. |
| 6 | Detection visualization | Done | Web UI renders multiple boxes with class/confidence labels and review states. |

## Highest-Impact Next Work

1. Produce a failure-analysis report from the existing 300-image localization rework.
   - Use `predictions.csv` and `summary.json`.
   - Categorize: missed detections, false positives, dense scenes, small objects, background confusion, and low-confidence review cases.
   - Output a table of counts, percentages, examples, and representative image paths.

2. Run a compact YOLO improvement matrix before trying larger models.
   - Keep the dataset fixed.
   - Vary only one major factor per run: image size, confidence, IoU, mosaic, MixUp, copy-paste, and epochs.
   - Use recall and mAP50 as promotion metrics.

3. Benchmark YOLO26s before YOLO26m.
   - YOLO26n did not improve recall enough over the older YOLO11 baseline.
   - YOLO26s is the next practical trade-off for CPU-bound deployment.
   - YOLO26m should be treated as research evidence unless latency remains acceptable.

4. Add RT-DETR as the comparative-study detector.
   - Train on the same fixed YOLO hard-case dataset.
   - Report precision, recall, mAP50, mAP50-95, and FPS.
   - Keep it separate from the production pipeline unless it wins both recall and practical inference speed.

5. Package Grad-CAM and cross-dataset results for the dissertation.
   - Grad-CAM supports explainability.
   - TrashNet-to-TACO supports domain-shift analysis.
   - Neither requires replacing the deployed system.

## Acceptance Rules

- Promote a classifier only if hard-case macro F1 and weak-class recall improve without increasing confident wrong routes.
- Promote a localizer only if recall and mAP50 improve on the fixed hard-case test split.
- Prefer higher recall for YOLO only when crop verification and review routing keep false positives manageable.
- Keep deployment readiness visible: report latency and model size with every detector comparison.
- Treat future open-vocabulary models such as Grounding DINO, Florence-2, LocateAnything-3B, and VLMs as future work unless they are benchmarked against the same data.

