# Tile-augmented training vs baseline (clean val + test)

Same recipe as the deployed baseline (100 epochs, imgsz 640, batch 16, cos_lr,
fresh from pretrained yolo26n) - only the training data differs. Baseline =
`runs/detect/yolo26n_hardcase_v2_long/weights/best.pt`. Tiled = every training
image kept plus 2x2 overlapping tile crops of images containing tiny GT boxes
(IoS-remapped boxes), ranked/capped toward tiny-organic content
(`scripts/build_tiled_training_dataset.py`). Both evaluated identically via
`scripts/eval_detector_sizeclass.py` on `external_datasets/yolo26_hardcase_clean_eval`.

| split | model | Precision | Recall | mAP50 | mAP50-95 | organic R | small-box R (n) |
|---|---|---:|---:|---:|---:|---:|---:|
| val | baseline | 0.777 | 0.650 | 0.721 | 0.542 | 0.359 | 0.208 (6746) |
| val | tiled | 0.782 | 0.648 | 0.714 | 0.538 | 0.335 | 0.192 (6746) |
| test | baseline | 0.590 | 0.456 | 0.474 | 0.348 | 0.109 | 0.275 (611) |
| test | tiled | 0.571 | 0.460 | 0.449 | 0.335 | 0.022 | 0.270 (611) |

**Verdict: tiled loses.** On clean test - the deployment-honest split - tiled is
worse on mAP50 (0.449 vs 0.474), mAP50-95 (0.335 vs 0.348), and small-box recall
(0.270 vs 0.275), and organic recall collapses (0.109 -> 0.022, a 5x drop). Val
tells the same story at smaller magnitude. Tiling made small objects easier to
see in isolated crops but taught the model a training-distribution that
generalizes worse to real (untiled) test images - most likely because tiles
change scale/context statistics the model doesn't see at inference. Baseline
weights were NOT touched; tiled weights are not deployed.

Raw JSON: `runs/audits/detector_sizeclass_baseline_v2long.json`,
`runs/audits/detector_sizeclass_tiled_v1.json`.
