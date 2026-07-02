# Failures, Root Causes, and Fixes — Engineering Log for the Final Report

Honest record of everything that went wrong in the WasteWise pipeline, why it went
wrong, how it was detected, and what fixed it. Each entry is thesis-usable evidence
(limitations / lessons-learned sections). Dates: June–July 2026.

## F1. Evaluation data leakage across dataset splits (CRITICAL)

- **Symptom:** none visible — metrics looked good. Found by a systematic risk audit.
- **Root cause:** the merged datasets combine the same source photos arriving through
  different distributions (TACO-official and Roboflow re-exports of TACO). Roboflow
  re-encodes/resizes images, so byte-level (MD5) deduplication cannot match them, and
  the same photograph landed in `train` under one name and `val`/`test` under another.
  For the classifier dataset the exact-duplicate cleanup was never run at all.
- **Measured impact** (`scripts/audit_model_risks.py`, MD5 + perceptual dHash Hamming<=4,
  visually verified in `runs/audits/near_dup_spotcheck.jpg`):
  - `yolo26_hardcase_dataset_v1`: 823/3,450 val (23.9%) and 376/1,275 test (29.5%)
    images duplicate another split.
  - `super_yolo_dataset`: 487 val + 190 test near-duplicates of train.
  - `merged_dataset_v5` (classifier): **1,718 byte-exact + ~1,196 near = 2,406/5,600
    test images (43%) leaked** → the EfficientNetB0 94.3% test accuracy is inflated.
- **Fix:** `scripts/build_clean_eval_splits.py` quarantines every leaked eval image
  (originals untouched) into clean eval sets: `external_datasets/yolo26_hardcase_clean_eval`
  (val 2,627 / test 899) and `data/merged_dataset_v5_clean_test` (test 3,194).
  All reported metrics must be re-measured on these. Manifest:
  `runs/audits/quarantine_manifest.csv`.
- **Measured correction (classifier):** EfficientNetB0 on original test 94.30% ->
  **91.77% on the clean test** (macro-F1 0.9431 -> 0.9114); leakage inflated accuracy
  by ~2.5pp. Largest per-class drops: paper F1 0.93->0.86, cardboard 0.94->0.89,
  metal 0.92->0.88. Evidence: `runs/audits/CLASSIFIER_CLEAN_TEST_EVAL.md`.
- **Measured correction (detector):** clean-split validation of the retrained
  YOLO26n gives mAP50 0.721 / P 0.777 / R 0.650 on clean val, slightly ABOVE the
  original val (0.707) - the quarantined images were hard TACO scenes, so the
  detector eval was not inflated. The source-aware test split (unseen TACO
  capture batches) is much harder: mAP50 0.474 - honest out-of-sample evidence
  of batch-level domain shift. Evidence: `runs/audits/DETECTOR_CLEAN_VAL.md`.
- **Lesson:** content-level (perceptual) deduplication is mandatory when merging
  overlapping community datasets; filename- and byte-level checks are insufficient.

## F2. Cross-dataset validation was broken (reported 17.83%)

- **Symptom:** domain-generalization experiment reported 17.83% accuracy with 4 of 7
  classes missing from the test split (zero support).
- **Root cause:** `is_trashnet_file()` filename heuristic only recognized some material
  names, silently routing plastic/metal/paper/cardboard into one domain. The script
  also referenced dead code paths.
- **Fix:** rewritten (`scripts/cross_dataset_validation.py`, commit 23582d2) using the
  acquisition-domain filename markers (`_train_/_test_/_val_/rf_` = FIELD real-world,
  else STUDIO lab), evaluated in both directions with all 7 classes.
- **Honest result:** STUDIO->FIELD 39.8% (in-domain 80.0%, gap 40.2pp);
  FIELD->STUDIO 44.5% (in-domain 72.8%, gap 28.3pp). The large gap is a *finding*,
  not a bug: 637-D handcrafted features overfit the acquisition domain.

## F3. PCA dimensionality report cherry-picked its headline number

- **Symptom:** report claimed "637->128 costs ~2%" citing Linear SVM (−2.52pp) while
  the *best* models (ExtraTrees/XGBoost) lost 5.7–9pp at 128 components.
- **Root cause:** headline selected the model closest to a pre-chosen 2pp narrative.
- **Fix:** sweep widened to 9 dimensions (16–637) and the report now leads with
  best-model-per-dimension plus the smallest dimension within 1pp/2pp of the
  full-feature ceiling (commit 332bc0f).

## F4. Detector recall limited by tiny objects, not classification

- **Symptom:** recall 0.59; per-class miss rates 24–53%.
- **Diagnosis:** confusion matrix shows cross-class errors <=11%; the failure mode is
  objects predicted as background. Miss rate tracks *box size*, not class frequency:
  organic is the largest class (39.7% of 80,993 boxes) yet worst detected (53% missed)
  because its median object covers 1.36% of the image (~74 px at 640) and 42.5% of its
  boxes are under 1% of image area. Background false-positives skew organic (model
  learned "small blob = organic").
- **Fixes:** (a) longer training - the 30-epoch run had decayed LR to 4.3e-5 while
  metrics were still climbing each epoch; retrained 100 epochs with cosine LR
  (`scripts/train_hardcase_long.py`); (b) planned: higher-resolution run
  (imgsz 960, batch 8 - VRAM verified to fit) as the primary tiny-object lever.

## F5. Detector precision/recall operating point was never tuned

- **Symptom:** web app used default confidence; precision complaints.
- **Fact:** mAP is threshold-independent, but deployed P/R are set by `conf`.
  Sweep (`scripts/yolo_precision_threshold_sweep.py`, commit 5794abb):
  P 0.759@conf0.001 -> 0.861@0.5 -> 0.922@0.7 (recall 0.592 -> 0.517 -> 0.424).
  High-precision deployments should run conf~0.5 instead of retraining.

## F6. Label/data hygiene issues (minor, quantified)

- ~27 corrupt JPEGs in hardcase val auto-restored by ultralytics on first read.
- Detect/segment mixed labels: 606 polygon rows among 20,989 boxes (ultralytics
  drops the polygons with a warning); source located via audit label scan.
- 0 undecodable images, 0 out-of-bounds boxes after `optimize_dataset.py` clamping;
  micro-boxes <12px pruned at dataset build.

## F7. Engineering/environment failures (cost: ~1 day of compute)

1. **Two Python environments**: the shell's bare `python` is a global CPU-only torch
   install; the real training env is `.venv311` (torch 2.4.1+cu121, CUDA available).
   Training scripts run with the wrong interpreter refuse to start (or silently run
   CPU-only elsewhere). Rule: always `.venv311\Scripts\python.exe` for GPU work.
2. **Disk-full cascade**: an unnecessary attempt to install CUDA torch into the
   *global* env hit a 100%-full C: drive, failed mid-install and left torch broken
   (orphaned `caffe2_nvrtc.dll`); required full purge + reinstall. ~30 GB was later
   reclaimed (pip cache, partial installs, unused 2.1 GB `outerview` dataset pool).
3. **CUDA OOM at batch 32**: yolo26n @ imgsz 640 batch 32 OOMs the RTX 3060 12 GB in
   the pin-memory stage; batch 16 + workers 4 is the box's stable config.
4. **Windows multiprocessing hang**: launching `model.train()` via `python -c "..."`
   deadlocks before any output — spawn-based DataLoader workers need a real script
   with an `if __name__ == "__main__"` guard. Burned ~2.4 CPU-hours producing nothing
   and stacked zombie process trees. Training must live in a `.py` file
   (`scripts/train_hardcase_long.py`).
5. **RAM/disk cache limits**: `cache=True` (RAM) needs ~24 GB for the 20k-image set
   (box has 17 GB); `cache='disk'` would refill the nearly-full drive. `cache=False`
   is the only safe mode on this machine.

## F8. Marginal architecture swap (YOLOv11 -> YOLO26n)

- Benchmark (`runs/detect/yolo11_vs_yolo26_benchmark.json`): mAP50 0.669 -> 0.672,
  precision 0.745 -> 0.757, recall ~equal, latency ~equal. The swap alone did not
  move the needle; convergence (F4) and data quality (F1) dominate.

## Status summary

| Failure | Severity | Status |
|---|---|---|
| F1 eval leakage | Critical | Quarantined eval sets built; re-validation pending training completion |
| F2 cross-dataset broken | High | Fixed + committed |
| F3 PCA cherry-pick | Medium | Fixed + committed |
| F4 tiny-object recall | High | Retrain running; imgsz-960 run planned |
| F5 untuned conf | Medium | Sweep committed; deploy conf decision pending |
| F6 label hygiene | Low | Quantified; no action needed |
| F7 environment | Medium | All root-caused; rules recorded |
| F8 architecture swap | Info | Documented |
