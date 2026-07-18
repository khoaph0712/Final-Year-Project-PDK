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
- **Measured correction (deployed ConvNeXt, hard_case_classifier_v1):** this
  dataset was audited last (2026-07-03): 545/2,696 test (20.2%) and 513/2,680 val
  (16.0%) images duplicate other splits. Re-evaluating the deployed ConvNeXt:
  original test 93.88% -> **92.93% on the clean test** (macro-F1 0.9398 -> 0.9290);
  clean val 92.02%. Inflation here was small (~1pp) - the deployed classifier is
  genuinely strong. Quote 92.93%. Evidence: `runs/audits/convnext_clean_eval.json`,
  clean split at `data/hard_case_classifier_v1_clean`.
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
  (`scripts/train_hardcase_long.py`); (b) tested: 960px fine-tune
  (`scripts/train_hardcase_960.py`, 40 epochs from the 640 best). Outcome: clean
  val a wash (mAP50 0.718 vs 0.721), clean TEST recall +3.9pp (0.456 -> 0.495) -
  resolution helps small objects on unseen batches but is NOT the binding
  constraint; training-data diversity is. 640 model kept deployed. Evidence:
  `runs/audits/DETECTOR_CLEAN_VAL_v3_960.md`.
- **Also tested: sliced (SAHI-style) inference** (`scripts/benchmark_sliced_inference.py`,
  800 clean-val images). Raw 2x2 slicing lifts tiny-object recall 0.266 -> 0.360
  but drops precision 0.801 -> 0.473; strict tile confidence and IoS fragment
  merging each recover some precision only by giving back the recall (best
  compromise: tiny 0.300 / precision 0.662 / other-recall -7.5pp). Not deployed.
  With both higher resolution AND sliced inference ruled out by experiment, the
  tiny-object limit is attributable to training data, not inference strategy.
  Evidence: `runs/audits/SLICED_INFERENCE_BENCHMARK.md`.

## F4b. Stage-2 train/serve skew (found and fixed 2026-07-03)

- **Symptom:** none visible in reports - the deployed classifier scored 92.93% on
  clean GT crops but only **76.91%** on the detector's own crops (what it actually
  receives in production): a 16pp train/serve skew.
- **Root cause:** the classifier trains on ground-truth crops; YOLO's crops have
  looser framing and detector-selected content (including false positives).
- **Fix:** built `data/detector_crops_v1` (detector predictions IoU-matched to GT
  labels, unmatched high-conf predictions kept as Background) and fine-tuned the
  deployed ConvNeXt on a detector-crop + GT-crop mixture. Detector-crop accuracy
  76.91% -> **88.88%** while clean GT test *improved* 92.93% -> **93.77%**. A
  detector-crops-only fine-tune was rejected by the no-forgetting gate (GT fell
  to 83%) - evidence that mixed-domain training is required.
- Scripts: `build_detector_crop_dataset.py`, `finetune_stage2_on_detector_crops.py`;
  results: `runs/dl/convnext_detector_crops_ft/finetune_result.json`.

## F4c. Tile-augmented training tested, rejected (2026-07-04)

- **Hypothesis:** F4 fixed convergence and data-diversity but tiny-object recall
  (small-box GT <1% image area) was still 0.208-0.275. If small objects appear
  larger in isolated tile crops during *training* (not just inference, which F4
  already ruled out via sliced inference), the model might learn them better.
- **Method:** `scripts/build_tiled_training_dataset.py` kept every original
  training image and additionally cut images containing tiny GT boxes into 2x2
  overlapping tiles (IoS-remapped boxes), capped to keep epoch time comparable.
  `scripts/train_tiled.py` used the identical recipe as the deployed baseline
  (100 epochs, imgsz 640, batch 16, cos_lr, fresh from pretrained yolo26n) so
  the training data was the only variable. Both baseline and tiled evaluated
  identically via `scripts/eval_detector_sizeclass.py` on the clean val+test.
- **Result: tiled loses on clean test (the deployment-honest split) on every
  metric** - mAP50 0.474 -> 0.449, mAP50-95 0.348 -> 0.335, small-box recall
  0.275 -> 0.270, and organic recall collapses 0.109 -> **0.022** (5x worse).
  Val shows the same direction at smaller magnitude. Training on tiles taught
  the model a scale/context distribution that generalizes worse to whole,
  untiled images at inference - the opposite of the intended fix.
- **Outcome:** tiled weights (`runs/detect/yolo26n_hardcase_tiled_v1/`) are
  **not deployed**; baseline (`yolo26n_hardcase_v2_long`) remains the deployed
  detector. Full comparison: `runs/audits/TILED_TRAINING_v1.md`.

## F5. Detector precision/recall operating point was never tuned

- **Symptom:** web app used default confidence; precision complaints.
- **Fact:** mAP is threshold-independent, but deployed P/R are set by `conf`.
  Sweep (`scripts/yolo_precision_threshold_sweep.py`, commit 5794abb):
  P 0.759@conf0.001 -> 0.861@0.5 -> 0.922@0.7 (recall 0.592 -> 0.517 -> 0.424).
  High-precision deployments should run conf~0.5 instead of retraining.
- **Update (2026-07-04), after F4c ruled out retraining:** re-swept on the
  deployed baseline against the clean val+test at the real serving imgsz (960,
  not 640) - `scripts/sweep_detector_conf_clean.py`,
  `runs/audits/DETECTOR_CONF_SWEEP.md`. Lowering `conf` 0.30 -> 0.10 nearly
  triples organic recall (test 0.087 -> 0.239) and lifts small-box recall
  (test 0.270 -> 0.360) for a precision cost within noise (test -2.3pp, val
  actually *higher* at 0.10). **Deployed:** `web/server.py` `YOLO_CONF`
  0.30 -> 0.10, with a new `YOLO_GATE_CONF = 0.30` keeping the "auto-label as
  waste" decision gate at the old, separately-validated threshold so only the
  candidate-box pool got wider, not the no-review-needed bar.

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

## F9. Serving resolution mismatch: 640-trained detector served at 960 (2026-07-05)

- **Symptom:** none visible - `YOLO_IMG_SIZE = 960` had been set in `web/server.py` as an
  "accuracy-first" setting (2026-06-09) and survived every later model promotion.
- **Root cause:** the deployed detector (`yolo26n_hardcase_v2_long`) is trained at
  imgsz=640. Serving it at 960 breaks the anchor-free scale priors it learned; the F4
  experiment that showed 960 helps was a 960 *fine-tune* (different weights, not
  deployed), and the serving default was never re-validated against the deployed 640
  weights.
- **Measured impact** (deployed weights, `runs/audits/detector_imgsz_sweep_clean_test.json`
  and `runs/audits/detector_imgsz_sweep_realworld_v2_test.json`):
  - clean test: recall 0.456@640 vs 0.429@960, mAP50 0.474 vs 0.409, mAP50-95 0.348 vs 0.229
  - realworld_v2 test: recall 0.492@640 vs 0.486@960, mAP50 0.499 vs 0.475
    (NOTE: this split is later shown to be 74% leaked - see F10. The 640>960
    *direction* is unaffected since leakage hits both sides equally, so the F9
    decision stands, but do NOT quote these as absolute field-generalization numbers.)
  - conf-sweep grids agree at the serving operating point (conf 0.10, clean test):
    P 0.590/R 0.456 @640 vs P 0.502/R 0.429 @960; organic recall 0.261 vs 0.239.
    Only small-box recall marginally favors 960 (0.360 vs 0.344).
  - 960 also costs ~2.25x the CPU inference compute on the Hugging Face Space.
- **Fix:** `web/server.py` `YOLO_IMG_SIZE` default 960 -> 640 (env-overridable as before).
  The conf=0.10 operating point remains optimal at 640 (best organic + small-box recall
  in the 640 grid with no overall recall cost).
- **Lesson:** serving-time inference parameters are model-coupled hyperparameters; every
  weight promotion must re-validate the full serving config (imgsz, conf, iou, max_det),
  not just the weights.

## F10. `yolo26_balanced_realworld_v2` is a leaked benchmark, not a field set (2026-07-05)

- **Symptom:** the set was named/treated as a "real-world" field benchmark and cited as a
  deployment proxy (F9). Investigated before launching a field-recall fine-tune.
- **Root cause:** `scripts/build_balanced_yolo_dataset.py` builds it by greedily
  box-count-rebalancing the SAME `yolo26_hardcase_dataset_v1` (studio-heavy) and
  hard-linking + hash-renaming the files (`test_*/val_*/train_*`), which hides their
  origin. It is not a new domain and it inherits F1's cross-split leakage.
- **Measured impact** (MD5 of raw bytes, hard links = identical files):
  **53/72 (74%) of realworld_v2 test images are byte-identical to hardcase TRAIN** -
  the deployed detector's own training data - plus 12 in hardcase val, 7 in hardcase test.
  0/72 are TACO field images. So R 0.492 / mAP50 0.499 on this split is train-test
  leakage, not generalization.
- **Consequence:** there is currently **no clean field detector benchmark with usable n**.
  The leakage-quarantined clean test has only 4 TACO images (167/171 TACO test images were
  quarantined as perceptual near-dups of train, F1); realworld_v2 is leaked studio; the only
  honest field signal is the 24-image TACO pipeline eval (R 0.265, ~+/-10pp noise).
- **Fix / policy:** realworld_v2 is retired from any accuracy claim (relative imgsz/conf
  comparisons only). The planned Step-1 "oversample the existing 1,034 TACO train images"
  is **rejected before running**: those TACO images are exactly the ones with perceptual
  near-dups in val/test, so a measured "+field recall" would be memorization on leaked
  eval images, and no clean split exists to measure it on anyway.
- **Corrected next action:** the field-domain work is gated on first building a clean,
  source-isolated field eval set (and matching train data) from EXTERNAL imagery not
  entangled with hardcase - PlastOPol (own images + class-agnostic litter boxes) with a
  fresh source-aware split and a mandatory dHash near-dup audit against every existing
  split (F1 procedure). Only then is a field fine-tune measurable.
- **Lesson:** a dataset's name is not its provenance. Audit any "real-world"/"balanced"
  derived set for its source and leakage before using it as a benchmark.

## F11. Field-rebalance fine-tune tested, rejected; honest field baseline established (2026-07-05)

- **Context:** F10 left no valid field benchmark. First built a clean, source-isolated
  field eval from the official TACO COCO annotations, split BY PHOTO with test/val drawn
  ONLY from photos the deployed model never trained on (MD5-verified 0 overlap with
  hardcase train): `external_datasets/taco_field_clean_v1` (train 1,113 / val 90 / test 120,
  461 test boxes). Script: `scripts/build_taco_field_clean.py`.
- **Honest field baseline** (deployed weights, class-agnostic IoU>=0.5 match at the serving
  operating point conf 0.10/imgsz 640 - class-agnostic because the pipeline re-labels each
  crop, so localization recall is what matters): **field recall 0.505**, small-box recall
  0.318, 1/120 zero-detection images. This CORRECTS the widely-quoted 0.265, which came from
  class-AWARE matching on 24 pipeline images and conflated localization misses with material
  mismatches. Evidence: `runs/audits/field_recall_baseline_deployed.json`. The real weakness
  is small objects (63% of field boxes are small).
- **Intervention:** fine-tuned the deployed YOLO26n on a 26.6%-field rebalanced set (studio
  19,559 x1 + clean TACO field x6 = `field_rebalance_v1`), from deployed weights, lr0 1e-4,
  mosaic on / mixup+copy_paste off, early-stop on field-val. Scripts:
  `build_field_rebalance.py`, `finetune_field_rebalance.py`.
- **Result (both gates failed):**
  - field test recall 0.505 -> **0.518 (+1.3pp)**, small-box 0.318 -> 0.336 (+1.8pp) -
    within noise (461 boxes), below the +5pp promote bar. Best epoch = 2 (model saturated
    on this field data in 2 epochs, then memorized oversampled dups -> val recall decayed
    0.68->0.62). Evidence: `runs/audits/field_recall_finetuned_k6.json`.
  - studio guard (clean val) recall 0.650 -> **0.627 (-2.3pp)**, mAP50-95 0.542 -> 0.495 -
    regressed beyond the -2pp guard. Evidence: `runs/audits/field_rebalance_k6_studio_guard.json`.
- **Root cause of the null result:** only 287 of the 1,113 field-train images were novel to
  the deployed model (it already trained on 826); reweighting data it already had cannot add
  field generalization. Same conclusion as F4c: the constraint is field data VOLUME/diversity,
  not the sampling factor. **Not promoted; deployed baseline kept.**
- **Next lever (evidence-backed):** external field imagery that adds NEW scenes - PlastOPol
  (own images + class-agnostic litter boxes, no entanglement with hardcase) merged with a
  fresh source-aware split + dHash near-dup audit (F1 procedure), evaluated on the now-clean
  `taco_field_clean_v1` test + a held-out PlastOPol test. Expected field recall 0.52 -> 0.60+
  only from genuinely new field scenes, not repetition.

## F12. Class-agnostic (nc=1) field expansion with PlastOPol — large field gains, studio tradeoff (2026-07-05)

- **Motivation:** F11 proved oversampling *existing* field data does nothing; the lever is NEW
  field data. Added PlastOPol (Roboflow v4, 2,418 real-world Marine-Debris-Tracker litter
  images, one-class). Since PlastOPol has no material labels, this also realigns the detector
  with the intended architecture: a **class-agnostic (nc=1) 'litter' detector**, classifier
  owns material ID.
- **Data hygiene (F1 procedure):** dHash audit of PlastOPol vs taco_field_clean_v1 + hardcase
  train found 21/2,418 near-dups (18 hardcase, 3 field-test) - all dropped. Clean split
  `plastopol_clean_v1` (1,671/247/479). Roboflow export was segmentation polygons -> converted
  to boxes. Audit: `runs/audits/plastopol_leakage_audit.json`.
- **Training:** collapsed studio+TACO labels to class 0, unioned with PlastOPol (field x2 =
  22.6% share), retrained YOLO26n nc=1 from the deployed 6-class backbone (696/708 items
  transferred, head reinitialized). Best epoch 27, field-val mAP50 0.803 (vs F11's ~0.70).
  Scripts: `build_class_agnostic_field.py`, `finetune_class_agnostic.py`.
- **Results** (class-agnostic IoU>=0.5 match; each model at its own tuned conf - the reinit
  head needed re-tuning 0.10->0.04, per F5/F9; `runs/audits/class_agnostic_field_v1_results.json`):

  | test set | baseline@0.10 | nc=1@0.04 | Δ recall | Δ small-box |
  |---|---|---|---|---|
  | TACO field (held-out) | 0.505 | **0.614** | **+10.9pp** | +16.5pp |
  | PlastOPol field | 0.573 | **0.777** | **+20.4pp** | +21.7pp |
  | studio (crowded clean-val) | 0.584 | 0.536 | **-4.8pp** | -3.9pp |

- **Verdict:** the field lever WORKS - large recall gains exactly on the weakness (small
  field objects, +16-22pp), confirming F11's conclusion that new data (not resampling) is
  the constraint. Cost: -4.8pp studio recall (crowded lab piles, NOT the real-world upload
  distribution the app serves). The pre-registered -2pp studio guard is not met, so not
  auto-promoted; this is a deployment-priority decision. Promotion also needs a small
  `web/server.py` change (drop the detector-class material vote in the alpha-blend, since the
  detector is now class-agnostic) + conf 0.10->0.04 + HF redeploy.
- **Initially promoted field-first, then REVERTED (2026-07-05) - see F13.** The promotion
  compared nc=1@conf0.04 against the 6-class baseline@conf0.10, an unfair operating-point
  comparison. Re-running the 6-class detector at matched conf 0.04 showed most of the apparent
  field gain was the conf drop (an ALLOWED lever), not the architecture: 6-class 0.505@0.10 ->
  0.586@0.04. The nc=1 head added only +2.8pp field over 6-class@0.04 while costing -10pp studio
  recall. Reverted to the 6-class detector; kept the conf lever instead (F13). Lesson: always
  compare candidate models at each model's own tuned operating point before attributing a gain
  to an architecture change.

## F13. Detector conf lever captures the field gain without the architecture change (2026-07-05)

- **Problem (taxonomy A, YOLO missed detection):** field recall 0.505 far below target; F12
  attributed the fix to a class-agnostic (nc=1) architecture change.
- **Debug-first check (the one F12 skipped):** swept the DEPLOYED 6-class detector's conf below
  0.10 - an allowed lever never tested on the field split. Clean held-out field test, class-agnostic
  match (`runs/audits/detector_conf_sweep_field_6class.json`):

  | conf | field recall | field small-box | studio recall |
  |---:|---:|---:|---:|
  | 0.10 (old) | 0.505 | 0.318 | 0.584 |
  | 0.05 | 0.557 | 0.390 | - |
  | **0.04** | **0.586** | **0.432** | **0.637** |

- **Result:** dropping conf 0.10 -> 0.04 lifts field recall **+8.1pp** AND studio recall **+5.3pp**
  AND small-box **+11pp** - a strict, no-cost recall gain across both domains from a pure threshold
  change. The nc=1 model (F12) reached 0.614 field but only +2.8pp over 6-class@0.04, at -10pp studio.
- **Fix:** reverted the F12 nc=1 promotion; deployed detector stays 6-class at
  `web/server.py YOLO_CONF = 0.04` (gate stays 0.30). End-to-end smoke test passes. The PlastOPol
  clean split and class-agnostic experiment are kept as documented evidence, not deployed.
  **Pending: HF Space redeploy.**
- **Lesson:** exhaust the cheap allowed levers (conf/IoU/aug/hard-neg) and compare at matched
  operating points BEFORE any architecture change. Recall is threshold-sensitive; a mismatched-conf
  comparison can make a threshold gain look like an architecture win.

## F14. Glass->Plastic: detector material vote overrode the classifier (2026-07-05)

- **Problem (taxonomy D, misclassification via pipeline):** real-world field glass labeled plastic.
- **Root cause (debug-first, NOT classifier weakness):** classifier is strong on glass (clean GT
  glass F1 0.921, recall 0.87-0.91). The alpha-blend (`server.py`) injected the DETECTOR's material
  vote at alpha up to 0.70. The detector is plastic-biased on field - it predicts plastic for
  516/662 (78%) field boxes, glass for 6 (measured on taco_field_clean test). So a confident field
  "plastic" box flips a correct glass crop: at alpha 0.70, plastic 0.52 > glass 0.22.
- **Fix:** cap alpha at 0.40 (top tier 0.70->0.40). A confident classifier call now survives a
  confident wrong detector (glass 0.45 > plastic 0.34). Studio alpha-cap sweep (n=2835, glass n=400):
  macroF1 0.893@0.70 -> 0.886@0.40 (-0.7pp), glass recall 0.912 -> 0.900. Small measured cost on the
  domain where the detector is reliable; removes the override on the domain where it is not.
- **Note:** benchmarks (clean GT crops, studio-trained detector) do NOT reproduce the bug - the blend
  even helps glass there. The failure is field-domain glass, for which there is no GT benchmark
  (taco field test has 2 glass boxes). Fix justified by the mechanism + the plastic-bias measurement,
  not a benchmark delta. **Pending: HF redeploy.**

## F15. Hard-negative mining for small-object recall (2026-07-05)

- **Problem (taxonomy A):** field small-object recall still low (~0.43); user reports missed small litter.
- **Approach (beats the F11 wall by NOT resampling seen data):** mined the deployed detector's
  small-object false negatives on the field train sets (`scripts/mine_small_fn.py`): 295 hard taco
  imgs / 673 small-FN, 237 hard PlastOPol imgs / 725 small-FN. PlastOPol is NEW (never trained on).
  Pseudo-labeled PlastOPol's class-agnostic boxes with the ConvNeXt classifier (the 92.9% material
  authority; `scripts/pseudo_label_plastopol.py`) so new field data can train the 6-class detector.
  Built a hard-emphasized fine-tune set (8,442 imgs, 52.6% field, hard imgs x4, studio x4000 anchor;
  `scripts/build_hardneg_finetune.py`) and fine-tuned from deployed weights with small-object aug
  (mosaic + scale 0.6; bbox-only so no copy_paste). Early-stopped epoch 5.
- **Results** (class-agnostic recall, conf 0.04; `runs/detect/hardneg_smallobj_v1/`):

  | test set | baseline | fine-tuned | Δ recall | Δ small-box |
  |---|---|---|---|---|
  | TACO field | 0.586 | 0.566 | -2.0pp (within noise) | -1.1pp (noise) |
  | PlastOPol field (held-out) | 0.573 | **0.746** | **+17.3pp** | **+16.1pp** |
  | studio guard | 0.637 | 0.647 | +1.0pp | +5.4pp |

- **Verdict: PROMOTED.** Large held-out gain on the deployment-like PlastOPol domain (ground-level
  real litter), studio preserved (guard passes), TACO flat (within ~1 SE). Same 6-class architecture
  so NO server change (conf 0.04 and the F14 alpha-cap still apply). Backup:
  `best_before_hardneg_20260705.pt`. Smoke-tested. **HF redeploy pending.**
- **Limit:** gain is PlastOPol-domain-specific; a universal small-object fix still needs new labeled
  small-object data (the actual RoLID boxes - the shared RoLID zip was unlabeled raw dashcam frames).

## F16. Waste-state gate loosening tested, rejected — "stale threshold" hypothesis falsified (2026-07-06)

- **Problem (taxonomy D/F):** confidently-classified trash is shown as `review`, not `waste`. Tracing
  the 6 demo samples through `predict_image`: on `outdoor_street_cans` the classifier calls 9 cans
  Metal @ 90-97%, yet 8/9 return `review` ("material known, disposal state unknown"). No sample ever
  returns `not_waste`. Every `not_waste`/"clean" path requires the detector to MISS the item (no box
  -> `Background` scene fallback) — i.e. it's the F15 recall problem, not the decision logic.
- **Hypothesis:** the waste gate needs `yolo_score >= YOLO_GATE_CONF (0.30)`, but F13 dropped the
  detector to conf 0.04, so box scores are now systematically 0.04-0.20. Guess: `0.30` is a *stale*
  threshold left over from the conf-0.10 detector, wrongly demoting confident material to `review`.
  Fix tried: also gate `waste` when `label_conf >= 0.55 AND yolo_key == label_key` (trust a confident
  material call when the localizer agrees on the class, regardless of box score).
- **Demo looked great, then the precision check killed it.** On the 6 samples the cans/bottles flipped
  to `waste -> Recycling` with zero `not_waste`/Background promotions. But a class-agnostic IoU>=0.5
  precision check vs GT on 120 field + 150 studio images, sweeping the confidence floor T:

  | gate | field waste-precision | studio waste-precision | promoted-box precision |
  |---|---|---|---|
  | OLD (`yolo_score>=0.30`) | **0.812** | **0.816** | — |
  | loosened, any T in 0.55-0.85 | 0.63-0.73 | 0.68-0.70 | ~0.48 both domains |

  Loosening ~doubled the boxes tagged `waste`; the extra promotions were only ~48% IoU-matched to a
  real object (duplicate / poorly-localized / spurious conf-0.04 boxes) — which makes the separate
  "detects wrong boxes" complaint WORSE.
- **Verdict: REJECTED, reverted.** `YOLO_GATE_CONF=0.30` is NOT stale — it is an active, well-calibrated
  precision guard worth ~0.81 on both domains; removing it costs ~15pp precision. The safe fallback is
  benign (`review` routes to manual check, not a wrong bin), so trading precision + more spurious boxes
  to relabel `review`->`waste` is the wrong trade for a waste-sorter. No server change shipped.
- **Lesson:** a 6-image demo is the F3 cherry-pick trap in miniature — a change that flatters the demo
  can regress a 270-image two-domain precision measurement. Validate serving-logic changes on a real
  matched-operating-point precision check BEFORE trusting them. The honest lever for "obvious trash
  shown as review" is upstream detector precision (tighter boxes / de-duplication), not a looser gate;
  a learned clean-vs-dirty *state* model remains blocked (no contamination label exists in any dataset).

## F17. Waste-state gate (S6) removed entirely — untrainable decision layer, net complexity (2026-07-18)

- **Decision (with supervisor):** remove the S6 waste-state decision (`waste` / `not_waste` /
  `review`) from the served pipeline. Root cause it never solved: the gate was rule logic standing in
  for a *trained* clean-vs-dirty state model, and no dataset in the project carries a contamination /
  disposal-state label to train one (the recurring blocker noted at the end of F16). It could therefore
  only ever be hand-tuned heuristics, and each retune (F16, and the joint-confidence tiers before it)
  added branches without a ground truth to validate against.
- **Evidence it wasn't paying its way:** the conference-paper Section 4.6 measurement already showed the
  gated deployed config at 96.26% bin accuracy vs the ungated variant at 97.41% on the 1,042-image
  external set — the gate cost ~1.15pp bin accuracy, and its only claimed benefit (waste-state routing)
  was exactly the untrainable part. `runs/audits/pipeline_bin_decision_eval_no_gate.json` had shown the
  same directional result earlier.
- **Change shipped:** deleted `estimate_detection_waste_state` and `choose_final_decision` from
  `web/server.py`; the headline material and bin route now come straight from the area×confidence
  detection vote (`ROUTES.get(dominant, "Review")`, ungated). Web GUI collapsed S1–S6 → S1–S5 (the
  waste-state fact row, the per-box review styling, and the "Decision" headline are gone). Decision
  regression tests, the pipeline eval script, and every workflow diagram were updated; the
  waste-state state-machine diagram was deleted (nothing left to depict).
- **Lesson:** a decision layer you cannot train is not a feature, it is a liability you keep re-tuning.
  When the only justification for a component is behaviour you have no ground truth for, the honest move
  is to remove it, not to keep hand-calibrating it. Report what the models actually measure (material +
  bin), not a disposal-state judgment the data never supported.

## Status summary

| Failure | Severity | Status |
|---|---|---|
| F1 eval leakage | Critical | Quarantined eval sets built; re-validation pending training completion |
| F2 cross-dataset broken | High | Fixed + committed |
| F3 PCA cherry-pick | Medium | Fixed + committed |
| F4 tiny-object recall | High | Long retrain + 960px + sliced-inference + tile-training all tried; baseline (long retrain) stays deployed |
| F5 untuned conf | Medium | Re-swept post-F4c at serving imgsz; conf 0.30->0.10 deployed |
| F6 label hygiene | Low | Quantified; no action needed |
| F7 environment | Medium | All root-caused; rules recorded |
| F8 architecture swap | Info | Documented |
| F9 serving resolution mismatch | Medium | Fixed: YOLO_IMG_SIZE 960 -> 640, re-measured on both eval domains |
| F10 realworld_v2 leaked benchmark | High | Quantified (74% leaked); retired from accuracy claims; field work gated on external clean eval |
| F11 field-rebalance fine-tune | Info | Rejected (field +1.3pp within noise, studio -2.3pp); honest field baseline 0.505 set; next lever = external field data (PlastOPol) |
| F12 class-agnostic + PlastOPol | Info | REVERTED - apparent gain was mostly the conf lever (see F13); kept as evidence, not deployed |
| F13 detector conf lever (field) | High | Deployed: 6-class @ conf 0.04 -> field recall +8.1pp, studio +5.3pp, small-box +11pp, no architecture change |
| F14 glass->plastic fusion bug | High | Fixed + deployed: alpha-blend cap 0.70->0.40 (detector vote can't override a confident classifier call); -0.7pp studio macro-F1 |
| F15 hard-negative mining | High | Promoted + deployed: PlastOPol field small-object recall +16pp, studio preserved, TACO flat; same architecture |
| F16 waste-gate loosening | Info | Rejected, reverted - 6-sample demo flattered it but a 270-image two-domain check showed ~15pp precision loss; the 0.30 box gate is a real precision guard, not stale |
| F17 waste-state gate (S6) removed | Info | Removed 2026-07-18: untrainable decision layer (no disposal-state label in any dataset), cost ~1.15pp bin accuracy for no measurable benefit; pipeline now S1-S5, material + bin only |
