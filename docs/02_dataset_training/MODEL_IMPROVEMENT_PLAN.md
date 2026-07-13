# WasteWise Model Improvement Plan

Last updated: 2026-06-09

## Completed Update - 2026-06-09

- Deployed the Step 1 web/API review-gate update to Hugging Face Space `khoaphung/wastewise-ai`.
- Hidden the visible technical Mode row from the result UI. The API may still return internal model metadata, but the public page now shows professional result fields only.
- Added `possibleMaterial` to the prediction response so low-confidence or out-of-distribution scans can say Review without throwing away the most likely material signal.
- Kept the accuracy-first detector settings restored:
  - `YOLO_IMG_SIZE = 960`
  - `YOLO_MAX_DETECTIONS = 80`
  - `MAX_CROP_VERIFICATIONS = 80`
- Added review thresholds for weak real-world scans:
  - `WASTE_REVIEW_CONF = 0.50`
  - `SCENE_REVIEW_CONF = 0.55`
- Redeployed Space commit: `7a8af57ec3c3efc5fa468d08c0f79c5e5b1f8f46`.
- Added `scripts/prepare_hard_case_audit.py` for trusted-source hard-case sampling without installing Hugging Face dependencies.
- Generated the first audit sample at `external_datasets/hard_case_audit/`:
  - 72 RealWaste images downloaded and image-validated
  - 100 Outerview metadata rows sampled
  - 172 total manifest records
  - gallery: `external_datasets/hard_case_audit/review_gallery.html`
- Added `scripts/download_hard_case_datasets.py` for full trusted-source downloads with parallel workers, resume/skip behavior, TACO fallback image handling, and combined manifests.
- Downloaded full hard-case sources into `external_datasets/hard_case_full/`:
  - RealWaste: 4,752 records / 4,752 images present
  - TACO official: 1,500 records / 1,500 images present, including 35 official 640px fallback downloads for dead original URLs
  - Outerview Global Trash & Debris Index: 25,000 metadata rows, 23,297 extracted images, 1,703 metadata-only rows
  - combined manifest: `external_datasets/hard_case_full/combined_manifest.csv`
  - full output size after extraction: about 7.0 GB
- Added `scripts/build_hard_case_splits.py` and generated `external_datasets/hard_case_full/source_aware_split_manifest.csv`.
- Current split recommendation:
  - classifier train candidates: 2,763
  - classifier hard validation: 580
  - classifier hard test: 596
  - YOLO train candidates: 1,115
  - YOLO hard validation: 200
  - YOLO hard test: 185
  - OOD review pool: 23,297
  - metadata-only pool: 1,703
  - review pool from RealWaste out-of-taxonomy labels: 813
- Exported classifier-ready hard-case dataset at `data/hard_case_classifier_v1/` with 16,539 linked records.
- Added `scripts/train_convnext_hardcase_tuned.py` and retrained the classifier on CUDA.
- New hard-case classifier artifact: `runs/dl/convnext_hardcase_tuned/best_convnext_ensemble_tuned.pth`.
- New hard-case classifier scaler: `runs/dl/convnext_hardcase_tuned/handcrafted_scaler.npz`.
- New classifier metrics:
  - best validation macro F1: 0.9322
  - validation accuracy: 93.13%
  - test accuracy: 93.88%
  - test macro F1: 0.9398
- Old deployed classifier on the same hard-case split:
  - validation accuracy: 83.43%
  - validation macro F1: 0.8385
  - test accuracy: 82.72%
  - test macro F1: 0.8304
- Verdict: promote the hard-case classifier after final local API smoke tests; it is clearly better on the new real-life hard-case split.
- Added YOLO hard-case preparation scripts:
  - `scripts/export_taco_yolo_hardcase.py`
  - `scripts/merge_yolo_hardcase_dataset.py`
  - `scripts/train_yolo26_hardcase.py`
- Exported TACO YOLO hard-case dataset at `external_datasets/hard_case_yolo_taco_v1/`:
  - 1,389 images
  - train/val/test: 1,034 / 184 / 171
  - mapped boxes: plastic 2,150; metal 361; glass 254; cardboard 243; paper 200; organic 8
- Merged YOLO26 training dataset at `external_datasets/yolo26_hardcase_dataset_v1/`:
  - train images: 20,593
  - val images: 3,450
  - test images: 1,275
- Upgraded Ultralytics from 8.3.40 to 8.4.62 and verified `models/pretrained/yolo26n.pt` loads.
- YOLO26 decision:
  - Test YOLO26n first because Hugging Face Space is CPU-bound and YOLO26 is designed for faster edge/CPU inference.
  - Do not promote blindly. Promote only if it beats current YOLO on hard-case mAP50, mAP50-95, per-class recall, and CPU latency.
  - YOLO26s is second priority only if YOLO26n is too weak on recall.
- Blocker: launching the YOLO26 GPU training job from Codex was blocked by the app usage/approval limit. Resume with `.\.venv311\Scripts\python.exe scripts\train_yolo26_hardcase.py --epochs 30 --batch 32 --imgsz 640 --workers 8 --cache disk`.

## Completed Update - 2026-06-07

- Removed static demo scanner samples from the web app.
- Replaced webpage mock campus/product numbers with real validation and dataset metrics.
- Ran `scripts/train_convnext_ensemble_tuned.py` on CUDA.
- Completed ConvNeXt-Tiny + 637 handcrafted-feature fine-tuning:
  - 7,000 balanced training images
  - 2,100 balanced validation images
  - best validation accuracy: 92.52%
  - artifact: `runs/dl/convnext_ensemble_tuned/best_convnext_ensemble_tuned.pth`
  - run note: `runs/dl/convnext_ensemble_tuned/RESULT.md`
- Calibrated web backend YOLO thresholds to the validated balanced localization setting:
  - `YOLO_CONF = 0.30`
  - `YOLO_RECOVERY_CONF = 0.30`
  - evidence: `runs/dl/localization_rework/THRESHOLD_SWEEP_300.md`

Training was paused after this completed run, then resumed on 2026-06-07 for YOLO localization training.

## Objective

Improve the deployed WasteWise scanner so it is less dependent on demo-like samples and handles real student uploads with higher material accuracy, better localization, and clearer review decisions.

Current deployed pipeline:

- Scene classifier: `runs/dl/convnext_ensemble_tuned/best_convnext_ensemble_tuned.pth`
- Feature scaler: `runs/dl/convnext_ensemble_tuned/handcrafted_scaler.npz`
- Localizer: `models/trained/yolov11_detector/best.pt`
- Decision logic: full-image classifier, YOLO boxes, crop reclassification, and a conservative waste-state gate.
- Web behavior: students upload their own image; no static demo samples are used on the scanner page.

## YOLO Resume Update - 2026-06-07

- Resumed `scripts/train_super_yolo.py` from `runs/detect/yolov11_super_dataset/weights/last.pt`.
- Completed epoch 30/30 on CUDA.
- Final validation metrics:
  - precision: 75.21%
  - recall: 61.00%
  - mAP50: 68.74%
  - mAP50-95: 51.36%
- Promoted new detector to `models/trained/yolov11_detector/best.pt`.
- Backed up previous detector to `models/trained/yolov11_detector/best_before_yolo_resume_20260607.pt`.
- Evidence note: `runs/detect/yolov11_super_dataset/RESULT_20260607.md`.
- Local web verification passed with the tuned ConvNeXt classifier plus the promoted YOLO detector.

## Success Metrics

- Raise macro F1 on the held-out material classifier set by at least 5 percentage points.
- Improve recall on the weakest material classes without dropping precision below the current baseline by more than 2 percentage points.
- Reduce the top confusion pairs found by `scripts/analyze_confusion.py`.
- Improve YOLO `mAP50` and per-class recall on real mixed-scene waste photos.
- Keep warm inference practical for the Hugging Face Space and document CPU latency before deployment.
- Add a measurable review gate: uncertain or non-waste images should route to Review instead of forcing a wrong bin.

## Phase 1: Build A Real Error Dataset

1. Export examples from the web app History after student/user testing.
2. Save the uploaded image, predicted class, confidence, bin route, waste state, and user correction.
3. Manually label each error case with:
   - true material class
   - whether it is actually waste
   - bounding boxes for visible waste items
   - notes for confusing cases such as dirty paper, coated cups, food packaging, and mixed materials
4. Keep this as a hard-case validation split, not only as training data.

Recommended output folder:

```text
external_datasets/student_hard_cases/
```

## Phase 1A: Trusted Hard-Case Dataset Acquisition

Use public datasets to avoid slow manual collection, but do not merge them blindly. The priority is to import hard cases that look like real uploads: cluttered backgrounds, hands holding objects, mixed material items, dirty/used trash, small objects, partial occlusion, low light, and non-waste lookalikes.

Primary sources to audit first:

- RealWaste on Hugging Face / UCI: landfill waste classification images with 4,752 rows and CC-BY-4.0 licensing. Best for material-classifier hard cases. Map labels into WasteWise classes; send textile/miscellaneous edge cases to `review` or `background` until manually checked.
- TACO official dataset: real litter in woods, roads, and beaches with COCO-format segmentation annotations. Best for YOLO/localization hard cases. Merge sparse source classes into WasteWise material classes before training.
- Outerview Global Trash & Debris Index on Hugging Face: geotagged real-world trash/debris observations, 30,000 public sample entries, CC-BY-4.0. Use as a hard-case/OOD audit source because labels are generated by CV systems, not guaranteed manual truth.
- OpenLitterMap: open litter and plastic-pollution observations with geotagged images and classified tags. Use for environmental litter variety after checking export/API access and ODbL attribution requirements.
- TrashCan marine debris: underwater trash with segmentation and bounding-box labels. Use only as robustness data if the model needs floating/marine debris; do not let it dominate campus/household waste training.

Secondary sources:

- Hugging Face waste search results, Kaggle, and Roboflow Universe can be useful, but treat them as candidates only. Accept a dataset only if license, source, label schema, image count, annotation type, and sample quality are clear.

Step-by-step import rule:

1. Add each candidate dataset to `docs/02_dataset_training/external_dataset_registry.json`.
2. Download metadata and 100-300 sample images per source first.
3. Create a label map into `Plastic`, `Glass`, `Metal`, `Paper`, `Cardboard`, `Organic`, and `Background`.
4. Visually audit the sample before any full download.
5. Put uncertain labels into a `review` bucket instead of forcing a wrong material.
6. Build a source-aware hard-case validation split so images from the same dataset source do not leak into both train and validation.
7. For the current expedited run, full RealWaste, TACO, and Outerview downloads are already present under `external_datasets/hard_case_full/`.
8. Retrain classifier first, then YOLO, then evaluate both against the hard-case split.
9. Deploy only if hard-case macro F1 and weak-class recall improve without creating more confident wrong routes.

## Phase 2: Audit And Rebalance The Data

Run or update these scripts before retraining:

```powershell
.\.venv311\Scripts\python.exe scripts\dataset_audit.py
.\.venv311\Scripts\python.exe scripts\optimize_dataset.py
.\.venv311\Scripts\python.exe scripts\cross_dataset_validation.py
```

Checks:

- Per-class image counts for plastic, glass, metal, paper, cardboard, organic, and background.
- Duplicate or near-duplicate images across train, validation, and test.
- Source leakage where images from the same dataset batch appear in both train and validation.
- Overrepresentation of clean product-style images compared with real cluttered scenes.
- Background and non-waste coverage.

Validation splits should be source-aware, not random-only.

## Phase 3: Fine-Tune The Material Classifier

Baseline:

```powershell
.\.venv311\Scripts\python.exe scripts\run_final_validation.py
.\.venv311\Scripts\python.exe scripts\analyze_confusion.py
```

Experiments:

```powershell
.\.venv311\Scripts\python.exe scripts\train_convnext_ensemble_tuned.py
.\.venv311\Scripts\python.exe scripts\convnext_ensemble_pipeline.py
.\.venv311\Scripts\python.exe scripts\hybrid_ensemble_pipeline.py
```

Training changes to test:

- Unfreeze only the last EfficientNet or ConvNeXt blocks first, then expand if validation is stable.
- Add stronger but realistic augmentations: blur, compression, rotation, crop, lighting shifts, partial occlusion, and background clutter.
- Use class weights or focal loss for weak classes.
- Track macro F1, per-class recall, top-2 accuracy, confusion matrix, and calibration.
- Keep a small hard-case validation set untouched until final comparison.

Acceptance rule:

- Choose the model that improves macro F1 and weak-class recall, not just overall accuracy.

## Phase 4: Improve YOLO Localization

Use the existing localizer training path:

```powershell
.\.venv311\Scripts\python.exe scripts\train_super_yolo.py
.\.venv311\Scripts\python.exe scripts\generate_clean_yolo_cm.py
```

Add new labels from:

- real student uploads
- campus bin-area photos
- mixed-waste photos
- small or partially hidden objects
- low-light indoor photos

Evaluate:

- `mAP50`
- per-class precision and recall
- false positives on clean/background scenes
- small-object recall
- number of boxes passed into crop verification

Acceptance rule:

- Prefer a detector with stronger recall if crop verification can control false positives.

## Phase 5: Train A Waste-State Head

The current web API uses conservative rules for `waste`, `not_waste`, and `review`. Replace this with a trained state model when enough labels exist.

Suggested labels:

- `waste`: disposable item visible and should enter routing logic
- `not_waste`: clean background, reusable object, hand, desk, wall, or scene with no disposable item
- `review`: ambiguous, mixed, blocked, too blurry, or policy-dependent

Training options:

- small classifier head on YOLO crops
- EfficientNet/MobileNet crop classifier with 3 state classes
- combined material plus state multitask head

Acceptance rule:

- The state head must reduce wrong confident routes. It is acceptable to increase Review decisions if that prevents incorrect bin guidance.

## Phase 6: Calibrate The Final Decision Logic

Tune thresholds using validation data:

- scene classifier confidence
- YOLO confidence
- crop classifier confidence
- state model confidence
- review threshold
- no-box fallback behavior

Decision output should be:

- material
- confidence
- bin route
- waste state
- review flag
- model evidence summary

Run:

```powershell
.\.venv311\Scripts\python.exe scripts\run_batch_demo.py
.\.venv311\Scripts\python.exe scripts\run_100_demo_test.py
```

Update the scripts if they still depend on curated demo samples; they should run against validation folders and hard-case folders.

## Phase 7: Export And Deploy

Before deploying:

```powershell
.\.venv311\Scripts\python.exe scripts\export_ensemble_onnx.py
.\.venv311\Scripts\python.exe scripts\export_tflite.py
.\.venv311\Scripts\python.exe scripts\tflite_fps_test.py
```

Deployment checklist:

- Replace model files under `models/trained/`.
- Run local `web/server.py` and test at least 10 real uploads.
- Confirm `web/app.js` still saves History records.
- Run `node --check web\app.js`.
- Redeploy Hugging Face Space with `scripts/deploy_hf_space.py`.
- Verify `/api/health` reports the expected model paths and thresholds.

## Immediate Next Work

1. Remove any remaining training scripts that assume static demo images as the evaluation source.
2. Add an error-capture export format from web History.
3. Label at least 200 real hard cases, balanced across classes.
4. Run the baseline validation and confusion analysis.
5. Fine-tune classifier and YOLO separately before changing the deployment decision logic.
6. Add the waste-state head only after enough `waste`, `not_waste`, and `review` labels exist.
