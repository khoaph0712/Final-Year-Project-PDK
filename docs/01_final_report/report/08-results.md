# 8. Results

All figures in this chapter are computed on the quarantined-clean splits defined
in §6.4 unless explicitly labelled *original*. Where both are available, both are
reported, so that the magnitude of the leakage correction stays visible rather
than being quietly absorbed.

## 8.1 Classical Machine Learning Branch

The 637 handcrafted features of §7.2 were swept across nine PCA dimensionalities
and the untransformed space, on 7,000 training and 2,100 test crops across seven
classes. ExtraTrees is the strongest classical model at every dimensionality.

| Feature space | Explained variance | Best model | Accuracy | Drop vs full |
|---|---:|---|---:|---:|
| PCA-16 | 99.33% | ExtraTrees | 59.76% | 14.00 pp |
| PCA-32 | 99.60% | ExtraTrees | 65.86% | 7.90 pp |
| PCA-64 | 99.78% | ExtraTrees | 67.71% | 6.05 pp |
| PCA-96 | 99.86% | ExtraTrees | 67.43% | 6.33 pp |
| PCA-128 | 99.90% | ExtraTrees | 67.81% | 5.95 pp |
| PCA-256 | 99.97% | ExtraTrees | 65.76% | 8.00 pp |
| PCA-384 | 99.99% | ExtraTrees | 63.67% | 10.09 pp |
| **Full 637-D** | **100.00%** | **ExtraTrees** | **73.76%** | — |

<!-- src: runs/ml/pca_feature_model_sweep/PCA_Model_Sweep_Report.md#best-model-at-each-dimension -->

*Table 8.1 — Classical branch: best model per feature-space dimensionality.
Macro-F1 at full dimensionality is 0.7381.*

The full 637-dimensional representation reaches 73.76% accuracy, roughly 20
points below the deep second-stage classifier of §8.2. This gap is the
quantified contribution of learned representation over engineered features, and
it is the reason Branch A is not the deployed system.

**An anomaly requiring explanation.** Accuracy is not monotonic in PCA
dimensionality. It peaks at 128 components (67.81%), declines through 384
(63.67%), then rises sharply to 73.76% at the full 637 dimensions. Since 384
components already capture 99.99% of the variance, a 10-point gap between
PCA-384 and the untransformed features cannot be explained by information loss.

The working explanation is that the PCA rotation destroys axis-aligned structure
that tree ensembles exploit. Of the 637 features, 576 are HOG bins, which are
sparse and individually interpretable, and ExtraTrees splits on individual
features. A dense rotation produces components that are linear combinations of
many bins, which axis-aligned splits handle poorly. Under this account the
full-dimensional result reflects better-*aligned* information rather than more
information.

**This explanation is provisional and is reported as such.** It has not been
verified experimentally, and an alternative account — an inconsistency in
preparation between the full-dimensional and PCA-transformed conditions, such as
differing normalisation — has not been excluded.

> A superseded report, `runs/dl/pca_experiments/PCA_Dimensionality_Report.md`,
> quotes different figures for this sweep. It selected the model closest to a
> pre-chosen "637→128 costs ~2%" narrative, citing Linear SVM's −2.52 pp while
> the strongest models lost 5.7–9 pp at the same dimensionality. It is recorded
> as Failure F3 and is superseded by Table 8.1.
> <!-- src: docs/01_final_report/FAILURES_AND_FIXES.md#F3 -->

![Classical model sweep across PCA dimensionalities](web/assets/figures/cmp_pca_model_sweep.png)

*Figure 8.1 — Accuracy by model and PCA dimensionality. The non-monotonic
ExtraTrees curve is the anomaly discussed above.*

## 8.2 Deep Learning Classification

### 8.2.1 Backbone Selection

Three Stage 2 backbones were compared under identical conditions.

| Backbone | Feature dim | Train (s) | Val acc | Test acc | Test macro-F1 |
|---|---:|---:|---:|---:|---:|
| ConvNeXtV2-Tiny | 1,405 | 17.05 | 0.9325 | 0.9455 | 0.9459 |
| Swin-Tiny | 1,405 | 29.22 | 0.9448 | **0.9458** | **0.9464** |
| EfficientNetV2-S | 1,917 | 24.47 | 0.7925 | 0.8023 | 0.7989 |

<!-- src: docs/01_final_report/stage2_backbone_reproduced_metrics.json#convnextv2,swin_tiny,efficientnetv2_s -->

*Table 8.2 — Stage 2 backbone comparison.*

Swin-Tiny and ConvNeXtV2-Tiny are statistically indistinguishable on test
accuracy, differing by 0.03 points across 2,696 samples. **ConvNeXt was selected
on training cost — 17.05 s against 29.22 s — not on accuracy.** This is stated
explicitly rather than constructing a performance justification the data does
not support. EfficientNetV2-S underperforms both by roughly 14 points and was
discarded.

### 8.2.2 Effect of the Leakage Correction

Two classifiers were re-evaluated after quarantine, on their respective datasets.

| Model | Dataset | Test set | n | Accuracy | Macro-F1 |
|---|---|---|---:|---:|---:|
| EfficientNetB0 | `merged_dataset_v5` | original | 5,600 | 94.30% | 0.9431 |
| EfficientNetB0 | `merged_dataset_v5` | **clean** | **3,194** | **91.77%** | **0.9114** |
| ConvNeXt ensemble (deployed) | `hard_case_classifier_v1` | original | 2,696 | 93.88% | 0.9398 |
| ConvNeXt ensemble (deployed) | `hard_case_classifier_v1` | **clean** | **2,151** | **92.93%** | **0.9290** |

<!-- src: runs/audits/classifier_clean_test_eval.json (EfficientNetB0); runs/audits/convnext_clean_eval.json#splits.test (ConvNeXt); docs/01_final_report/FAILURES_AND_FIXES.md#F1 (original ConvNeXt figures) -->

*Table 8.3 — Classifier accuracy before and after removing cross-split
duplicates. The two rows measure different models on different datasets and are
not directly comparable to each other.*

Leakage inflated the EfficientNetB0 figure by 2.53 points of accuracy and 3.17
of macro-F1. For the deployed ConvNeXt the inflation is far smaller — 0.95
points — which is itself evidence that the deployed classifier is genuinely
strong rather than a beneficiary of contamination. **92.93% is the figure
carried forward** for the deployed model, and it is the baseline against which
§8.2.3 measures fine-tuning.

### 8.2.3 Detector-Crop Fine-Tuning: Principal Result

| Model | Detector-crop accuracy | Clean GT-crop accuracy |
|---|---:|---:|
| Baseline, trained on GT crops | 76.91% | 92.93% |
| **Fine-tuned on detector crops** | **88.88%** | **93.77%** |
| **Difference** | **+11.97 pp** | **+0.84 pp** |

<!-- src: runs/dl/convnext_detector_crops_ft/finetune_result.json -->

*Table 8.4 — Effect of training Stage 2 on the distribution it actually
receives.*

This is the strongest result in the project. Training the classifier on the
distribution it encounters in production improves production accuracy by 11.97
percentage points, and clean ground-truth accuracy does not regress — it
improves slightly — so this is a strict improvement, not a trade-off.

The magnitude deserves emphasis relative to the architectural work around it.
Promoting the detector backbone from YOLO26n to YOLO26m, raising input
resolution to 960 px, and substituting the Stage 2 backbone across three modern
architectures each produced changes of a few points or fewer. **Correcting the
training distribution at the stage interface produced twelve.**

One qualification: macro-F1 of the fine-tuned model on detector crops is 0.7989,
well below its 88.88% accuracy, so the gain is not uniform across classes.

### 8.2.4 Domain Transfer

Training on one acquisition domain and evaluating on the other, both sharing all
seven classes. "Kaggle" denotes Kaggle Garbage Classification studio imagery;
"Roboflow" denotes pooled real-world crops from the Roboflow-hosted detection
exports.

| Direction | In-domain | Cross-domain | Gap | Cross macro-F1 |
|---|---:|---:|---:|---:|
| Kaggle → Roboflow | 80.00% | 39.81% | **40.19 pp** | 0.4023 |
| Roboflow → Kaggle | 72.76% | 44.49% | 28.27 pp | 0.4074 |

<!-- src: runs/audits/classifier_domain_gap.json; docs/01_final_report/WasteWise_Conference_Paper_DRAFT.md#4.5 -->

*Table 8.5 — Domain-holdout ablation.*

A Kaggle-trained classifier retains less than half its accuracy on real-world
imagery. The asymmetry is the informative part: Roboflow-trained models transfer
to studio conditions better (−28.27 pp) than the reverse (−40.19 pp), because
the real-world data contains occlusion, clutter, lighting variation and
deformation that the studio data lacks. A model trained on the harder domain has
seen conditions resembling the easier one; the converse does not hold.

The practical implication is that real-world data carries more value per image
than curated studio data for this task, and that accuracy reported on curated
benchmarks should not be read as an estimate of deployed performance.

After the two-stage design and detector-crop fine-tuning are applied, the
deployed Stage 2 classifier attains **96.49%** on Kaggle crops (n = 940) and
**91.30%** on Roboflow crops (n = 2,254) — a residual domain gap of 5.19 points.
<!-- src: runs/audits/classifier_domain_gap.json#domains.studio.accuracy,domains.field.accuracy -->

![Domain transfer matrix](web/assets/figures/heat_domain_transfer.png)

*Figure 8.2 — Cross-domain transfer accuracy in both directions.*

## 8.3 Detection and Localisation

### 8.3.1 Deployed Detector

The deployed detector is YOLO26m trained at 640 px for 100 epochs.

| Eval set | Split | Precision | Recall | mAP50 | mAP50-95 |
|---|---|---:|---:|---:|---:|
| Original | val | 0.834 | 0.685 | 0.757 | 0.586 |
| **Clean** | **val** | **0.834** | **0.672** | **0.749** | **0.570** |
| Original | test | 0.622 | 0.529 | 0.529 | 0.417 |
| **Clean** | **test** | **0.605** | **0.479** | **0.482** | **0.366** |

<!-- src: runs/audits/detector_clean_val_yolo26m_final100.json -->

*Table 8.6 — Deployed detector, before and after leakage quarantine. Note that
`runs/audits/detector_clean_val.json` records an earlier, smaller backbone and
must not be read as the deployed model.*

Two observations follow.

First, the leakage correction is real but moderate for detection: mAP50 falls
0.7 points on validation and 4.7 points on test. The larger test effect is
consistent with contamination concentrated there. This is smaller than the 9–14
point effect Barz and Denzler (2020) report for CIFAR, which is attributable to
detection metrics being less sensitive to memorisation than whole-image
classification.

Second, and more important, **the validation-to-test gap is large and survives
the audit**: mAP50 falls from 0.749 to 0.482. The clean test split consists of
unseen TACO capture batches, so this gap is genuine domain shift across capture
sessions rather than an artefact. **The test figure, not the validation figure,
is the honest estimate of deployed detection performance.**

A 960 px fine-tune was evaluated to test whether resolution was the binding
constraint. It was not: clean validation mAP50 was essentially unchanged
(0.718 against 0.749) and clean test recall improved from 0.479 to 0.495. The
inference cost was judged not to justify a 1.6-point recall gain.
<!-- src: runs/audits/detector_clean_val_v3_960.json -->

![Detector training curves](web/assets/figures/train_yolo26m_curves.png)

*Figure 8.3 — YOLO26m training curves over 100 epochs at 640 px.*

### 8.3.2 Controlled Cross-Architecture Comparison

The seven-architecture comparison described in §7.4 was executed on rented GPU
infrastructure. **Results were not retrieved before this report was compiled**,
and no numbers are quoted from memory or from earlier non-comparable runs.

| # | Model | Group | mAP50 | mAP50-95 | Train (h) | Status |
|---|---|---|---:|---:|---:|---|
| 1 | YOLO26m | ultralytics | `TBD` | `TBD` | `TBD` | trained, not fetched |
| 2 | YOLOv8m | ultralytics | `TBD` | `TBD` | `TBD` | trained, not fetched |
| 3 | YOLO11m | ultralytics | `TBD` | `TBD` | `TBD` | trained, not fetched |
| 4 | RT-DETR-l | ultralytics | `TBD` | `TBD` | `TBD` | training |
| 5 | Faster R-CNN-R50-FPN | torchvision | `TBD` | `TBD` | `TBD` | training |
| 6 | RetinaNet | torchvision | `TBD` | `TBD` | `TBD` | training |
| 7 | FCOS | torchvision | `TBD` | `TBD` | `TBD` | training |

<!-- BLOCKED: runs/detect/vast_comparison/ does not exist; fetch via scripts/vast/fetch_results.sh before filling -->

*Table 8.7 — Controlled comparison at 30 epochs, 512 px, batch 16. Per §7.4,
only mAP50 and mAP50-95 are comparable across the Ultralytics and torchvision
groups; precision and recall are measured at different operating points.*

When results are fetched, RT-DETR-l's row must be read alongside its disclosed
optimizer deviation (§7.4), and each run's resolved optimizer read from its
`args.yaml`.

## 8.4 End-to-End Production Pipeline

The deployed pipeline — the ungated configuration, per §7.3.5 — was evaluated on
an independent set of 1,042 images used to train neither stage. **Overall
material accuracy is 94.53%.**
<!-- src: runs/audits/pipeline_bin_decision_eval_no_gate.json#materialAccuracy -->

| Ground-truth class | n | % of set | Recall |
|---|---:|---:|---:|
| paper | 514 | 49.3% | 94.9% |
| plastic | 346 | 33.2% | 94.5% |
| metal | 169 | 16.2% | 94.7% |
| cardboard | 9 | 0.9% | 88.9% |
| organic | 4 | 0.4% | 50.0% |
| **glass** | **0** | **0.0%** | **not evaluable** |
| **Macro-average** | | | **84.60%** |

<!-- src: runs/audits/pipeline_bin_decision_eval_no_gate.json#confusion (recalls and macro computed from the confusion matrix) -->

*Table 8.8 — Per-class recall of the deployed pipeline.*

Three classes make up 98.7% of the set and glass does not appear at all. Macro
recall is 84.60%, nearly ten points below the overall figure, and is the more
honest characterisation. Recall on the two rare classes — 88.9% and 50.0% on
supports of 9 and 4 — is too poorly estimated to carry meaning.

### 8.4.1 Bin Routing Is a Degenerate Metric

This is a negative result and it is more useful than the number it replaces.

The routing taxonomy maps five of six materials to Recycling. On this evaluation
set 1,038 of 1,042 images (99.62%) have Recycling as their ground-truth bin and
only 4 (0.38%) have Compost. **A constant predictor that always outputs
Recycling therefore scores 99.62%.**

| Predictor | Bin accuracy |
|---|---:|
| **Constant "Recycling" baseline** | **99.62%** |
| WasteWise, deployed (ungated) | 97.41% |
| WasteWise, superseded gated variant | 96.26% |

<!-- src: runs/audits/pipeline_bin_decision_eval_no_gate.json#binAccuracy; runs/audits/pipeline_bin_decision_eval_baseline.json#binAccuracy -->

*Table 8.9 — Bin-routing accuracy against its majority-class baseline.*

The deployed system scores 2.21 points below the trivial baseline. This is
reported in preference to quoting 97.41% alone, because that figure without its
baseline would misrepresent the system. **On this data the bin-routing metric
does not measure routing competence; it measures how often the system declines
to commit.**

### 8.4.2 A Corrected Protocol

The underlying cause of the absent glass class is that the evaluation script
reduced each image to its single most frequent ground-truth class before
scoring. An image containing six paper items and one glass item was scored only
as paper, and the glass was invisible to the metric. This makes the evaluation
an image-level single-label task rather than the per-object routing task it was
intended to be.

A corrected per-object protocol has been written
(`scripts/eval_pipeline_per_object.py`): it matches predictions to ground truth
by IoU, reports per-class recall across all objects, and prints the
majority-class baseline alongside every accuracy figure. **It has not been
executed, and its results are left to future work.**
<!-- BLOCKED: scripts/eval_pipeline_per_object.py not yet run -->

## 8.5 Deployed Web Application

The system is deployed as a web application at `wastewise-fyp.vercel.app`,
serving the two-stage pipeline over an upload-and-classify interface with
per-object overlays, material labels, confidence scores, and the routed bin.

Measured single-image CPU latency for the deployed detector over 100 images is
138.2 ms mean, 120.8 ms median, and 230.7 ms at the 95th percentile.
<!-- src: runs/audits/cpu_latency_yolo26n_vs_yolo26s.json#deployed_yolo26n_hardneg -->

![Deployed application](web/assets/figures/app_screenshot_1.png)

*Figure 8.4 — Deployed web application: upload, detection overlay, and bin
routing.*

![End-to-end pipeline output on a street scene](web/assets/figures/result_pipeline_street_cans.png)

*Figure 8.5 — Pipeline output on field imagery: detector proposals with Stage 2
material verification.*

## 8.6 Model Comparison and Selection Evidence

### 8.6.1 Classical versus Deep

| Branch | Best model | Accuracy | Macro-F1 |
|---|---|---:|---:|
| Classical, 637 handcrafted features | ExtraTrees | 73.76% | 0.7381 |
| Deep, Stage 2 on clean GT crops | ConvNeXt ensemble (deployed) | 92.93% | 0.9290 |
| Deep, Stage 2 after detector-crop fine-tuning | ConvNeXt ensemble | 93.77% | — |

<!-- src: runs/ml/pca_feature_model_sweep/PCA_Model_Sweep_Report.md; runs/audits/convnext_clean_eval.json#splits.test; runs/dl/convnext_detector_crops_ft/finetune_result.json#finetuned.clean_gt_test_acc -->

*Table 8.10 — Branch A against Branch B on clean evaluation. The classical
branch trails the deployed deep classifier by 19.17 points.*

### 8.6.2 Pipeline Configuration Ablation

Nine pipeline configurations were evaluated against the same 1,042 images.

| Configuration | Material acc | Bin acc |
|---|---:|---:|
| **no_gate (deployed)** | **94.53%** | **97.41%** |
| improved | 94.63% | 95.68% |
| margin20 | 94.53% | 96.16% |
| baseline | 94.43% | 96.26% |
| final3 | 94.43% | 96.26% |
| final | 94.43% | 95.97% |
| no_prior_damping | 94.43% | 95.97% |
| final2 | 94.43% | 94.63% |
| dominant_fix | 93.67% | 96.16% |

<!-- src: runs/audits/pipeline_bin_decision_eval_*.json#tag,materialAccuracy,binAccuracy -->

*Table 8.11 — All nine evaluated pipeline configurations.*

**A methodological caution.** Bin accuracy across these nine ranges from 94.63%
to 97.41%. Reporting the best of nine as though it were a held-out result would
overstate performance, and this is acknowledged rather than concealed. The
deployed configuration is not selected on this metric — §8.4.1 establishes the
metric as degenerate — but on the removal of an untrainable decision layer,
recorded as Failure F17.

### 8.6.3 Detector Backbone Progression

| Backbone | Clean val mAP50 | Clean test mAP50 | Source |
|---|---:|---:|---|
| YOLO26n (v3, 960 px) | 0.718 | 0.476 | `detector_clean_val_v3_960.json` |
| YOLO26s (100 ep) | 0.748 | 0.502 | `detector_clean_val_yolo26s_final100.json` |
| **YOLO26m (100 ep, deployed)** | **0.749** | **0.482** | `detector_clean_val_yolo26m_final100.json` |

<!-- src: runs/audits/detector_clean_val_*.json -->

*Table 8.12 — Detector backbone progression on quarantined-clean splits.*

The progression is worth reading carefully, because it does not support a simple
"bigger is better" narrative. YOLO26m improves clean validation mAP50 by only
0.1 points over YOLO26s, and is 2.0 points **worse** on clean test. The
architectural work across this progression bought considerably less than the
single distribution correction of §8.2.3, which is the chapter's central
comparison.
