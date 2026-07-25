# 9. Failures, Root Causes, and Fixes

This chapter is an honest engineering log. It is included because the project's
most useful findings came from things that went wrong, and because a report that
presents only the final configuration conceals the evidence on which that
configuration was chosen. Seventeen numbered failures were recorded between June
and July 2026; all are summarised below and six are examined in detail.
<!-- src: docs/01_final_report/FAILURES_AND_FIXES.md -->

## 9.1 Summary

| ID | Failure | Severity | Outcome |
|---|---|---|---|
| F1 | Evaluation data leakage across splits | Critical | Quarantined clean eval sets built; all metrics re-measured |
| F2 | Cross-dataset validation broken (reported 17.83%) | High | Fixed; honest domain gap established |
| F3 | PCA report cherry-picked its headline | Medium | Sweep widened to 9 dimensions; report leads with best-per-dimension |
| F4 | Detector recall limited by tiny objects | High | Four remedies tried; baseline retained |
| F4b | Stage 2 train/serve skew | High | **Fixed: +11.97 pp on production distribution** |
| F4c | Tile-augmented training | Info | Tested, rejected — worse on every clean-test metric |
| F5 | Detector operating point never tuned | Medium | Re-swept at serving resolution; conf 0.30 → 0.10 deployed |
| F6 | Label and data hygiene | Low | Quantified; no action required |
| F7 | Environment and infrastructure | Medium | Root-caused; operating rules recorded |
| F8 | YOLOv11 → YOLO26n swap | Info | Documented as marginal |
| F9 | Serving resolution mismatch (640-trained, 960-served) | Medium | Fixed: `YOLO_IMG_SIZE` 960 → 640 |
| F10 | `realworld_v2` was a leaked benchmark | High | 74% leaked; retired from all accuracy claims |
| F11 | Field-rebalance fine-tune | Info | Rejected — gain within noise, studio regressed |
| F12 | Class-agnostic + PlastOPol expansion | Info | Reverted — gain was the confidence lever, not the architecture |
| F13 | Detector confidence lever for field recall | High | Deployed: field recall +8.1 pp with no architecture change |
| F14 | Glass→plastic score-fusion bug | High | Fixed: fusion cap 0.70 → 0.40 |
| F15 | Hard-negative mining | High | Promoted: field small-object recall +16 pp |
| F16 | Waste-gate loosening | Info | Rejected — 6-sample demo flattered it; 270-image check showed ~15 pp precision loss |
| F17 | Waste-state gate (S6) removed | Info | Removed 2026-07-18 as an untrainable decision layer |

*Table 9.1 — Complete failure register.*

## 9.2 F1 — Evaluation Data Leakage

**Symptom: none.** The metrics looked good. This failure was found only because a
systematic risk audit was run against data that no one suspected.

**Root cause.** The merged datasets combine the same source photographs arriving
through different distribution channels — TACO official releases and Roboflow
re-exports of TACO. Roboflow re-encodes and resizes, so byte-level MD5
deduplication cannot match the two copies, and the same photograph landed in
`train` under one name and in `val` or `test` under another. For the classifier
dataset, exact-duplicate cleanup had never been run at all.

**Measured impact.** Quantified in §6.4: 43.0% of `merged_dataset_v5`'s test
split duplicates training data, alongside 33.5%, 20.2% and 19.8% for the other
three datasets.

**Fix.** `scripts/build_clean_eval_splits.py` quarantines every leaked evaluation
image, leaving originals untouched, into clean evaluation sets. All reported
metrics were re-measured against those.

**Correction magnitude.** EfficientNetB0 fell from 94.30% to 91.77% on the clean
classifier test set. The deployed ConvNeXt fell only from 93.88% to 92.93%,
which is itself informative: the deployed model was not substantially a
beneficiary of contamination.
<!-- src: runs/audits/CLASSIFIER_CLEAN_TEST_EVAL.md; runs/audits/convnext_clean_eval.json -->

Counter-intuitively, detector *validation* performance went **up** after
quarantine, because the removed images were hard TACO scenes rather than easy
memorised ones. Leakage does not always flatter a metric, which is an argument
for measuring rather than assuming.

**Lesson.** Content-level perceptual deduplication is mandatory when merging
overlapping community datasets. Filename and byte-level checks are insufficient
by construction, because re-encoding defeats both.

## 9.3 F4b — Train/Serve Skew at the Stage Interface

**Symptom: none visible in any report.** The deployed classifier scored 92.93%
on clean ground-truth crops while scoring 76.91% on the detector's own crops —
the input it actually receives in production. A 16-point skew was invisible
because nothing had ever evaluated the classifier on detector output.

**Root cause.** The classifier trains on ground-truth crops: tight, centred,
always containing an object. The detector supplies loose, off-centre,
occasionally truncated crops that sometimes contain nothing. The system's own
architecture created the distribution mismatch.

**Fix.** A detector-crop dataset was built by IoU-matching detector predictions
to ground-truth labels, retaining unmatched high-confidence predictions as
Background, and the classifier was fine-tuned on a **mixture** of detector crops
and ground-truth crops. Detector-crop accuracy rose 76.91% → 88.88% while clean
ground-truth accuracy also improved, 92.93% → 93.77%.

A detector-crops-**only** fine-tune was rejected by the no-forgetting promotion
gate, with ground-truth accuracy falling to 83%. Mixed-domain training is
therefore required, not incidental.
<!-- src: docs/01_final_report/FAILURES_AND_FIXES.md#F4b; runs/dl/convnext_detector_crops_ft/finetune_result.json -->

**Lesson.** In any cascade, evaluate each stage on the distribution the previous
stage actually emits. This single correction outperformed every architectural
change attempted in the project.

## 9.4 F4c — Tile-Augmented Training, Tested and Rejected

**Hypothesis.** With convergence and data diversity addressed, tiny-object recall
remained at 0.208–0.275. If small objects appeared larger during *training* by
cutting images into overlapping tiles, the model might learn them better.

**Method.** Every original training image was retained and images containing tiny
ground-truth boxes were additionally cut into 2×2 overlapping tiles with
remapped boxes. The training recipe was otherwise identical to the deployed
baseline — 100 epochs, 640 px, batch 16, cosine LR, from the same pretrained
weights — so training data was the only variable.

**Result: tiled loses on every clean-test metric.** mAP50 0.474 → 0.449, mAP50-95
0.348 → 0.335, small-box recall 0.275 → 0.270, and organic recall collapsed
0.109 → 0.022, five times worse. Training on tiles taught a scale and context
distribution that generalises worse to whole images at inference — the opposite
of the intended effect.
<!-- src: runs/audits/TILED_TRAINING_v1.md -->

**Lesson.** An augmentation that changes the scale distribution changes what the
model expects at inference. The tiled weights were not deployed. This is
recorded because a rejected hypothesis with clean measurement is more useful
than an unrecorded one.

## 9.5 F10 and F12 — Two Ways to Fool Yourself About Field Performance

These two are presented together because they share a root cause: crediting an
improvement to the wrong variable.

**F10.** `yolo26_balanced_realworld_v2` was adopted as a "real-world" benchmark
and used to claim field performance. An audit found **74% of it leaked** from
training data. It was retired from all accuracy claims, and subsequent field
work was gated on an external clean evaluation set.

**F12.** A class-agnostic single-class expansion using external PlastOPol data
appeared to produce large field gains. It was reverted: the apparent gain came
almost entirely from the **confidence threshold** that had been changed at the
same time, not from the architectural change. Isolating the confidence lever
alone (F13) reproduced the gain — field recall +8.1 pp, studio +5.3 pp,
small-box +11 pp — with no architecture change at all.
<!-- src: docs/01_final_report/FAILURES_AND_FIXES.md#F10,#F12,#F13 -->

**Lesson.** Match operating points before crediting an architecture. Two
variables moved together and the credit was initially assigned to the
interesting one rather than the responsible one.

## 9.6 F14 — Score Fusion Overriding a Confident Classifier

**Symptom.** Glass objects were routinely predicted as plastic.

**Root cause.** The pipeline blended the detector's material vote with the Stage 2
classifier's prediction. The blend cap allowed the detector's vote to dominate
even when the classifier was confident. Because the detector is heavily
plastic-biased on field imagery — 653 of 771 field boxes predicted plastic,
84.7%, against a per-class field precision of 0.453 — that bias propagated
straight through to the final answer.
<!-- src: runs/audits/yolo26m_conf_gate_alpha_sweep.json#field_class_bias_conf004.class_pred_count -->

**Fix.** The alpha-blend cap was lowered from 0.70 to 0.40, so the detector vote
can no longer override a confident classifier call. Cost: −0.7 pp studio
macro-F1. Benefit: the glass→plastic failure mode was eliminated.

**Lesson.** When a cascade produces a systematic error, check the fusion rule
before blaming either model. Neither stage was individually broken.

## 9.7 F16 and F17 — A Decision Layer That Could Not Be Trained

**F16.** A hypothesis held that the waste-state gate's 0.30 box threshold was
stale and could be loosened. A six-sample demonstration supported this. A
270-image two-domain evaluation showed roughly 15 pp of precision loss. The
change was reverted, and the threshold was confirmed as a real precision guard.

**Lesson.** A six-sample demonstration is not evidence.

**F17.** The waste-state gate (S6) was removed entirely on 2026-07-18. Its
justification had always been the waste-state routing behaviour it enabled — but
**no dataset in the project carries a disposal-state label**, so the gate could
never be trained as a genuine decision layer. It was conservative rule logic
standing in for a model that could not exist given the available data. It cost
approximately 1.15 pp of bin accuracy for no measurable benefit.
<!-- src: docs/01_final_report/FAILURES_AND_FIXES.md#F17; runs/audits/pipeline_bin_decision_eval_no_gate.json -->

The deployed pipeline is now S1–S5, producing material and bin only. This is the
configuration all §8 figures describe.

**Lesson.** A component whose justification cannot be measured is a component
that cannot be defended. Removing it simplified the system and improved the
metric.

## 9.8 F7 — Infrastructure

Recorded briefly because it consumed roughly a day of compute and the operating
rules that resulted are genuine project constraints.

Two Python environments coexisted, with the shell default being a CPU-only
build; GPU work requires the project virtual environment explicitly. An attempt
to install CUDA PyTorch into the global environment hit a full system drive and
left a broken install requiring a full purge. YOLO26n at 640 px batch 32 exhausts
the 12 GB development GPU; batch 16 is the stable configuration. Launching
training through `python -c` deadlocks under Windows spawn-based multiprocessing,
so training must live in a script with a `__main__` guard. RAM caching needs
roughly 24 GB for the 20,000-image set against 17 GB available, so caching is
disabled.
<!-- src: docs/01_final_report/FAILURES_AND_FIXES.md#F7 -->

**Lesson.** Environment failures produce no metrics and no artefacts, so they are
invisible in a results-oriented log unless recorded deliberately.
