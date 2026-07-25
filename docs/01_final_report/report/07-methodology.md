# 7. Methodology

## 7.1 System Overview: Two Branches

The project develops two parallel branches over the same data and taxonomy.

**Branch A** is an explainable classical machine-learning pipeline built on
handcrafted features. It is not the deployed system. Its purpose is to quantify
how much of the material-classification signal is available to engineered
features without representation learning, which makes the contribution of the
deep branch measurable rather than assumed.

**Branch B** is the deployed production system: a localisation-first, two-stage
cascade in which a detector proposes object regions and a second-stage
classifier assigns each region a material. Branch B is what the web application
serves.

The two branches share the six-class material taxonomy — plastic, glass, metal,
paper, cardboard, organic — extended in the classifier by a seventh Background
class used to reject detector false positives.

## 7.2 Branch A: Explainable Classical Machine Learning

Each crop is reduced to 637 handcrafted features in four groups.

| Group | Count | Description |
|---|---:|---|
| Spatial | 8 | Geometric and moment-based descriptors |
| Frequency | 9 | FFT-derived spectral statistics |
| Colour | 44 | Channel statistics and histogram descriptors |
| HOG | 576 | Histogram of oriented gradients |
| **Total** | **637** | |

<!-- src: docs/01_final_report/WasteWise_Conference_Paper_DRAFT.md#3.4; scripts/custom_feature_extractor.py -->

*Table 7.1 — Handcrafted feature representation.*

These features serve two purposes. They form the input to the standalone
classical branch evaluated in §8.1, and they are concatenated with deep backbone
features in the Stage 2 classifier described in §7.3.4. Because 576 of the 637
dimensions are HOG bins, the representation is dominated by sparse, individually
interpretable gradient statistics — a property that turns out to matter for the
dimensionality-reduction result reported in §8.1.

The branch sweeps classical classifiers across PCA-reduced feature spaces from
16 to 384 components and against the untransformed 637 dimensions, on 7,000
training and 2,100 test crops across seven classes.
<!-- src: scripts/pca_feature_model_sweep.py; runs/dl/pca_experiments/ -->

## 7.3 Branch B: Two-Stage, Localisation-First Production Pipeline

### 7.3.1 Rationale

A single-stage detector must localise and classify simultaneously, and its
classification head sees only the features the detection backbone found useful
for localisation. Splitting the problem allows each stage to be trained,
evaluated, and corrected independently, and — critically — allows the second
stage to reject the first stage's false positives through an explicit Background
class. That rejection mechanism is what makes the two-stage design improve
precision over single-stage detection.

### 7.3.2 Stage 1: Detection

Stage 1 is a YOLO26 detector trained on the six-class hard-case dataset at
640 px for 100 epochs. The backbone was promoted progressively — YOLO26n to
YOLO26s to YOLO26m — with the confidence threshold, decision gate, and fusion
parameter re-swept at each promotion rather than carried across model sizes.
Carrying an operating point across a backbone change is the specific error that
§9 records as Failure F13.
<!-- src: docs/01_final_report/FAILURES_AND_FIXES.md#F13 -->

A 960 px fine-tune was evaluated on the YOLO26n backbone to test whether input
resolution was the binding constraint on recall. It was not: validation
performance was essentially unchanged and test recall improved by 3.9 points, at
an inference cost judged not to justify the gain. This is reported as evidence
that training-data diversity, not resolution, limits the present system.
<!-- src: runs/audits/detector_clean_val_v3_960.json; docs/01_final_report/FAILURES_AND_FIXES.md#F9 -->

### 7.3.3 Stage 2: Material Verification

Stage 2 classifies each detector-proposed crop into one of the six material
classes or Background. The architecture is a frozen ImageNet-pretrained backbone
whose pooled features are concatenated with the 637 handcrafted features of
§7.2 and passed to a multilayer perceptron head.

Backbone selection compared ConvNeXtV2-Tiny, Swin-Tiny, and EfficientNetV2-S
under identical conditions; the comparison and its outcome are reported in §8.2.

### 7.3.4 Detector-Crop Fine-Tuning

This is the central methodological step of the project.

A classifier trained on ground-truth crops observes tightly bounded, correctly
centred objects. In deployment it receives detector output: crops that are
loosely bounded, off-centre, sometimes truncated, and sometimes containing no
object at all. These are different distributions, and the mismatch is created by
the architecture of the system itself rather than by any deficiency in the data.

A training set is therefore constructed from the detector's own output crops on
the training images, and Stage 2 is fine-tuned on a mixture of detector crops
and ground-truth crops. A promotion gate accepts the fine-tuned model only if it
improves detector-crop accuracy **without** regressing clean ground-truth
accuracy, so the procedure cannot trade one distribution against the other
silently.
<!-- src: scripts/finetune_stage2_on_detector_crops.py; runs/dl/convnext_detector_crops_ft/finetune_result.json#promote -->

### 7.3.5 Routing and the Removed Decision Gate

Per-object predictions are aggregated by area × confidence into a single
dominant material per image, which maps to a disposal bin: plastic, glass,
metal, paper and cardboard route to Recycling; organic routes to Compost; an
unknown or Background-dominant image routes to Review as the default.

$$
m^{*} = \arg\max_{c \in C} \sum_{\hat{y}_i = c} a_i s_i
$$

*Equation (5)*

where $a_i$ is the area of detection $i$, $s_i$ its confidence, and $\hat{y}_i$
its Stage 2 material label.

An earlier design added a waste-state gate (S6) that additionally marked each
object `waste`, `not_waste`, or `review`, withholding a bin route for anything
the gate was unsure of. **That gate was removed on 2026-07-18.** It was
conservative rule logic standing in for a trained state model the project never
had the data to train, and §8.6 shows it cost bin accuracy with no measurable
offsetting benefit. The deployed system is therefore the ungated configuration,
and all deployed figures in §8 are reported for that configuration.
<!-- src: docs/01_final_report/FAILURES_AND_FIXES.md#F17; runs/audits/pipeline_bin_decision_eval_no_gate.json -->

A related bias was quantified during confidence-gate tuning: on field imagery
the detector's own class vote assigned 653 of 771 boxes to plastic — 84.7% —
against a per-class field precision of 0.453 for that class. The detector's
material vote is therefore not trusted over the Stage 2 classifier, which is the
fix recorded as Failure F14.
<!-- src: runs/audits/yolo26m_conf_gate_alpha_sweep.json#field_class_bias_conf004.class_pred_count -->

## 7.4 Controlled Cross-Architecture Comparison

Seven detectors are trained under a single shared configuration to isolate the
effect of architecture: YOLOv8m, YOLO11m, YOLO26m, and RT-DETR-l via
Ultralytics, and Faster R-CNN-R50-FPN, RetinaNet, and FCOS via torchvision.

The shared budget is **30 epochs at 512 px, batch 16**.
<!-- src: scripts/vast/run_comparison.py#parse_args defaults and module docstring -->
This differs from the deployed detector's 640 px, 100-epoch configuration, and
the two must not be conflated. One consequence is stated plainly: YOLO26m
appears twice in this report with different figures — once as the deployed
100-epoch model, and once inside this 30-epoch comparison. The lower figure
reflects the reduced budget and is not a regression.

The equal-budget constraint is the design, not a limitation of it. Comparing a
30-epoch challenger against the deployed 100-epoch model would confound
architecture with training duration.

**Disclosed deviations from the shared configuration.** Honesty about the
comparison requires stating where it is not perfectly identical:

- `OPTIM_OVERRIDES` switches RT-DETR-l to AdamW at `lr0` 1e-4. The Ultralytics
  default learning rate diverged it to NaN at epoch 4 on two consecutive
  attempts, and RT-DETR's own published recipe uses AdamW at approximately this
  rate. This is a genuine deviation and RT-DETR-l's row must be read with it in
  mind.
- `BATCH_OVERRIDES` sets RT-DETR-l to batch 16. At the documented command this
  is **inert**, because the shared default is already batch 16; the override
  only takes effect if a larger batch is requested. All seven models therefore
  train at batch 16.
- No optimizer is set explicitly for the remaining six models, so they inherit
  the Ultralytics `optimizer='auto'` policy. The resolved optimizer and learning
  rate for each run must be read from that run's `args.yaml` rather than assumed
  from the script.
<!-- src: scripts/vast/run_comparison.py#BATCH_OVERRIDES,OPTIM_OVERRIDES -->

Every model is initialised from COCO-pretrained weights with only its
classification head replaced, so the comparison is not confounded by differing
quantities of pretrained signal.

**Two measurement caveats.** First, precision and recall are not measured at the
same operating point across groups: Ultralytics reports them at a per-class
best-F1 threshold, whereas the torchvision models are reported at a fixed
confidence of 0.5 with greedy class-matched assignment at IoU 0.5. Precision and
recall are comparable *within* each group but not across them. Mean average
precision integrates over thresholds and is unaffected, so **mAP50 and mAP50-95
are the only quantities used for cross-architecture claims.** Second, all seven
share the data split, augmentation settings, and random seed.

## 7.5 Evaluation Protocol

Evaluation is conducted at four levels:

1. **Detector** — on quarantined-clean validation and test splits (§6.4).
2. **Classifier** — on clean ground-truth crops and, separately, on detector
   output crops, since §7.3.4 establishes these as different distributions.
3. **Cross-domain transfer** — train on one acquisition domain, test on the
   other, in both directions.
4. **End-to-end** — on an independent set of 1,042 images used to train neither
   stage.

Two reporting rules apply throughout. For every class-imbalanced evaluation,
macro-averaged metrics are reported alongside overall accuracy. For every
routing metric, the majority-class baseline is reported alongside the system's
figure:

$$
\text{acc}_{\text{macro}} = \frac{1}{|C|}\sum_{c \in C} \frac{\mathrm{TP}_c}{n_c}
\qquad
\text{baseline} = \max_{c \in C} \frac{n_c}{N}
$$

*Equation (6)*

§8.4 demonstrates why the second rule is not optional: on the available
evaluation data the routing metric is degenerate, and quoting it without its
baseline would materially misrepresent the system.
