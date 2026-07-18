# WasteWise: A Two-Stage Detection and Verification Pipeline for Waste Material Classification Under Domain Shift

**Khoa Phung**

---

## Abstract

Automated waste sorting systems are typically trained and evaluated on curated
studio datasets in which a single object is photographed against a uniform
background. Such systems degrade substantially when deployed on real-world
imagery. This paper presents WasteWise, a two-stage pipeline that separates
localization from material verification. A YOLO26m detector proposes object
regions, and a ConvNeXt-based classifier augmented with 637 handcrafted
features assigns a material class to each region. The principal methodological
finding is that training the second-stage classifier on the detector's own
output crops, rather than on ground-truth crops, raises accuracy on the
production distribution from 76.91% to 88.88% while also improving clean
ground-truth accuracy from 92.93% to 93.77%. The domain-shift problem
motivating this design is quantified directly: a classifier trained on studio
images loses 40.19 percentage points of accuracy when evaluated on field images
of the same seven material classes. All reported figures are computed on
evaluation splits from which cross-split duplicates have been removed by
perceptual-hash auditing. A total of 918 validation and 457 test images were
quarantined from the detection set, together with 2,406 images from the
classification test set, and both pre-audit and post-audit figures are
reported. On an independent evaluation set the deployed system attains 94.43%
material accuracy. However, this paper also demonstrates that the corresponding
bin-routing metric is uninformative on the available data, because the routing
taxonomy maps five of six materials to a single bin. This negative result is
reported explicitly, and a corrected per-object evaluation protocol is
provided.

**Keywords:** waste classification, object detection, domain shift, dataset
leakage, two-stage pipeline, evaluation methodology

---

## 1. Introduction

Municipal waste sorting remains substantially manual. Automating the
identification of material type from images is therefore an attractive target
for computer vision, and a considerable literature has developed around the
task. Much of that literature reports high accuracy, frequently above 90%, on
datasets such as TrashNet (Thung & Yang, 2016), in which each image contains a
single object, centred against a white background under controlled lighting.

These conditions do not describe waste as it is encountered in practice. Real
waste appears in cluttered scenes, at varying scales, partially occluded,
against arbitrary backgrounds, and frequently deformed or soiled. The
discrepancy between the two settings is not incidental. Section 4.5 measures it
directly and finds that a classifier achieving 80.00% in-domain accuracy on
studio images falls to 39.81% on field images of the same classes, a reduction
of 40.19 percentage points.

This paper describes WasteWise, a deployed system constructed around that
observation, and reports the findings that emerged during its development. The
contributions are fourfold.

First, a two-stage detection and verification architecture is presented that
separates the task of locating objects from the task of identifying their
material, allowing each stage to be trained on the distribution it will
encounter in deployment (Sections 3.3 to 3.5).

Second, a training-distribution correction is introduced. Fine-tuning the
second-stage classifier on crops produced by the first-stage detector, rather
than on ground-truth crops, improves accuracy on the production distribution by
11.97 percentage points, from 76.91% to 88.88%, without architectural
modification and without sacrificing clean-crop accuracy (Section 4.3). This
constitutes the principal result.

Third, a leakage-audited evaluation protocol is described. The initial
evaluation splits were found to contain cross-split near-duplicates arising
from the same source photographs entering multiple community datasets.
Following the methodology of Barz and Denzler (2020), these are detected by
perceptual hashing and quarantined, and only corrected figures are reported
alongside the pre-audit values (Sections 3.2 and 4.1).

Fourth, a negative result concerning bin-routing evaluation is reported. The
bin-routing accuracy commonly used to characterize such systems is shown to be
degenerate on the available evaluation data, because the routing taxonomy
collapses five of six material classes into a single bin. On the evaluation set
used here, a constant predictor achieves 99.62%, exceeding the 96.26% attained
by the deployed system. This finding is reported in place of the headline
figure, and a corrected per-object protocol is supplied (Section 4.6).

The fourth contribution is atypical for a paper of this kind, and it is
included deliberately. The project documentation for this system initially
reported the 96.26% figure as a system result. It is not one.

---

## 2. Literature Review

### 2.1 Waste image datasets

The most widely used benchmark for waste material classification is TrashNet
(Thung & Yang, 2016), developed as a Stanford CS229 project. It comprises 2,527
RGB images across six classes, namely glass (501), paper (594), cardboard
(403), plastic (482), metal (410) and general trash (137), each photographed
against a white posterboard under natural or room lighting. This construction
makes TrashNet a clean classification benchmark and, simultaneously, a poor
proxy for deployment conditions, since the uniform background removes precisely
the contextual difficulty that characterizes real classification.

TACO, or Trash Annotations in Context (Proença & Simões, 2020), was introduced
to address this limitation. TACO contains 1,500 images with 4,784 annotations
spanning 60 litter categories organized under 28 supercategories, captured in
natural settings ranging from beaches to urban streets, with instance
segmentation masks. The authors report Mask R-CNN performance and note
explicitly that the dataset requires considerably more manual annotation before
satisfactory in-the-wild detection becomes achievable. The realism of TACO
makes it valuable, while its scale renders it insufficient in isolation. Its
long-tailed 60-class taxonomy additionally requires mapping before it can be
used with a coarse material taxonomy.

Subsequent work has expanded the available data. RealWaste provides authentic
landfill reception imagery, and various community-hosted datasets aggregate and
re-annotate existing sources. Such aggregation is convenient and, as discussed
in Section 2.4, hazardous. The same source photographs propagate into multiple
derived datasets, so combining datasets can silently place duplicates of the
same image on both sides of a train and test split.

### 2.2 Classification approaches

Early approaches to waste classification employed handcrafted features with
classical classifiers. The original TrashNet work (Thung & Yang, 2016) paired
SIFT features with a support vector machine, alongside a convolutional neural
network that reached approximately 75% test accuracy. Histogram of Oriented
Gradients descriptors (Dalal & Triggs, 2005), introduced for pedestrian
detection, remain a common texture and edge representation in this setting,
because material identity is substantially a texture problem. The distinction
between paper and plastic frequently depends more on surface microstructure
than on shape.

Contemporary work is dominated by transfer learning from ImageNet-pretrained
backbones. ConvNeXt (Liu et al., 2022) modernizes a standard ResNet toward
vision-transformer design principles and reaches 87.8% ImageNet top-1 accuracy
while retaining the efficiency of a purely convolutional network. It is adopted
here as the second-stage backbone. Section 4.2 reports the comparison against
EfficientNetV2-S and Swin-Tiny that motivated this selection.

Recent surveys of deep learning for waste classification report a broad range
of architectures applied to this task, with reported accuracies clustering
between 85% and 95%. These figures are difficult to compare across studies,
because the underlying evaluation sets differ in realism, class taxonomy and,
as argued below, duplicate contamination.

### 2.3 Detection architectures

The YOLO family remains the dominant choice for real-time waste detection, with
recent literature applying successive versions to garbage detection tasks. This
work uses YOLO26 (Sapkota & Karkee, 2025; Ultralytics, 2025), released in
September 2025, which removes non-maximum suppression and Distribution Focal
Loss from the inference path in favour of native end-to-end inference, and
introduces the MuSGD optimizer, progressive loss and STAL label assignment. The
reported improvements on small objects are directly relevant to the present
setting, in which litter frequently occupies a small fraction of the frame.

A recurring architectural question concerns whether waste understanding should
be treated as single-stage detection with material classes, or as two-stage
detection followed by classification. The single-stage formulation is simpler
and faster. The two-stage formulation is adopted here for a specific reason
developed in Section 3.5: localization and material identification impose
different data requirements, and coupling them obliges a single model to
satisfy both from one training distribution.

### 2.4 Evaluation integrity

Barz and Denzler (2020) demonstrated that the CIFAR-10 and CIFAR-100 test sets
contain near-duplicates of training images, at rates of 3.3% and 10%
respectively, and that removing them reduces measured classification accuracy
by 9 to 14 percentage points. Their conclusion is that reported accuracy on
contaminated benchmarks partly measures memorization rather than
generalization. They released duplicate-free replacements, designated ciFAIR,
to address the problem.

This finding applies directly to waste classification, and appears to be
under-appreciated within the domain. Waste datasets are frequently assembled by
aggregating community sources, and those sources overlap. A model trained on
such an aggregate and tested on a random split of it will report inflated
figures. Section 3.2 describes the audit performed here, and Section 4.1
reports its effect. No prior waste-classification work known to the author
reports a perceptual-hash leakage audit of its evaluation splits, and it is
suggested that this should become standard practice.

### 2.5 Domain shift

The discrepancy between curated and deployed performance constitutes a specific
instance of domain shift. In the waste setting this admits an unusually clean
experimental treatment, since several datasets are unambiguously studio in
character, comprising uniform backgrounds and single objects, while others are
unambiguously field-based, comprising natural scenes, and both share a material
taxonomy. This permits direct measurement of cross-domain transfer, reported in
Section 4.5.

### 2.6 Gap addressed by this work

The existing literature establishes that studio benchmarks overstate deployed
performance, that duplicate contamination overstates benchmark performance, and
that two-stage pipelines constitute a plausible architecture for cluttered
scenes. What remains absent is a treatment of the interface between the two
stages. When a classifier is trained on ground-truth crops but deployed on
detector output crops, it encounters a distribution shift that is internal to
the system and entirely self-inflicted. Detector crops are misaligned, loosely
bounded, and include false positives that ground-truth crops never contain.
Section 4.3 quantifies this effect and demonstrates that it constitutes the
largest single lever available.

---

## 3. Methodology

### 3.1 Data

The system draws on multiple sources, mapped into a six-class material taxonomy
comprising plastic, glass, metal, paper, cardboard and organic, together with a
Background class used by the classifier to reject detector false positives.

| Source | Role | Character |
|---|---|---|
| TACO (Proença & Simões, 2020) | Detection training, field evaluation | Real-world litter in context |
| TrashNet (Thung & Yang, 2016) | Classification training | Studio, single-object |
| RealWaste | Classification training | Landfill reception imagery |
| Community sets | Detection training | Aggregated, re-annotated |
| Independent test set | System evaluation | Held out from all training |

The final detection dataset is a source-aware merge emphasizing hard cases,
comprising small objects, cluttered scenes and materials that are visually
confusable. The second-stage classifier training set is constructed from
detector output crops with per-class caps, specifically 6,000 each for plastic,
glass, cardboard, metal and organic, 4,136 for paper, and 3,000 for Background.
These caps reflect availability rather than a designed balance.

### 3.2 Leakage audit

Because several sources aggregate overlapping community data, all splits were
audited for cross-split duplication before any figure was reported. Perceptual
hashes were computed for every image, and pairs with small Hamming distance
across split boundaries were flagged. The audit identified both exact
duplicates, at Hamming distance 0, and near-duplicates, at distance 4 or below,
between training and test partitions. These included cases in which the same
photograph had entered the corpus under two different source prefixes at
Hamming distance 0.

| Split | Kept | Quarantined |
|---|---:|---:|
| Detection validation | 2,620 | 918 |
| Detection test | 906 | 457 |
| Classification test | 3,194 | 2,406 |

All figures in Section 4 are computed on the quarantined-clean splits unless
labelled as original. Both are reported so that the magnitude of the correction
remains visible.

### 3.3 Stage 1: detection

The first stage is a YOLO26 detector trained on the six-class hard-case dataset
at 640 px for 100 epochs. The backbone was promoted progressively, from YOLO26n
to YOLO26s and subsequently to YOLO26m, with the confidence threshold, decision
gate and fusion parameter re-swept at each step rather than transferred across
model sizes.

A 960 px fine-tune was additionally evaluated on the YOLO26n backbone in order
to determine whether input resolution constituted the binding constraint. It did
not. Validation performance was unchanged and test recall improved by only 3.9
points, which was judged insufficient to justify the associated inference cost.
This is reported as evidence that training-data diversity, rather than
resolution, limits the present system.

### 3.4 Handcrafted feature representation

A total of 637 handcrafted features are extracted per crop.

| Group | Count | Description |
|---|---:|---|
| Spatial | 8 | Geometric and moment-based descriptors |
| Frequency | 9 | FFT-derived spectral statistics |
| Colour | 44 | Channel statistics and histogram descriptors |
| HOG (Dalal & Triggs, 2005) | 576 | Histogram of oriented gradients |
| **Total** | **637** | |

These features serve two purposes. They are concatenated with deep backbone
features in the second-stage classifier, and they support a standalone
classical branch, reported in Section 4.4, which provides interpretable
evidence regarding which signal types carry material information.

### 3.5 Stage 2: material verification

The second stage classifies each detector-proposed crop into one of the six
material classes or into Background. The architecture consists of a frozen
ImageNet-pretrained backbone whose pooled features are concatenated with the 637
handcrafted features and passed to a multilayer perceptron head. The Background
class is load-bearing, since it permits the second stage to reject first-stage
false positives. This constitutes the mechanism by which the two-stage design
improves precision relative to single-stage detection.

### 3.6 Detector-crop fine-tuning

This constitutes the central methodological step of the present work. A
classifier trained on ground-truth crops observes tightly bounded, correctly
centred objects. In deployment it receives detector output, comprising crops
that are loosely bounded, off-centre, occasionally truncated, and occasionally
containing no object at all. These are different distributions, and the
mismatch is created by the architecture of the system itself.

A training set is therefore constructed from the detector's own output crops on
the training images, and the second-stage classifier is fine-tuned on a mixture
of these detector crops and ground-truth crops. A promotion gate accepts the
fine-tuned model only if it improves detector-crop accuracy without regressing
clean ground-truth accuracy.

### 3.7 Routing

Per-object predictions are aggregated (area x confidence) into a single dominant
material per image, which is then mapped to a disposal bin as follows. Plastic,
glass, metal, paper and cardboard route to Recycling; organic routes to Compost;
and an unknown or Background-dominant image routes to Review as the default.
Section 4.6 examines the implications of this taxonomy for evaluation, and the
conclusion is not favourable.

An earlier design added a decision gate (S6) that additionally marked each object
`waste`, `not_waste` or `review` and withheld a bin route for anything the gate
was unsure of. That gate was removed on 2026-07-18: it was conservative rule
logic standing in for a trained state model the project never had the data to
train, and Section 4.6 shows it cost bin accuracy without a measurable offsetting
benefit. The deployed system therefore now corresponds to the ungated variant.

### 3.8 Evaluation protocol

Evaluation is conducted at four levels. The detector is evaluated on
quarantined-clean validation and test splits. The classifier is evaluated on
clean ground-truth crops and on detector output crops. Cross-domain transfer is
assessed by training on one acquisition domain and testing on the other. The
complete system is evaluated end-to-end on an independent test set used to train
neither stage.

For all class-imbalanced evaluations, macro-averaged metrics are reported
alongside overall accuracy, and for the routing metric the majority-class
baseline is reported. Section 4.6 explains why the latter is necessary.

---

## 4. Results

### 4.1 Detector performance and the effect of the leakage audit

| Eval set | Split | Precision | Recall | mAP50 | mAP50-95 |
|---|---|---:|---:|---:|---:|
| Original | val | 0.834 | 0.685 | 0.757 | 0.586 |
| **Clean** | **val** | **0.834** | **0.672** | **0.749** | **0.570** |
| Original | test | 0.622 | 0.529 | 0.529 | 0.417 |
| **Clean** | **test** | **0.605** | **0.479** | **0.482** | **0.366** |

Two observations follow. First, the leakage correction is real but moderate for
the detector, with mAP50 falling by 0.7 points on validation and 4.7 points on
test. The larger effect on the test set is consistent with contamination being
concentrated there. This magnitude is smaller than the 9 to 14 point effect
reported by Barz and Denzler (2020) for CIFAR, which is attributed here to
detection metrics being less sensitive to memorization than whole-image
classification.

Second, and of greater importance, the validation-to-test gap is substantial
and survives the audit, with mAP50 falling from 0.749 to 0.482. The clean test
split consists of unseen TACO capture batches, so this gap represents genuine
domain shift across capture sessions rather than an artifact. The test figure
is therefore regarded as the honest estimate of deployed detection performance.

### 4.2 Backbone selection for Stage 2

| Backbone | Feature dim | Train (s) | Val acc | Test acc | Test macro-F1 |
|---|---:|---:|---:|---:|---:|
| ConvNeXtV2-Tiny | 1,405 | 17.05 | 0.9325 | 0.9455 | 0.9459 |
| Swin-Tiny | 1,405 | 29.22 | 0.9448 | **0.9458** | **0.9464** |
| EfficientNetV2-S | 1,917 | 24.47 | 0.7925 | 0.8023 | 0.7989 |

Swin-Tiny and ConvNeXtV2-Tiny are statistically indistinguishable on test
accuracy, differing by 0.03 points across 2,696 samples. The ConvNeXt family
was selected on the basis of training cost, at 17.05 s against 29.22 s under
identical conditions, rather than on accuracy. This is stated explicitly rather
than presenting a performance justification that the data does not support.
EfficientNetV2-S underperforms both alternatives by approximately 14 points and
was discarded.

### 4.3 Detector-crop fine-tuning: principal result

| Model | Detector-crop accuracy | Clean GT-crop accuracy |
|---|---:|---:|
| Baseline, trained on GT crops | 76.91% | 92.93% |
| **Fine-tuned on detector crops** | **88.88%** | **93.77%** |
| **Difference** | **+11.97 pp** | **+0.84 pp** |

Training the classifier on the distribution it actually encounters improves
production accuracy by 11.97 percentage points. Clean ground-truth accuracy
does not regress, but improves slightly, so this represents a strict
improvement rather than a trade-off between the two distributions.

The magnitude of this effect relative to the architectural modifications tested
merits emphasis. Promoting the detector backbone from YOLO26n to YOLO26m,
increasing input resolution to 960 px, and substituting the second-stage
backbone across three modern architectures each produced changes of a few
points or fewer. Correcting the training distribution at the stage interface
produced twelve. The macro-F1 of the fine-tuned model on detector crops is
0.7989, notably below its accuracy, which indicates that the gain is not
uniform across classes.

### 4.4 Classical machine learning branch

Employing the 637 handcrafted features with classical classifiers on 7,000
training and 2,100 test crops across seven classes yields the following
results. ExtraTrees is the strongest classical model at every dimensionality.

| Feature space | Explained variance | Best model | Accuracy |
|---|---:|---|---:|
| PCA-16 | 99.33% | ExtraTrees | 59.76% |
| PCA-32 | 99.60% | ExtraTrees | 65.86% |
| PCA-64 | 99.78% | ExtraTrees | 67.71% |
| PCA-128 | 99.90% | ExtraTrees | 67.81% |
| PCA-256 | 99.97% | ExtraTrees | 65.76% |
| PCA-384 | 99.99% | ExtraTrees | 63.67% |
| **Full 637-D** | **100.00%** | **ExtraTrees** | **73.76%** |

The full 637-dimensional representation reaches 73.76% accuracy, with a
macro-F1 of 0.7381, approximately 20 points below the deep second-stage
classifier. This quantifies the contribution of the learned representation
relative to engineered features.

**An anomaly requiring explanation.** Accuracy is non-monotonic in PCA
dimensionality. Performance peaks at 128 components, at 67.81%, declines
through 384 components, at 63.67%, and then rises to 73.76% at the full 637
dimensions. Since 384 components already capture 99.99% of the variance, a
10-point gap between PCA-384 and the untransformed features cannot be
attributed to information loss.

The working explanation advanced here is that the PCA rotation destroys
axis-aligned structure that tree ensembles exploit. Of the 637 features, 576
are HOG bins, which are individually interpretable and sparse, and ExtraTrees
splits on individual features. A dense rotation of this space produces
components that are linear combinations of many bins, which axis-aligned splits
handle poorly. Under this account, the full-dimensional result reflects
better-aligned information rather than a greater quantity of information.

It should be noted that this explanation has not been verified experimentally,
and that an alternative explanation, namely an inconsistency between the
preparation of the full-dimensional and PCA-transformed conditions such as
differing normalization, has not been excluded. This result should accordingly
be treated as provisional.

### 4.5 Cross-domain generalization

Training on one acquisition domain and evaluating on the other, with both
domains sharing all seven classes, yields the following.

| Direction | In-domain | Cross-domain | Gap | Cross macro-F1 |
|---|---:|---:|---:|---:|
| Studio to Field | 80.00% | 39.81% | **40.19 pp** | 0.4023 |
| Field to Studio | 72.76% | 44.49% | 28.27 pp | 0.4074 |

A studio-trained classifier retains less than half of its accuracy on field
imagery. The asymmetry is informative. Field-trained models transfer to studio
conditions more successfully, losing 28.27 percentage points, than the reverse,
which loses 40.19. Field data contains the variation that studio data lacks,
including occlusion, background clutter, lighting variation and deformation, so
a field-trained model has encountered conditions resembling the studio case,
whereas the converse does not hold.

The practical implication is that field data carries greater value per image
than studio data for this task, and that reported accuracy on studio benchmarks
should not be interpreted as an estimate of deployed performance.

Separately, the deployed second-stage classifier evaluated across both domains
attains 96.49% on studio crops, with n = 940, and 91.30% on field crops, with
n = 2,254. This represents a residual gap of 5.19 points after the two-stage
design and detector-crop fine-tuning have been applied.

### 4.6 End-to-end system evaluation and a negative result

The deployed pipeline was evaluated on an independent test set of 1,042 images
used to train neither stage. Overall material accuracy is 94.43%. The class
distribution, however, is severely imbalanced.

| Ground-truth class | n | % of set | Recall |
|---|---:|---:|---:|
| paper | 514 | 49.3% | 94.9% |
| plastic | 346 | 33.2% | 93.9% |
| metal | 169 | 16.2% | 95.9% |
| cardboard | 9 | 0.9% | 77.8% |
| organic | 4 | 0.4% | 50.0% |
| **glass** | **0** | **0.0%** | **not evaluable** |
| **Macro-average** | | | **82.50%** |

Three classes constitute 98.7% of the set, and glass does not appear at all.
The macro-averaged recall is 82.50%, twelve points below the overall figure.
The macro-average is regarded here as the more honest characterization. Recall
on the two rare classes, at 77.8% and 50.0% on supports of 9 and 4
respectively, is too poorly estimated to carry meaning.

**Bin routing as a degenerate metric.** The routing taxonomy maps five of six
materials to Recycling. On this evaluation set, 1,038 of 1,042 images, or
99.62%, have Recycling as their ground-truth bin, and only 4, or 0.38%, have
Compost. A constant predictor that always outputs Recycling therefore achieves
99.62% bin accuracy. The deployed system achieves 96.26%, which is 3.36 points
below the trivial baseline. The deficit arises because the decision gate routes
some correctly detected waste to Review, which the constant predictor never
does.

| Predictor | Bin accuracy |
|---|---:|
| **Constant Recycling baseline** | **99.62%** |
| WasteWise, ungated variant | 97.41% |
| WasteWise, deployed and gated | 96.26% |

This is reported in preference to the 96.26% figure alone, because that figure,
presented without its baseline, would misrepresent the system. The bin-routing
metric does not measure routing competence on this data. It measures how
frequently the system declines to commit.

**A methodological caution regarding configuration selection.** Nine pipeline
configurations were evaluated against these same 1,042 images, with bin
accuracy ranging from 94.63% to 97.41%. Reporting the best of nine as a
held-out result would overstate performance. It is noted further that the
ungated configuration, at 97.41%, outperforms the deployed gated configuration,
at 96.26%, on this metric. The justification for the gate must therefore rest
on the waste-state routing behaviour it enables rather than on bin accuracy.

> **Update (2026-07-18).** The waste-state gate (S6) was subsequently removed: the
> project had no dataset with which to train it as a genuine decision layer, so its
> only justification — the waste-state routing behaviour noted above — could not be
> substantiated. With the gate gone, the deployed system is the ungated
> configuration (97.41% on this metric). The gated figures above are retained as the
> evidence that motivated the removal. The analysis of the metric itself as
> degenerate on this label distribution is unaffected.

**Corrected protocol.** The underlying cause of the absent glass class is that
the evaluation script reduced each image to its single most frequent
ground-truth class before scoring. An image containing six paper items and one
glass item was scored only as paper, and the glass was invisible to the metric.
This renders the evaluation an image-level single-label task rather than a
per-object routing task. A corrected per-object protocol is provided, which
matches predicted to ground-truth boxes by intersection over union, reports
per-class recall across all objects, and prints the majority-class baseline
alongside every accuracy figure. Results from this protocol are not yet
available and are left to future work.

---

## 5. Discussion and Limitations

**What the evidence supports.** The detector-crop fine-tuning result reported in
Section 4.3 constitutes the most robust finding presented here. It exhibits a
large effect, a like-for-like comparison on a fixed evaluation set, a clear
causal mechanism, and no selection across multiple configurations. The
domain-shift measurements in Section 4.5 are similarly direct.

**What the evidence does not support.** Three limitations are material. First,
bin-routing performance is unmeasured. Section 4.6 establishes that no
defensible claim regarding routing accuracy can presently be made, because the
taxonomy is too coarse relative to the available evaluation data and because
the evaluation protocol discarded minority objects. This constitutes the
largest gap in the work. Second, rare-class performance is unknown. Glass,
organic and cardboard are essentially absent from the independent evaluation
set, so claims regarding six-class performance rest on evidence for three
classes. Third, the PCA result is provisional, as discussed in Section 4.4.

**Threats to validity.** The leakage audit employs perceptual hashing, which
detects near-duplicate images but not the subtler case of the same physical
object photographed twice from different angles. Residual contamination of this
kind remains possible. Additionally, the clean test split for the detector
derives from TACO capture batches, and performance on a genuinely different
geography or waste stream remains untested.

---

## 6. Conclusion

This paper has presented WasteWise, a two-stage waste detection and material
classification pipeline, and has reported its construction and evaluation with
particular emphasis on measurement integrity.

The principal finding is that the interface between pipeline stages constitutes
a larger performance lever than the choice of architecture at either stage.
Fine-tuning the material classifier on the detector's own output crops improved
production-distribution accuracy from 76.91% to 88.88%, an effect several times
larger than any produced by changing backbone or input resolution. Systems
constructed as detector-and-classifier cascades should therefore train the
classifier on detector output as a matter of routine practice.

The domain shift motivating this design was further quantified. Studio-trained
classifiers lose 40.19 points of accuracy on field imagery, and field data
transfers to studio conditions considerably more successfully than the reverse.

Finally, two negative results were reported that are arguably more useful than
the positive figures they replace. Cross-split duplicate contamination inflated
the initial evaluation figures, and correcting it reduced detector test mAP50
by 4.7 points. Bin-routing accuracy, the metric most naturally employed to
characterize such a system, was shown to be degenerate on the available data,
since a constant predictor outperforms the pipeline because the routing
taxonomy assigns five of six materials to a single bin. The majority-class
baseline is reported alongside the metric, and a corrected per-object
evaluation protocol is supplied.

Future work should prioritize the construction of an evaluation set with
genuine representation of glass, organic and cardboard, together with
re-evaluation under the per-object protocol. Until such a set exists, the
honest characterization of this system is 82.50% macro-averaged material recall
across three well-represented classes, with routing performance unmeasured.

---

## References

Barz, B., & Denzler, J. (2020). Do we train on test data? Purging CIFAR of
near-duplicates. *Journal of Imaging*, 6(6), 41.

Dalal, N., & Triggs, B. (2005). Histograms of oriented gradients for human
detection. In *Proceedings of the IEEE Computer Society Conference on Computer
Vision and Pattern Recognition (CVPR)* (pp. 886–893). San Diego, CA.

Liu, Z., Mao, H., Wu, C.-Y., Feichtenhofer, C., Darrell, T., & Xie, S. (2022).
A ConvNet for the 2020s. In *Proceedings of the IEEE/CVF Conference on Computer
Vision and Pattern Recognition (CVPR)* (pp. 11976–11986). New Orleans, LA.

Proença, P. F., & Simões, P. (2020). TACO: Trash annotations in context for
litter detection. *arXiv preprint arXiv:2003.06975*.

Sapkota, R., & Karkee, M. (2025). Ultralytics YOLO evolution: An overview of
YOLO26, YOLO11, YOLOv8 and YOLOv5 object detectors. *arXiv preprint
arXiv:2510.09653*.

Thung, G., & Yang, M. (2016). *Classification of trash for recyclability
status* (CS229 project report). Stanford University.

Ultralytics. (2025). *Ultralytics YOLO26 documentation*. Retrieved from
https://docs.ultralytics.com/models/yolo26

---

### Author notes (remove before submission)

- Section 2.2 refers to survey accuracies in the 85 to 95 per cent range without
  citation. Either add a verified survey reference or delete the sentence.
- Section 4.4's PCA anomaly should be re-run to exclude a preparation
  inconsistency before this section is submitted.
- The per-object evaluation script (`scripts/eval_pipeline_per_object.py`) has
  not been executed. If results become available before the deadline, Section
  4.6 should be updated with real routing figures.
- `extract_preds()` in that script assumes the detection list is located at
  `result["model"]["items"]`. Confirm against `web/server.py` before running.
