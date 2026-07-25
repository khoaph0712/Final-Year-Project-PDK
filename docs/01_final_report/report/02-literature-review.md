# 2. Literature Review

## 2.1 General AI and Computer Vision Models

### 2.1.1 Object Detection Architectures

The detection design space divides into families that differ in how candidate
regions are proposed and how predictions are assigned to ground truth. This
section establishes that space so that the controlled comparison of §7.4 and
§8.3.2 has a frame.

**Two-stage region proposal.** Faster R-CNN (Ren et al., 2015) introduced a
learned Region Proposal Network that shares convolutional features with the
detection head, replacing the external proposal generators that preceded it.
Accuracy is strong and the design is well understood, but the two-pass structure
imposes an inference cost that matters for real-time deployment.

**Anchor-based dense prediction.** Single-stage detectors evaluate a dense grid
of predefined anchor boxes. This is fast but produces a severe foreground–
background imbalance during training. RetinaNet (Lin et al., 2017b) addressed
this with focal loss, which down-weights well-classified examples so that the
large population of easy negatives cannot dominate the gradient. Feature Pyramid
Networks (Lin et al., 2017a) supplied the multi-scale representation these
detectors depend on.

**Anchor-free dense prediction.** FCOS (Tian et al., 2019) removed anchor boxes
entirely, predicting distances from each location to the four sides of its
enclosing box and introducing a *centerness* branch to suppress low-quality
predictions produced far from object centres. This eliminates anchor
hyperparameters — scales, aspect ratios, matching thresholds — that otherwise
require dataset-specific tuning.

**The YOLO lineage.** YOLOv8 (Jocher et al., 2023), YOLO11 (Khanam & Hussain,
2024), and YOLO26 (Ultralytics, 2025; Sapkota & Karkee, 2025) form an
incrementally refined family of anchor-free single-stage detectors that
currently define the accuracy–latency frontier for practical deployment. The
present project deploys YOLO26.

**Transformer set prediction.** RT-DETR (Lv et al., 2023) applies the DETR
formulation — bipartite Hungarian matching between predictions and ground truth,
removing the need for non-maximum suppression — with an efficient hybrid encoder
that makes it competitive in real-time settings.

**The gap this project addresses.** Prior applied work in waste detection almost
universally varies model *size* within a single lineage — comparing YOLO nano
against small against medium — and reports the result as an architecture study.
Varying the lineage itself, under a genuinely equal training budget, is rare.
§7.4 specifies such a comparison across all five families above.

### 2.1.2 Handcrafted Features and Classical Computer Vision

Before learned representations, material and texture discrimination relied on
engineered descriptors. Histograms of Oriented Gradients (Dalal & Triggs, 2005)
capture local edge-orientation structure and remain a strong texture descriptor.
Colour histograms and channel statistics capture material appearance directly,
and frequency-domain statistics capture periodic surface structure such as
weave or corrugation.

This project's classical branch combines all three into the 637-dimensional
representation described in §7.2. Its purpose is measurement: it establishes how
much material signal is recoverable without representation learning, which makes
the deep branch's contribution a measured quantity rather than an assumption.

### 2.1.3 Deep Classification Architectures

ConvNeXt (Liu et al., 2022) modernised the pure convolutional network by
adopting design decisions from vision transformers — larger kernels, inverted
bottlenecks, fewer activations and normalisations — while retaining convolutional
inductive biases and efficiency. Vision transformers with hierarchical windowed
attention offer comparable accuracy at higher training cost. §8.2.1 compares
representatives of both alongside an efficiency-oriented convolutional baseline,
and reports that the selection between the leading two was made on training cost
rather than accuracy, because the accuracy difference was not statistically
meaningful.

### 2.1.4 Context-Aware Visual Search

Scene context constrains object identity: the probability of a given object
depends on the scene it appears in (Torralba et al., 2006). This motivates the
scene-prior damping term evaluated among the pipeline configurations in §8.6.2.
Its measured contribution was not decisive, which is reported as such.

## 2.2 Applications in Waste Detection and Classification

### 2.2.1 Waste Image Datasets and Prior Systems

**TACO** (Proença & Simões, 2020) is the reference dataset for in-the-wild litter,
supplying COCO-format instance annotations over 60 fine-grained categories
photographed in uncontrolled conditions. Its realism is its value and its size is
its limitation.

**TrashNet** (Thung & Yang, 2016) established the standard six-class
classification baseline, but its images are staged single objects on clean
backgrounds. It is a classification benchmark, not a detection resource, and
§8.2.4 quantifies how badly models trained on this style of data transfer.

**Kaggle Garbage Classification** extends the same studio-condition premise to
twelve classes and supplies the studio half of this project's classification
data.

**Roboflow Universe community projects** supply the bulk of this project's
detection data. Community-sourced annotation quality is not independently
verifiable, which is recorded as a standing limitation in §6.1 rather than
discovered later.

Several further datasets were evaluated and rejected — MJU-Waste, GINI, TrashCan,
ZeroWaste — for reasons given in §6.1.

### 2.2.2 Evaluation Integrity

This subsection motivates §6.4 and is the literature most directly load-bearing
for this project's contribution.

Barz and Denzler (2020) demonstrated that CIFAR-100's test set contains
substantial near-duplicates of its training set, and that purging them reduces
reported accuracy by 9 to 14 points. The finding generalises beyond CIFAR: any
corpus assembled by aggregating overlapping sources is a candidate for the same
defect, and the defect is invisible to byte-level deduplication because
re-encoding and resizing change the bytes while preserving the content.

Waste datasets are unusually exposed to this failure. The community re-hosts,
re-annotates and re-exports the same underlying photographs — TACO images
circulate through multiple Roboflow projects — so a merged corpus assembled from
several such sources will contain the same photograph under several names. §6.4
confirms this empirically across four datasets and quantifies the consequence.

### 2.2.3 Domain Shift in Deployed Vision Systems

The gap between curated training data and deployment conditions is well
recognised in general, but is rarely quantified for waste systems specifically,
and reported accuracies are seldom accompanied by any cross-domain measurement.
§8.2.4 supplies one in both directions and finds the transfer to be strongly
asymmetric — a result with a direct practical consequence for how data
collection effort should be allocated.

## 2.3 Gap Addressed by This Work

Three gaps follow from the above. First, waste-detection work compares model
sizes rather than architectural families under equal budgets. Second, evaluation
integrity is assumed rather than audited, despite the community's re-hosting
practices making contamination likely. Third, the interface between stages in a
detection-plus-classification cascade is treated as plumbing rather than as a
place where a train/serve distribution mismatch is created by the architecture
itself — which §8.2.3 shows to be the largest available performance lever.
