---
title: "WasteWise: A Two-Stage Detection and Verification Pipeline for Waste Material Classification"
author: "Phung Danh Khoa"
---

# UNDERGRADUATE FINAL YEAR PROJECT REPORT

**WasteWise: A Two-Stage Detection and Verification Pipeline for Waste Material
Classification**

Phung Danh Khoa

Bachelor of Science with Honours in Computing

REG. NO: `[REG. NO. — fill before submission]`

Submission date: `[DATE — fill before submission]`

---

## Abstract

Automated visual waste sorting is widely reported at accuracies above ninety per
cent, but those figures are measured on curated benchmarks of single, centred,
well-lit objects and do not survive contact with deployment conditions. This
report presents WasteWise, a two-stage waste detection and material
classification pipeline, and reports its construction and evaluation with the
gap between benchmark and field performance as the object of study rather than
as a closing caveat.

The system localises objects with a YOLO26 detector and verifies each proposed
region with a second-stage classifier that combines a frozen pretrained backbone
with 637 handcrafted features and an explicit Background class for rejecting
first-stage false positives. A parallel classical branch over the same
handcrafted representation is reported for comparison, reaching 73.76% accuracy
and establishing the contribution of learned representation as a measured
quantity.

Before any figure was reported, every dataset split was audited for cross-split
contamination. All four datasets leaked: between 19.8% and 43.0% of each test
split duplicated or near-duplicated training data, with 43.0% of the
classification test set and 1,723 byte-identical file pairs in the worst case.
The audit combines MD5 equality with a 64-bit perceptual hash at Hamming
distance 4, made tractable across 406 million cross-split pairs by a
pigeonhole-partitioned index. Correcting it reduced classifier accuracy by 2.53
points and detector test mAP50 by 4.7 points; every figure in this report is
measured on the decontaminated splits.

The principal finding concerns the interface between stages. A classifier trained
on ground-truth crops scored 92.93% on that distribution but only 76.91% on the
detector's own output — the input it actually receives. Fine-tuning on a mixture
of detector crops and ground-truth crops raised production accuracy to 88.88%,
an improvement of 11.97 percentage points, while clean accuracy also improved to
93.77%. This effect is several times larger than any produced by changing either
stage's architecture or input resolution, and it generalises as practice: any
detector-and-classifier cascade should train its classifier on detector output.

Domain transfer was measured in both directions and is strongly asymmetric. A
classifier trained on studio imagery loses 40.19 points on real-world imagery,
while the reverse direction loses 28.27, indicating that real-world data carries
greater value per image than curated data for this task.

Two negative results are reported that are more useful than the figures they
replace. The deployed pipeline reaches 94.53% overall material accuracy on 1,042
independent images but only 84.60% macro-averaged recall, and glass is absent
from the evaluation set entirely. Bin-routing accuracy, the metric most natural
for such a system, is shown to be degenerate: because the taxonomy maps five of
six materials to a single bin, a constant predictor scores 99.62% against the
deployed system's 97.41%, so the metric measures how often the system declines to
commit rather than whether it routes correctly. The honest characterisation of
this system is 84.60% macro-averaged material recall across the three classes
adequately represented, with routing performance unmeasured.

**Keywords:** waste classification, object detection, two-stage pipeline,
evaluation integrity, dataset leakage, near-duplicate detection, domain shift,
train/serve skew, degenerate metrics

---

## Acknowledgement of Assistance and Encouragement

I would like to thank my supervisor for guidance throughout this project, and in
particular for the insistence on evidence over assertion that led directly to the
data audit reported in Section 6.4 — the finding on which most of this report's
contribution rests.

I am grateful to the maintainers of the public datasets this work depends on: the
TACO project for in-the-wild litter annotations, and the Roboflow Universe and
Kaggle communities for the classification and detection resources that made a
project of this scope feasible for an individual student. The limitations of
community-sourced annotation are discussed in Section 6.1; the availability of
that data at all is what made the work possible.

Finally, I thank my family and friends for their patience and encouragement
across a project whose most productive weeks produced results that had to be
thrown away.

---

## Declaration of AI Use

Generative AI assistance was used during this project in the following capacities:
code drafting and review for data-processing, training and evaluation scripts;
drafting and editing of report prose; and assistance in structuring the
evaluation and audit methodology.

All experimental results, metrics, and figures reported in this document were
produced by executing the project's own code against the project's own data. No
reported number was generated by, or estimated by, an AI system. Every factual
figure in this report is traceable to a named artefact in the repository, and the
verification procedure is documented in Appendix B.2.

Section 6.1 records one instance where a script's simulation fallback produced
plausible-looking dataset statistics for datasets not present in the project.
Those figures are explicitly excluded from this report, and the exclusion is
noted at the point where such data would otherwise have appeared.

---

## Table of Contents

*Generated on conversion. See Appendix B.2 for the build procedure.*

## List of Figures

*Generated on conversion.*

## List of Tables

*Generated on conversion.*
