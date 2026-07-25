# 10. Product Evaluation

## 10.1 Strengths

**The measurements can be trusted, and that is not trivially true.** Every
evaluation split was audited for cross-split contamination before any figure in
this report was quoted, and the contamination found — between 19.8% and 43.0%
across four datasets — was substantial enough that the pre-audit figures were
materially wrong. The clean splits, the quarantine manifests and the audit script
are all in the repository, so the correction is reproducible rather than
asserted.

**The central finding is robust.** The detector-crop fine-tuning result of §8.2.3
has the properties that make a result believable: a large effect (11.97 pp), a
like-for-like comparison on a fixed evaluation set, an identified causal
mechanism, no selection across multiple configurations, and a promotion gate that
would have rejected it had it traded one distribution against another. It also
generalises as advice — any detector-plus-classifier cascade should train its
classifier on detector output.

**The domain-shift measurement is direct and actionable.** The 40.19-point
studio-to-field gap, and its asymmetry with the 28.27-point reverse, is a
measured quantity with a concrete implication for where data-collection effort
should go.

**The system works and is deployed.** It accepts arbitrary photographs, returns
per-object materials with confidences and a routing decision, and does so in
138 ms mean CPU latency.

**Negative results were kept.** Four hypotheses were tested and rejected with
clean measurement (F4c, F11, F12, F16), one component was removed for being
untrainable (F17), and one metric was shown to be degenerate (§8.4.1). A report
that presented only what worked would be shorter and less useful.

## 10.2 Limitations and Challenges

These are stated in order of severity, and the first is more serious than
anything in §10.1 is strong.

**Bin-routing performance is effectively unmeasured.** §8.4.1 establishes that on
the available evaluation data, a constant predictor that always answers
"Recycling" scores 99.62% against the deployed system's 97.41%. The routing
taxonomy assigns five of six materials to one bin, and the evaluation set is
99.62% that bin. **No defensible claim about routing accuracy can be made from
this data.** This is the largest gap in the project.

**Rare-class performance is unknown.** Glass does not appear at all in the
end-to-end evaluation set; organic and cardboard appear four and nine times
respectively. Recalls of 50.0% and 88.9% on those supports carry no information.
Claims about six-class performance therefore rest on evidence for three classes,
and the honest headline is 84.60% macro-averaged recall across the classes that
are actually represented.

**The evaluation protocol discarded minority objects.** The end-to-end script
reduced each image to its single most frequent ground-truth class before scoring,
making an image containing six paper items and one glass item count only as
paper. This is why glass is absent. A corrected per-object protocol is written
but has not been run (§8.4.2).

**The architecture comparison is incomplete.** The seven-architecture controlled
sweep of §7.4 was specified, provisioned and launched, but its results were not
retrieved before compilation. Objective O5 is therefore not met, and §8.3.2 is a
schema rather than a result.

**The PCA anomaly is unexplained.** §8.1's non-monotonic accuracy curve has a
plausible account — PCA rotation destroying axis-aligned structure that tree
ensembles exploit — but it has not been verified, and a preparation
inconsistency has not been excluded. It is reported as provisional.

**The leakage audit has a known blind spot.** Perceptual hashing detects
near-duplicate *images*. It does not detect the same physical object photographed
twice from different angles, which is a plausible contamination mode in
community-sourced waste data. Residual contamination of this kind cannot be
excluded.

**Detector generalisation beyond TACO capture batches is untested.** The clean
test split consists of unseen TACO capture batches. Performance on a genuinely
different geography or waste stream remains unknown, and the substantial
validation-to-test gap (0.749 → 0.482 mAP50) suggests it would be worse again.

**Annotation quality upstream is unverifiable.** The Roboflow community projects
supplying most detection data cannot be independently checked for annotation
correctness.

## 10.3 Overall Assessment

The system does what it was built to do, and the report's contribution is less
the system than the measurement discipline applied to it. The most useful
sentence this project can offer is not its best accuracy figure but this: the
figures it started with were wrong by between two and five points because of
contamination nobody had looked for, its largest single improvement came from a
distribution mismatch created by its own architecture, and its most natural
headline metric is degenerate on its own evaluation data. Each of those was found
by measuring something that was assumed rather than tested.
