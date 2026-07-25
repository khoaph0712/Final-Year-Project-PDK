# 11. Conclusion and Future Work

## 11.1 Knowledge, Experience, and Insights

Four lessons from this project generalise beyond it.

**Audit the data before trusting any metric.** The leakage in this project was
invisible — no symptom, no anomaly, metrics that looked good. It was found only
because an audit was run against data nobody suspected. Between 19.8% and 43.0%
of every evaluation split was contaminated, and byte-level checks could not have
found it, because the duplicate copies had been re-encoded in transit through
community re-hosting. Content-level perceptual deduplication is mandatory when
merging overlapping public datasets.

**Evaluate each stage on what the previous stage actually produces.** The
classifier scored 92.93% on ground-truth crops and 76.91% on the detector's own
crops. Nothing in the project was positioned to notice a 16-point gap, because
every evaluation used the clean distribution. Correcting this produced a larger
improvement than every architectural change attempted combined.

**Match operating points before crediting an architecture.** An apparent field
gain from a class-agnostic architectural change turned out to belong almost
entirely to a confidence threshold that had been changed at the same time.
Isolating the threshold reproduced the gain with no architecture change at all.
Two variables moved together and the credit went initially to the more
interesting one.

**Report the baseline beside the metric.** A 97.41% bin-routing accuracy sounds
like a result until it is placed beside the 99.62% achieved by always answering
"Recycling". The metric was not measuring routing competence; it was measuring
how often the system declined to commit.

The uncomfortable common thread is that each of these was found by measuring
something the project had assumed. None was found by building more.

## 11.2 Achievement Assessment

Assessed against the objectives of §1.2:

| # | Objective | Status |
|---|---|---|
| O1 | Deployed per-object classification and routing system | **Met** — live, 138 ms mean CPU latency |
| O2 | Audit all splits for contamination and re-measure | **Met** — four datasets audited, clean splits built, all figures re-measured |
| O3 | Quantify the studio-to-field domain gap in both directions | **Met** — 40.19 pp and 28.27 pp |
| O4 | Identify and correct the largest cascade performance lever | **Met** — stage-interface distribution mismatch, +11.97 pp |
| O5 | Justify the detector architecture under equal-budget comparison | **Not met** — sweep launched but results not retrieved (§8.3.2) |
| O6 | Report negative results and degenerate metrics honestly | **Met** — §8.4.1, §9, and four rejected hypotheses |

*Table 11.1 — Objective assessment.*

Five of six objectives are met. O5 failed on execution rather than design: the
protocol is specified in §7.4 with its deviations disclosed, the infrastructure
is provisioned, and §8.3.2 carries the exact table schema — the sweep simply did
not return in time. That gap is stated rather than papered over with numbers from
a non-comparable earlier run.

The project's honest headline is this: **84.60% macro-averaged material recall
across the three classes represented in the evaluation data, with routing
performance unmeasured and rare-class performance unknown.**

## 11.3 Future Work

**Build an evaluation set with genuine rare-class representation.** This is the
highest-priority item and it blocks the others. Glass is entirely absent from the
current end-to-end set, and organic and cardboard appear four and nine times.
Until this exists, six-class claims cannot be made.

**Run the corrected per-object evaluation protocol.**
`scripts/eval_pipeline_per_object.py` is written and matches predictions to
ground truth by IoU rather than collapsing each image to its dominant class. It
has not been executed. Note that its `extract_preds()` assumes the detection list
sits at `result["model"]["items"]`, which should be confirmed against
`web/server.py` before running.

**Redesign the routing taxonomy or its evaluation.** A taxonomy mapping five of
six materials to one bin cannot produce a meaningful routing metric. Either the
taxonomy needs finer bins reflecting real municipal streams, or routing must be
evaluated per-object on a class-balanced set.

**Complete the architecture comparison.** Fetch the seven-model sweep, read each
run's resolved optimizer from its `args.yaml` rather than assuming it, and report
mAP50 and mAP50-95 only — precision and recall are on different operating points
between the Ultralytics and torchvision groups.

**Resolve the PCA anomaly.** Re-run the full-dimensional and PCA-transformed
conditions under verified-identical preparation to exclude a normalisation
inconsistency before the axis-alignment explanation is accepted.

**Extend the leakage audit to object identity.** Perceptual hashing cannot detect
the same physical object photographed from a different angle. An embedding-based
near-duplicate check would close this blind spot.

**Test generalisation beyond TACO.** The clean test split is unseen TACO capture
batches. Given the size of the validation-to-test gap already observed, a
genuinely different geography or waste stream should be expected to be harder,
and that should be measured rather than assumed.
