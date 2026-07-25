# 1. Introduction

## 1.1 Problem Statement and Motivation

Recycling systems fail at the point of sorting. Contamination — a single greasy
cardboard box or an unrinsed container in a recycling stream — can downgrade or
condemn an entire batch, and the decision that causes it is made in a few
seconds by a person standing at a bin with no feedback and no expertise. Automated
visual waste sorting is therefore an obvious application for computer vision, and
the published accuracy figures suggest the problem is close to solved.

Those figures mislead. Most reported waste-classification accuracy is measured on
curated benchmarks: single objects, centred, well lit, photographed against clean
backgrounds. Deployment conditions are the opposite — objects are small,
occluded, cluttered, partially out of frame, and photographed under whatever
light happens to be available.

**The gap between curated-benchmark accuracy and field performance is the thesis
of this report.** It is not treated as a caveat at the end of the results but as
the object of study. This project measures that gap directly, finds it to be
approximately 40 percentage points in the direction that matters (§8.2.4),
identifies a specific and correctable cause of it inside the system's own
architecture (§8.2.3), and reports what remains unmeasured after the correction.

A second, related failure mode is discovered along the way. Before any of these
measurements could be trusted, the datasets themselves had to be audited, and
that audit found that between 20% and 43% of every evaluation split in the
project was contaminated with training data (§6.4). The accuracy figures the
project had been reporting to itself were inflated, and correcting them changed
the conclusions.

## 1.2 Project Objectives

The project is assessed in §11.2 against the following objectives, each stated so
that it can be verified rather than asserted:

1. **O1.** Build a deployed system that accepts an arbitrary photograph and
   returns per-object material classifications and a disposal routing decision.
2. **O2.** Establish evaluation integrity by auditing every dataset split for
   cross-split contamination before reporting any metric, and re-measure all
   affected figures on decontaminated splits.
3. **O3.** Quantify the domain gap between curated studio imagery and real-world
   field imagery, in both transfer directions.
4. **O4.** Identify and correct the largest single performance lever available in
   a two-stage detection-and-classification cascade.
5. **O5.** Justify the detector architecture empirically, under a controlled
   equal-budget comparison rather than by citation or convention.
6. **O6.** Report negative results and degenerate metrics honestly, including
   where the system fails to beat a trivial baseline.

O5 is the objective that is not fully met; §11.2 states why.

## 1.3 Project Scope and Class Taxonomy

The system classifies six material classes — **plastic, glass, metal, paper,
cardboard, organic** — extended by a seventh **Background** class used internally
by the classifier to reject detector false positives.
<!-- src: external_datasets/yolo26_hardcase_dataset_v1/data.yaml -->

These six were selected because they map onto the disposal decisions a household
or public bin actually offers, and because they are the classes for which
adequate annotated data exists across the available public sources.

Explicitly out of scope: hazardous waste (batteries, chemicals, medical),
electronic waste, textiles, and any judgement about an item's cleanliness or
disposal *state*. That last exclusion is not incidental — §9.7 records the
removal of a component that attempted state classification without any dataset
that labelled it.

Routing maps the six materials onto three bins: plastic, glass, metal, paper and
cardboard to Recycling; organic to Compost; and anything unknown or
Background-dominant to Review. §8.4.1 shows that this taxonomy makes the routing
metric degenerate on the available evaluation data, which is reported as a
finding rather than hidden.

## 1.4 Project Plan

Work proceeded in five phases across June and July 2026: dataset acquisition and
consolidation; classical feature-based baselines; deep classification and
backbone selection; detection, localisation and the two-stage cascade; and
deployment with end-to-end evaluation.

The plan as executed did not follow this order cleanly. The leakage audit (§6.4)
landed in the middle of the third phase and forced re-measurement of everything
completed to that point. §5.5 discusses why this makes the project's actual
process closer to a Spiral model than to the linear plan originally drawn.

## 1.5 Project Outcomes

- A deployed web application serving the two-stage pipeline at
  `wastewise-fyp.vercel.app`, with a measured single-image CPU latency of
  138 ms mean.
- Two model branches: an explainable classical branch reaching 73.76% accuracy on
  637 handcrafted features, and a deep two-stage production branch reaching
  92.93% on clean crops.
- A reusable cross-split leakage audit, applied to four datasets, with a
  tractable near-duplicate index and a quantified correction to every affected
  metric.
- A measured domain-transfer characterisation showing a 40.19-point accuracy
  loss from studio to field imagery.
- The project's central result: correcting the training distribution at the
  interface between pipeline stages improved production accuracy by 11.97
  percentage points, several times more than any architectural change attempted.
- A seventeen-entry engineering failure register (§9), including four hypotheses
  that were tested and rejected with clean measurement.

## 1.6 Project Evaluation

Success is judged in §10 against three criteria: whether the deployed system
performs its function, whether the measurements supporting that claim are
trustworthy, and whether the limitations are stated with enough precision to be
acted on. The second criterion is weighted most heavily, because §6.4 and §8.4.1
together establish that this project's initial measurements were not trustworthy,
and the work of making them so is the bulk of its contribution.
