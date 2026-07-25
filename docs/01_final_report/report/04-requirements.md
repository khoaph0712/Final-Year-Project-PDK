# 4. Software Product Requirements

## 4.1 Overview of Similar Products

Existing consumer-facing waste identification tools fall into three groups.
Municipal council applications offer text lookup against a local disposal
schedule; they are authoritative on policy but require the user to already know
what the item is, which is the actual difficulty. Commercial smart-bin systems
perform automated sorting at high accuracy but operate on a controlled conveyor
with fixed lighting and camera geometry, conditions that do not transfer to a
handheld photograph. General-purpose image recognition applications identify
objects but not materials, and offer no disposal routing.

The gap is a system that accepts an uncontrolled photograph, identifies the
material of each object in it, and returns a disposal decision. That is what this
project builds, and §8.4 reports honestly on how far it gets.

## 4.2 User Stories and Use Case Diagram

Three actors interact with the system.

| Actor | Description |
|---|---|
| **Public User** | Photographs waste and receives material and bin routing |
| **Examiner / Lecturer** | Reviews evidence, metrics, and reproduction steps |
| **Developer / Maintainer** | Retrains, evaluates, and redeploys models |

*Table 4.1 — System actors.*

**Public User stories.**

- As a Public User, I want to upload or capture a photograph so that I can find
  out what the items in it are made of.
- As a Public User, I want each detected object labelled with its material and a
  confidence score, so that I can judge how much to trust the answer.
- As a Public User, I want a disposal bin recommendation, so that the
  classification leads to an action.
- As a Public User, I want the system to say when it is unsure rather than guess,
  so that I am not confidently misled.

**Examiner / Lecturer stories.**

- As an Examiner, I want every reported metric traceable to the artefact that
  produced it, so that I can verify claims rather than accept them.
- As an Examiner, I want evaluation splits demonstrably free of training
  contamination, so that reported accuracy means what it appears to mean.
- As an Examiner, I want negative results and rejected hypotheses recorded, so
  that I can assess the process and not only the outcome.

**Developer / Maintainer stories.**

- As a Maintainer, I want dataset construction driven by scripts, so that any
  generation can be rebuilt from its sources.
- As a Maintainer, I want a promotion gate on model updates, so that an
  improvement on one distribution cannot silently regress another.
- As a Maintainer, I want operating points re-swept after any backbone change,
  so that a threshold tuned for one model is never carried to another.

![Use case diagram](docs/diagrams/png/wf_use_case.png)

*Figure 4.1 — Use case diagram showing the three actors and their interactions
with the system.*

## 4.3 System Flow and Activity Diagram

The request flow is: image upload → Stage 1 detection producing candidate boxes
above the deployed confidence threshold → per-box cropping → Stage 2 material
classification with Background rejection → area × confidence aggregation to a
dominant material → bin mapping → response with per-object overlays.

![Machine learning pipeline](docs/diagrams/png/wf_ml_pipeline.png)

*Figure 4.2 — End-to-end pipeline from upload to bin routing.*

The deployed pipeline is stages S1 through S5. A sixth stage, the waste-state
gate S6, existed in earlier versions and was removed on 2026-07-18; §9.7 records
why, and all figures in §8 describe the five-stage configuration.

![Deployment architecture](docs/diagrams/png/wf_deployment_uml.png)

*Figure 4.3 — Deployment view: static front end, inference service, model
artefacts.*

## 4.4 Experiment and Dataset Artefact Model

Because the project's central claim is about evaluation integrity, the
relationships between datasets, splits, model runs and evaluation artefacts are
themselves part of the design rather than incidental bookkeeping. Each dataset
generation carries a source manifest recording the provenance of every image;
each training run writes resolved arguments alongside its weights; each
evaluation writes a JSON artefact naming the weights and the split it used.

This is what allows §8 to distinguish
`detector_clean_val_yolo26m_final100.json` from the four similarly named files
beside it, and it is the mechanism by which a claim in this report can be
checked against the run that produced it.

![Entity relationship diagram](docs/diagrams/png/wf_er_diagram.png)

*Figure 4.4 — Entity relationship model for datasets, splits, runs, and
evaluation artefacts.*

## 4.5 Web Application Sitemap

The application is organised as a single-page presentation with six sections:
Home (overview and problem statement), Data (sources, generations, and the
leakage audit), EDA (class balance and object-size distributions), Modeling
(architecture and training), Results (metrics and comparisons), and Demo (live
upload and classification).

![Graphical user interface layout](docs/diagrams/png/wf_gui.png)

*Figure 4.5 — Application layout and navigation.*

## 4.6 Non-Functional Requirements

| Requirement | Target | Measured |
|---|---|---|
| Single-image inference latency | Interactive (< 1 s) | 138 ms mean, 231 ms p95 (CPU) |
| Availability | Publicly reachable | Deployed at `wastewise-fyp.vercel.app` |
| Reproducibility | Every metric traceable | Script-driven runs with JSON artefacts |
| Honesty of reporting | Baselines beside metrics | Enforced in §8.4.1 |

<!-- src: runs/audits/cpu_latency_yolo26n_vs_yolo26s.json#deployed_yolo26n_hardneg -->

*Table 4.2 — Non-functional requirements and their measured outcomes.*
