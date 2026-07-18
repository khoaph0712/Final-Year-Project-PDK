# Presentation Site Redesign — Design Spec

Date: 2026-07-18. Approved direction: replace the current site with a presentation-style
multi-page site (lecture will present from the webpage instead of slides), figure-heavy,
every figure captioned with a description.

## Goal

The deployed site (wastewise-fyp.vercel.app) becomes the presentation. Nav:
**Home · Data Collection · EDA · Modeling · Demo**. Content follows the lecturer's
report flow, minus Prompt Engineering (not applicable — no LLM stage):

1. Introduction → Home
2. Dataset Overview → Data Collection
3. Image Preprocessing → Data Collection
4. EDA → EDA
5. Model Pipeline → Modeling
6. Model Training (curves + confusion matrices) → Modeling
7. Evaluation Metrics → Modeling
8. Experimental Results → Modeling
9. Model Comparison → Modeling
10. Discussion → Modeling
11. Conclusion & Future Work → Modeling
12. Live Demo → Demo (scanner + history)

## Approach

Extend the existing vanilla hash-routed SPA in `web/` (no framework, no build step).
Existing scanner, API integration, history, and design system are kept. Pages renamed:
`PAGES = ["home", "data", "eda", "modeling", "demo"]`.

## Presentation mechanics

- Every section: numbered kicker badge (e.g. `04 / 11`), `id="s4"` anchor.
- Prev/Next footer per section walking the ordered flow across pages.
- Keyboard ← / → advance/rewind the flow (disabled while typing/scanning).
- Header progress rail: 11 dots, current section highlighted.
- Figures: `<figure class="fig">` with `<img loading="lazy">` + `<figcaption>` holding a
  bold title + 1–3 sentence description tied to real numbers. Click opens a lightbox
  (single reusable overlay, Esc/click to close).

## Figure manifest (copied to web/assets/figures/)

Home: wf_overall, wf_use_case, wf_gui, 2 app screenshots.
Data Collection: cross_dataset_comparison, wadaba_sample_grid, wadaba_class_distribution,
taco/trashnet/gini source class distributions, wf_er_diagram, balanced_class_distribution,
wf_dfd, detector labels.jpg (final dataset label stats).
EDA: tuned_dataset_v1 class balance / box-area hist / small-box ratio,
merged_v3 source composition, wf_leakage_audit, near_dup_spotcheck, domain importance chart.
Modeling — pipeline: wf_ml_pipeline, wf_dl_pipeline, workflow_pipeline.svg (existing),
wf_sequence, wf_deployment_uml.
Modeling — training: yolo26m results.png (deployed curves), yolo26s results.png,
classifier training plots (EfficientNet tuned), resnet50/mobilenetv2 plots, tiny-CNN loss.
Modeling — confusion: yolo26s CM normalized, comparison CM grid, EfficientNetB0 CM,
ExtraTrees CM.
Modeling — metrics: BoxPR + BoxF1 curves (yolo26s run).
Modeling — results: 3 stage2 pipeline output images, 2 Grad-CAM samples.
Modeling — comparison: chart_ml_vs_dl, merged_6class model comparison, PCA dimensionality
chart, PCA model sweep.

~45 figures total, lazy-loaded.

## Content sources

Copy redistributed from current Pipeline/Evidence/Limits/About pages (nothing invented).
New copy for Intro/Dataset/Preprocessing/EDA drawn from runs/ reports and the leakage
audit. Real audited numbers only (93.77% clean-test acc, 74.9 mAP50, 2,406 leaked images,
76.9→88.9 crop fine-tune, field recall 70.9→78.1, etc.).

## Out of scope

No framework, no PDF export, no speaker notes, no per-section URLs beyond hashes,
no new backend work. server.py untouched.
