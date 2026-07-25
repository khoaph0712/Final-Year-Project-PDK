# FYP Final Report — Full Rewrite Design

**Date:** 2026-07-25
**Status:** Approved (pending spec review)

## Problem

The final report exists as ten `.docx` variants in `docs/01_final_report/`, none
authoritative. Numbers in the prose cannot be traced to the runs that produced
them, `runs/` holds ~30 result directories with near-identical names (five
`detector_clean_val*` JSONs, 85 `val-N/` directories), and the working skeleton
records five open blockers — two of which are outright contradictions between
the prose and its own figures.

The structure is not the problem: `FINAL_REPORT_SKELETON.md` already mirrors the
reference report's heading structure and is sound. The problem is that the
document has no single source of truth and no traceability from claim to
evidence.

## Goals

1. One canonical, version-controlled report source. The `.docx` becomes a build
   output, never an input.
2. Every factual number traceable to a repo path by grep, with no separate
   manifest to keep in sync.
3. Blocked numbers visibly blocked rather than silently fabricated.
4. Full rewrite of all prose from the skeleton, in one voice.

## Non-goals

- Reorganising `runs/`. The mess there is real but tangential; the evidence
  comments pin exact paths, which is enough.
- Preserving the existing report's word count, figure count, or section
  emphasis. Free hand, per the constraints decision.
- Fetching or re-running the vast.ai detector sweep. Out of scope; handled by
  placeholder hooks.

## Approach

Chapter-per-file Markdown, concatenated and converted by pandoc using a
`reference.docx` style template lifted from the current report.

Rejected: a single 18k-word `REPORT.md` (unreviewable, painful to edit
surgically) and a hand-rolled python-docx builder (the existing
`scripts/build_milestones_report_docx.py` spends 750 lines formatting a smaller
document; reproducing 33 tables, 54 figures and 16 equations that way is
hundreds of lines of layout code pandoc already does correctly).

### Layout

```
docs/01_final_report/
  report/
    00-frontmatter.md          08-results.md
    01-introduction.md         09-failures.md
    02-literature-review.md    10-product-evaluation.md
    03-technology.md           11-conclusion.md
    04-requirements.md         12-references.md
    05-methodology-selection.md
    06-datasets.md             appendix-a-comparison.md
    07-methodology.md          appendix-b-repo.md
    reference.docx
    build.py
  archive/                     # ten superseded .docx variants
```

Figures are referenced in place from `web/assets/figures/` (54 files, matching
the report's figure count), `docs/diagrams/png/` and
`docs/01_final_report/figures/`. They are not copied into `report/`.

### Build

`build.py` concatenates `report/*.md` in filename order into
`_build/REPORT.md`, then invokes pandoc:

```
pandoc _build/REPORT.md -o WasteWise_FYP_Final_Report.docx \
  --reference-doc=report/reference.docx \
  --resource-path=.:../..:../../web/assets/figures \
  --toc --toc-depth=3
```

`--resource-path` is rooted at the repo so figure paths in the Markdown are
written repo-relative and resolve without copying assets.

Dependency: pandoc (`winget install --id JohnMacFarlane.Pandoc`). Not currently
installed; verified absent at design time.

`reference.docx` carries only styles — pandoc reads `styles.xml` and ignores
reference-doc body content. Seeded from the current report so the established
Times New Roman 12pt / justified / 1.5-spacing formatting carries over without
being re-specified.

### Evidence convention

Every factual number carries an inline HTML comment naming its source:

```markdown
Bin-routing accuracy reaches 97.41%
<!-- src: runs/audits/pipeline_bin_decision_eval_no_gate.json#binAccuracy -->
```

Pandoc strips HTML comments during docx conversion, so they never reach the
submitted document. `grep -c "<!-- src:" report/*.md` gives coverage; a number
without one is visibly unsourced under review.

Chosen over a separate manifest plus validator script: no second file to drift
out of sync, no tooling to maintain, and the source sits adjacent to the claim
where it is actually checkable.

Blocked numbers use a distinct marker so one grep finds every hole:

```markdown
| RT-DETR-l | `TBD` | `TBD` |
<!-- BLOCKED: runs/detect/vast_comparison not fetched -->
```

### Writing order

Dependency order, not document order:

1. **§6 datasets → §7 methodology → §8 results → §9 failures.** Evidence-bound;
   they constrain everything else.
2. **§1–§5.** These forward-reference results, so they are written once the
   results are fixed.
3. **§10 evaluation → §11 conclusion.**
4. **Front matter, references, appendices.** The abstract quotes headline
   numbers and is written last.

## Resolved decisions

| Decision | Resolution |
|---|---|
| Source of truth | Markdown master; `.docx` is generated output |
| Rewrite depth | Full rewrite of all prose from the skeleton |
| Detector sweep | Placeholder hooks, nothing fabricated |
| Constraints | Real dataset names only (Kaggle Garbage Classification, TACO, Roboflow projects) — no STUDIO/FIELD internal labels. No word-count, figure-count or emphasis constraint. |
| §8.4 configuration | Quote `no_gate` throughout: 94.53% material, 97.41% bin-routing |
| Actor vocabulary | Figure 1's names — Public User / Examiner-Lecturer / Developer-Maintainer |

## Open blockers and their handling

**1. Seven-architecture detector sweep has no results.** `runs/detect/vast_comparison/`
does not exist. The commit `c81ad7e` "retrain comparison models" retrained the
*classifier* comparison (MobileNetV2/ResNet50), not the detector sweep. §8.3.2,
§8.6.3 and Appendix A get `TBD` cells with the exact table schema and `BLOCKED`
markers.

**2. §7.4 comparison config — two corrections to the skeleton's own account.**
Verified against `scripts/vast/run_comparison.py`:

- Shared config is `--epochs 30 --imgsz 512 --batch 16`. The deployed detector
  runs at 640px. The report must not conflate the two.
- `BATCH_OVERRIDES = {"rtdetr_l": 16}` **is inert at the documented command**,
  because the shared default is already `batch 16`. The skeleton's claim that it
  "halves its batch" holds only if `--batch 32` is passed. At the documented
  config all seven models train at batch 16, which makes the identical-config
  claim stronger than the skeleton states.
- **There is no explicit shared optimizer.** `OPTIM_OVERRIDES` sets RT-DETR-l to
  AdamW at `lr0 1e-4`, but the shared `SGD at lr0 0.01` appears only in a code
  comment explaining what diverged RT-DETR; the script sets no optimizer, so the
  sweep inherits Ultralytics `optimizer='auto'`, which itself selects between
  AdamW and SGD by dataset size. The shared optimizer is therefore **not
  determinable from the script** and must be read from each run's resolved
  `args.yaml` when results land. Until then it is a `BLOCKED` item, not a claim.

**3. 84.7% plastic-vote claim — resolved.** Sourced to
`runs/audits/yolo26m_conf_gate_alpha_sweep.json`, key
`field_class_bias_conf004.class_pred_count` (653 of 771 field boxes), alongside
per-class field precision 0.453 for plastic.

**4. Figure 49 caption contradiction — resolved by decision.** Quoting `no_gate`
throughout requires regenerating the chart so the "deployed" marker sits on the
`no waste-state gate` bar, and correcting the caption's "right-most" (the
right-most bar is the prior-damping variant). This aligns the figure with
§8.6.4 and Failure F17, which already describe the ungated configuration as
deployed.

**5. Actor naming — resolved by decision.** Prose adopts Figure 1's vocabulary.
Text-only change; no diagram regeneration.

## Verification

- `grep -c "<!-- src:"` across `report/*.md` reports evidence coverage.
- `grep -rn "BLOCKED"` enumerates every unfilled hole before submission.
- `grep -rniE "STUDIO|FIELD domain"` must return no hits in prose (dataset
  naming constraint).
- `build.py` runs clean and produces a `.docx` that opens with the expected
  heading hierarchy and a generated table of contents.

## Risks

- **Fabrication risk.** The dominant risk in rewriting a results-heavy report.
  Mitigated by the `src:` convention: any number written without reading a file
  is visibly missing its comment.
- **Pandoc fidelity.** Equations (16 numbered) and complex tables may need
  hand-checking after conversion. Verified on §6.4's equations first, before the
  remaining chapters commit to the same pattern.
- **Figure numbering drift.** With 54 figures and a full rewrite, numbering will
  not match the old document. Figure references are written as Markdown links to
  paths, and numbering is assigned in a single pass once all chapters exist.
