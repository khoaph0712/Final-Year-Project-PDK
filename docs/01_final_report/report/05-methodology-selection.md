# 5. Software Development Methodology

## 5.1 Waterfall Model

Waterfall proceeds through requirements, design, implementation, verification and
maintenance as sequential phases, each completed before the next begins. Its
strengths are documentation discipline and predictable scheduling, both valuable
for a project with a fixed submission deadline. Its defining weakness is that it
assumes requirements are knowable in advance and that verification will confirm
rather than overturn earlier phases.

For a machine learning project this assumption is unsafe. Verification does not
merely confirm; it frequently invalidates the data on which every prior phase
depended.

## 5.2 Spiral Model

The Spiral model organises development into repeated cycles, each comprising
objective-setting, risk identification and resolution, development, and planning
of the next cycle. Its distinguishing feature is that **risk analysis is an
explicit, scheduled activity** rather than an implicit hope, and that a cycle may
conclude that the previous cycle's output must be discarded.

The cost is overhead: each cycle carries planning and analysis that a linear
process avoids.

## 5.3 Rapid Application Development and Prototyping

RAD prioritises rapid working prototypes over specification, with requirements
refined through user feedback on successive builds. It suits projects where the
interface is the primary uncertainty. Where the uncertainty is instead whether
the underlying measurements are valid, rapid prototyping accelerates the
production of results without improving their trustworthiness.

## 5.4 Agile Methodology

Agile organises work into short iterations with continuous stakeholder feedback,
emphasising working software and responsiveness to change over comprehensive
documentation. Its iteration structure suits experimental work well. Its reliance
on stakeholder feedback as the primary correction signal fits less well here: in
this project the corrective signal came from audits of the project's own data,
not from any external stakeholder, and no stakeholder was positioned to notice
that the accuracy figures were inflated.

## 5.5 Selection and Justification

**The Spiral model is the honest description of the process this project actually
followed.** This section justifies that against the recorded evidence rather than
describing an idealised process.

The claim rests on §9. The project's trajectory was determined by a sequence of
risk discoveries, each of which forced a return to an earlier phase:

- **Cycle 1** produced datasets, classical baselines and initial deep
  classification results, and reported accuracy figures in the mid-nineties.
- **Cycle 2** performed a risk audit that was not prompted by any visible symptom.
  It found cross-split contamination in every dataset (F1), invalidating every
  metric produced in Cycle 1 and requiring the construction of quarantined
  evaluation splits and full re-measurement.
- **Cycle 3** built the two-stage pipeline and then discovered, again through
  audit rather than symptom, a 16-point train/serve skew at the stage interface
  (F4b) that no existing evaluation had been positioned to detect.
- **Cycle 4** addressed field recall through four successive hypotheses. Three
  were rejected on measurement — tile-augmented training (F4c), field-rebalance
  fine-tuning (F11), and a class-agnostic architectural expansion whose apparent
  gain proved to belong to a confidence threshold changed at the same time (F12,
  F13).
- **Cycle 5** removed a component, the waste-state gate, on the grounds that no
  dataset in the project could train it (F17).

This is Spiral's signature: **each cycle's principal output was a risk
identified, and in three cases the resolution was to discard work.** A Waterfall
process would have carried the Cycle 1 figures to submission, because nothing in
the verification phase of a linear process is positioned to question the
integrity of the evaluation data itself. Agile's iteration would have delivered
the same cadence, but its correction signal — stakeholder feedback — was not the
signal that actually corrected this project.

Two elements were borrowed from other methodologies. Documentation discipline
from Waterfall is visible in the pinned dependency set (§3) and in the artefact
model (§4.4), both of which exist so that a claim can be traced to its
production. Prototyping from RAD governed the web application, where the
interface genuinely was the uncertainty and feedback on a working build was the
efficient way to resolve it.

**A limitation of the process as executed.** Risk analysis was scheduled
retrospectively rather than at the start of each cycle. The leakage audit ran
after three phases of work had already been built on the contaminated data, and
the train/serve skew was found after the cascade had been assembled and
evaluated. A stricter Spiral discipline would have placed a data-integrity audit
before the first model was trained. That this project's most valuable findings
came from audits it ran late is an argument for running them early, and it is
recorded here as the principal process lesson.
