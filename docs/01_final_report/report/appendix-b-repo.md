# Appendix B. Repository Structure and Reproduction

## B.1 Layout

```
FYP/
├── data/                       Active project data
│   ├── merged_dataset_v5/          classification, 24,039 train / 5,600 test
│   ├── merged_dataset_v5_clean_test/   quarantined clean test, n = 3,194
│   ├── hard_case_classifier_v1/    classification hard cases
│   ├── hard_case_classifier_v1_clean/  quarantined clean splits
│   └── detector_crops_v1/          detector-output crops for stage 2 fine-tuning
├── external_datasets/          Symlinked large datasets
│   ├── yolo26_hardcase_dataset_v1/ detection, 26,100 images + source manifest
│   ├── super_yolo_dataset/         superseded detection dataset
│   ├── yolo26_hardcase_clean_eval/ quarantined clean detection splits
│   └── taco_official/, taco_yolo/, rf_*/   raw sources
├── scripts/                    All dataset, training and evaluation code
│   └── vast/                       rented-GPU comparison sweep
├── runs/                       Every training and evaluation artefact
│   ├── audits/                     leakage, clean evals, conf sweeps, pipeline evals
│   ├── detect/                     detector training runs
│   ├── dl/                         deep classification runs
│   ├── ml/                         classical model sweeps
│   └── comparisons/                cross-model comparison tables
├── web/                        Deployed application and inference service
└── docs/01_final_report/       This report
    └── report/                     markdown chapters + build.py
```

## B.2 Reproducing the Report

The report is written as one Markdown chapter per file. `build.py` concatenates
them in filename order and converts the result with pandoc, using the previous
report as a style template.

```bash
python docs/01_final_report/report/build.py
```

Inline `<!-- src: ... -->` comments record the origin of every factual number.
Pandoc strips HTML comments during conversion, so they do not appear in the
generated document. To audit evidence coverage:

```bash
grep -rc "<!-- src:" docs/01_final_report/report/*.md
```

To find every figure still blocked on an unfetched result:

```bash
grep -rn "BLOCKED" docs/01_final_report/report/
```

## B.3 Reproducing Key Results

**Leakage audit (§6.4).** Recomputes MD5 and perceptual hashes across all splits
and writes both the JSON artefact and the Markdown summary:

```bash
python scripts/audit_model_risks.py
```

**Clean evaluation splits (§9.2).** Quarantines every leaked evaluation image,
leaving originals untouched:

```bash
python scripts/build_clean_eval_splits.py
```

**Stage 2 detector-crop fine-tuning (§8.2.3).** Builds the detector-crop dataset,
then fine-tunes on a detector-crop and ground-truth mixture behind the
no-forgetting promotion gate:

```bash
python scripts/build_detector_crop_dataset.py
```

```bash
python scripts/finetune_stage2_on_detector_crops.py
```

**Classical branch PCA sweep (§8.1).** Sweeps classical models across nine PCA
dimensionalities and the untransformed 637-D space:

```bash
python scripts/pca_feature_model_sweep.py
```

**End-to-end pipeline evaluation (§8.4).**

```bash
python scripts/eval_pipeline_bin_decisions.py
```

**Corrected per-object protocol (§8.4.2, not yet run).** Confirm that
`extract_preds()` matches the response shape in `web/server.py` before running:

```bash
python scripts/eval_pipeline_per_object.py
```

## B.4 Completing the Architecture Comparison

The sweep requires a rented 24 GB GPU; `scripts/vast/README.md` documents the
full procedure, which must be run under the account holder's own credentials.

```bash
bash scripts/vast/fetch_results.sh
```

Once `runs/detect/vast_comparison/` is populated, fill Table 8.7 and Table A.2,
reading each run's resolved optimizer and learning rate from its `args.yaml`
rather than assuming them from the sweep script (§7.4). Report mAP50 and
mAP50-95 only for cross-architecture claims.

## B.5 Environment

Training requires the project virtual environment, not the shell default, which
is a CPU-only build. Dependencies are pinned in `requirements.txt`; the torch
2.4.1 pin is load-bearing for the ONNX and TFLite export chain (§3.1).

```bash
pip install -r requirements.txt
```

Known machine constraints, recorded as Failure F7: batch 16 is the stable
configuration on a 12 GB card; dataset caching must be disabled; and training
must run from a script file with a `__main__` guard rather than through
`python -c`, which deadlocks under Windows spawn-based multiprocessing.
