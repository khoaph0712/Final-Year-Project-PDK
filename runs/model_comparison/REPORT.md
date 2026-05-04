# ML vs DL Comparison Report

## Scope

- ML-first analysis is kept as primary direction.
- A lightweight DL model (tiny CNN) is added as a comparison baseline.

## Why these models

- **LogReg:** linear baseline on handcrafted features.
- **SVM-RBF:** non-linear boundary baseline.
- **RandomForest:** tree baseline with feature importance.
- **Tiny CNN:** compact end-to-end DL comparator.

## Results


| Model    | Family | Accuracy | F1-macro |
| -------- | ------ | -------- | -------- |
| rf       | ML     | 0.4747   | 0.4548   |
| tiny_cnn | DL     | 0.4813   | 0.4303   |
| svm_rbf  | ML     | 0.4493   | 0.4215   |
| logreg   | ML     | 0.3633   | 0.3231   |


## Conclusion

- Best by F1-macro in this run: **rf** (ML) with Accuracy `0.4747` and F1-macro `0.4548`.
- Keep ML-first pipeline for explainability and lecture requirements; DL is used for benchmark comparison.

## Artifacts

- `chart_ml_vs_dl.png`
- `comparison_metrics.json`
- `runs/feature_ml_analysis/`*
- `runs/dl_baseline/*`