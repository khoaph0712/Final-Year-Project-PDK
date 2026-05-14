# ML vs DL Comparison Report

## Scope
- ML-first analysis is kept as primary direction.
- A lightweight DL model (tiny CNN) is added as a comparison baseline.

## Why these models
- **extra_trees:** tree ensemble baseline for noisy, high-dimensional handcrafted features.
- **rf:** tree baseline with feature importance for feature-group analysis.
- **linear_svm:** margin-based baseline for the larger color+HOG feature vector.
- **logreg:** linear baseline on standardized handcrafted features.
- **tiny_cnn:** compact end-to-end DL comparator on object crops.

## Results
| Model | Family | Accuracy | F1-macro |
|---|---|---:|---:|
| extra_trees | ML | 0.6312 | 0.6113 |
| rf | ML | 0.6317 | 0.6111 |
| linear_svm | ML | 0.5960 | 0.5642 |
| logreg | ML | 0.5864 | 0.5558 |
| tiny_cnn | DL | 0.4813 | 0.4303 |

## Conclusion
- Best by F1-macro in this run: **extra_trees** (ML) with Accuracy `0.6312` and F1-macro `0.6113`.
- Keep ML-first pipeline for explainability and lecture requirements; DL is used for benchmark comparison.

## Artifacts
- `chart_ml_vs_dl.png`
- `comparison_metrics.json`
- `runs\ml\feature_ml_enhanced_6class_4k/*`
- `runs/dl/dl_baseline/*`