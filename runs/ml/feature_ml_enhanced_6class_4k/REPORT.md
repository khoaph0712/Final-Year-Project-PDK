# Feature + ML Analysis Report

## Dataset and classes (examiner checklist)
- **Dataset:** YOLO-format merged dataset at `C:\FYP_v2\merged_dataset_v3` (config: `C:\FYP_v2\merged_dataset_v3\data.yaml`).
- **Classes defined in `data.yaml`:** **7** — plastic, glass, metal, paper, cardboard, organic, other.
- **Classes used in this ML run:** **6** — plastic, glass, metal, paper, cardboard, organic (excluded from training/eval here: **other**).
- **Class balance:** The **raw** dataset splits can be **imbalanced** (different sources merged into `merged_dataset_v3`). When per-class caps are enabled, this script samples **up to N object crops per class** on train/test so counts are **intentionally balanced at the crop level** where enough boxes exist; see `class_support.json` for exact train/test counts. Rare classes may still fall **below** the cap.

## Handcrafted features used for ML (not raw pixels)
Each object crop is resized to 64x64 internally, then summarized into **637** floats.
### Spatial domain (8 features)
1. `mean_intensity`
2. `std_intensity`
3. `p10_intensity`
4. `p50_intensity`
5. `p90_intensity`
6. `grad_mean`
7. `grad_std`
8. `edge_density`
### Frequency domain (9 features; FFT radial bins + `high_freq_energy`)
9. `fft_bin_1`
10. `fft_bin_2`
11. `fft_bin_3`
12. `fft_bin_4`
13. `fft_bin_5`
14. `fft_bin_6`
15. `fft_bin_7`
16. `fft_bin_8`
17. `high_freq_energy`
### Color domain (44 features)
- HSV histograms plus BGR/HSV mean and standard deviation.
### HOG texture/shape domain (576 features)
- Histogram of Oriented Gradients descriptor from each 64x64 crop.

## Models trained on extracted features
| Model | Role |
|---|---|
| `logreg` | Logistic Regression on **StandardScaler**-normalized features (linear baseline). |
| `linear_svm` | **Linear SVM** on scaled features, suitable for the larger color+HOG vector. |
| `extra_trees` | **ExtraTreesClassifier** (350 trees) - stronger high-dimensional classical baseline. |
| `rf` | **RandomForestClassifier** (250 trees) - tree baseline + feature importance for charts. |

## Figures to include in the thesis / lecturer report
**Classical ML (this folder)**
- `chart_model_comparison.png` — Accuracy & F1-macro across ML models (no epoch-wise loss; ML is not trained by gradient descent here).
- `confusion_logreg.png`, `confusion_linear_svm.png`, `confusion_rf.png`, `confusion_extra_trees.png` - confusion matrices.
- `chart_domain_importance.png` — spatial/frequency/color/HOG contribution (RF feature importance).
- `ml/frequency_analysis/` — spatial/frequency summary CSVs + optional spectrum plots.
**Deep learning (separate runs; loss / training curves)**
- `runs/dl/trash_yolov8n_v3/results.png` — Ultralytics training curves (loss, mAP, precision, recall).
- `runs/dl/trash_yolov8n_v3/results.csv` — numeric log for custom plots.
- Run `python scripts/plot_training.py` → writes `training_curves.png` next to the chosen run’s `quality_check/`.
- `runs/dl/dl_baseline/training_loss.png` — tiny CNN baseline loss vs epoch on crops.

## Scope
- Mobile is intentionally excluded in this stage.
- Focus: feature extraction + classical ML model comparison.
- Classes in this run: plastic, glass, metal, paper, cardboard, organic

## Lecture workflow checklist
1. **Spatial + frequency domains** — each crop gets handcrafted **spatial** statistics (intensity, gradients, edges) and **frequency** descriptors (2D FFT radial energy bins + high-frequency summary).
2. **Comment how objects differ** — before judging ML scores, read the per-class notes below and `ml/frequency_analysis/spatial_summary.csv` + `frequency_summary.csv` (class-wise means).
3. **Extract features, then ML** — LogReg / Linear SVM / RandomForest / ExtraTrees are trained **only** on the stacked 637-D feature vectors from step 1 (not raw pixels inside this script).

## Pipeline (implementation order)
1. Crop objects from YOLO boxes; build the fixed-length feature vector per crop.
2. Export domain CSVs + object-difference commentary (spatial/frequency/color/HOG).
3. Fit classical ML on `X_train`; evaluate on `X_test`.

## Data
- Train object crops: **24000**
- Test object crops: **4168**
- Counts come from a **per-class** cap when enabled (each class can reach the cap independently; this is not a single 4000-total budget split across classes).
- Classes: plastic, glass, metal, paper, cardboard, organic

## Comments: how object classes differ (feature domains)
Each bullet compares that class’s **mean feature vector** to the **global mean** over all training crops: which domain (spatial / frequency / both) shows the largest shift, and which named descriptors move most.
- Compared to the dataset average, **organic** differs more in the **spatial** domain than in frequency: strongest spatial shifts involve `grad_std, p90_intensity, grad_mean`; strongest frequency shifts involve `fft_bin_1, fft_bin_2, fft_bin_3`.
  - *Scores:* overall `36.1271`, spatial `27.1808`, frequency `0.0332`.
- Compared to the dataset average, **paper** differs more in the **spatial** domain than in frequency: strongest spatial shifts involve `p50_intensity, grad_mean, mean_intensity`; strongest frequency shifts involve `fft_bin_1, fft_bin_2, fft_bin_3`.
  - *Scores:* overall `26.1043`, spatial `20.4569`, frequency `0.0335`.
- Compared to the dataset average, **plastic** differs more in the **spatial** domain than in frequency: strongest spatial shifts involve `grad_mean, p10_intensity, grad_std`; strongest frequency shifts involve `fft_bin_1, fft_bin_2, fft_bin_3`.
  - *Scores:* overall `20.8884`, spatial `9.4096`, frequency `0.0136`.
- Compared to the dataset average, **metal** differs more in the **spatial** domain than in frequency: strongest spatial shifts involve `grad_mean, grad_std, p10_intensity`; strongest frequency shifts involve `fft_bin_1, fft_bin_2, fft_bin_3`.
  - *Scores:* overall `18.2261`, spatial `15.8436`, frequency `0.0037`.
- Compared to the dataset average, **cardboard** differs more in the **spatial** domain than in frequency: strongest spatial shifts involve `grad_mean, p10_intensity, p50_intensity`; strongest frequency shifts involve `fft_bin_1, high_freq_energy, fft_bin_2`.
  - *Scores:* overall `14.5180`, spatial `6.3501`, frequency `0.0167`.
- Compared to the dataset average, **glass** differs more in the **spatial** domain than in frequency: strongest spatial shifts involve `p50_intensity, p10_intensity, grad_std`; strongest frequency shifts involve `fft_bin_1, high_freq_energy, fft_bin_3`.
  - *Scores:* overall `11.6731`, spatial `9.9506`, frequency `0.0075`.

## ML results (features → models)
| Model | Accuracy | F1-macro |
|---|---:|---:|
| extra_trees | 0.6312 | 0.6113 |
| rf | 0.6317 | 0.6111 |
| linear_svm | 0.5960 | 0.5642 |
| logreg | 0.5864 | 0.5558 |

## Model choice rationale
- **Logistic Regression:** simple linear baseline on standardized feature vectors.
- **Linear SVM:** scalable margin-based baseline for high-dimensional handcrafted descriptors.
- **Random Forest:** robust tree-based baseline and interpretable feature importance.
- **ExtraTrees:** tree ensemble baseline that often improves over RF on noisy, high-dimensional features.

## Chart comments
- `chart_domain_importance.png`: compares spatial, frequency, color, and HOG contribution based on model feature importance (not raw magnitude, so it is scale-safe).
- `chart_model_comparison.png`: compares Accuracy/F1 across ML models to justify chosen baseline.
