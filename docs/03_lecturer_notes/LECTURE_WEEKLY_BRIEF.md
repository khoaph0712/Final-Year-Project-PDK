# Weekly project brief — lecture update

**Project:** WasteWise — waste-type classification (YOLO + classical ML + mobile path)  
**Student:** Khoa Phung  
**Period covered:** This week  
**Purpose:** Summarise progress for the lecture and invite guidance on next steps.

---

## 1 · One-line summary

The repository is **version-controlled on GitHub**, the **ML analysis pipeline** now follows your **spatial → frequency → comment on object differences → then ML** workflow, the **dataset sampling** targets **up to 4 000 object crops per class** with `**other` excluded** for the six main classes, and the **project folders** are **reorganised** so ML outputs, DL training runs, and comparison reports are easy to find.

---

## 2 · What was completed this week


| Area                             | Done                                                                                                                                                                                                                                                                                                                                                                                          |
| -------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Version control**              | Local Git initialised; project pushed to **[https://github.com/khoaph0712/FYP_v2](https://github.com/khoaph0712/FYP_v2)**.                                                                                                                                                                                                                                                                    |
| **Lecturer workflow (ML stage)** | `feature_ml_analysis.py` documents and runs: **(1)** spatial + FFT-based **frequency** features per crop, **(2)** **per-class commentary** on how objects differ from the global mean (spatial vs frequency emphasis), **(3)** **classical ML only after** feature extraction (LogReg, SVM-RBF, RandomForest). Outputs: `REPORT.md`, `object_difference.json`, `ml/frequency_analysis/*.csv`. |
| **Sampling policy**              | Train/test caps are **per class** (not a single 4 000 total). Default: exclude class `**other`**; **4 000** train crops/class and **4 000** test crops/class where data allows.                                                                                                                                                                                                               |
| **Experimental results**         | One full run saved under `runs/ml/feature_ml_6class_4k/`. On that run, **test accuracy was ~38–47%** (best: Random Forest ~**47%**). This is **below** an informal ~80% goal — useful as a baseline for the report and for discussing what to improve next.                                                                                                                                   |
| **Repository layout**            | `**runs/ml/`** — feature + ML reports; `**runs/dl/`** — YOLO runs + tiny-CNN baseline; `**runs/comparisons/**` — ML vs DL chart + report; `**ml/frequency_analysis/**` — domain summaries. **README** and script **defaults** updated (canonical dataset `**merged_dataset_v3`**, latest YOLO run `**runs/dl/trash_yolov8n_v3`**).                                                            |
| **Housekeeping**                 | Removed unused root calibration `.npy`; `**.gitignore`** extended so virtualenv folders are not committed.                                                                                                                                                                                                                                                                                    |


---

## 3 · How this maps to your lecture guide

1. **Frequency domain + spatial domain** — Implemented as a **fixed 17-D vector** per object crop: **8 spatial** descriptors (intensity, percentiles, gradients, edge density) + **9 frequency** descriptors (FFT radial energy bins + high-frequency summary).
2. **Comment the difference of objects** — For each class, the pipeline reports **which domain shifts more** versus the dataset mean and **which named features** move most; this is written into `**REPORT.md`** and `**object_difference.json`**.
3. **Extract features, then run ML** — Code order and report structure state explicitly: **features first**, **interpretation second**, **ML third** (no raw-pixel classical ML in this script).

---

## 4 · Where to look in the repo (for demo / discussion)

- **Lecture-aligned ML report:** `runs/ml/feature_ml_6class_4k/REPORT.md` (and sibling `metrics_summary.json`, confusion PNGs).  
- **Object-difference / domain tables:** `runs/ml/feature_ml_6class_4k/object_difference.json`, `ml/frequency_analysis/spatial_summary.csv`, `ml/frequency_analysis/frequency_summary.csv`.  
- **ML vs DL comparison (chart):** `runs/comparisons/model_comparison/chart_ml_vs_dl.png` and `REPORT.md` (regenerate with `python scripts/compare_ml_dl.py` after changing inputs).  
- **Deep learning (YOLO) artefacts:** `runs/dl/trash_yolov8n_v3/` (weights, curves, quality checks).

---

## 5 · Honest limitations (good topics for lecturer feedback)

- **Accuracy gap:** Handcrafted features + classical ML are a **deliberate baseline**; ~47% test accuracy suggests **feature strength**, **class imbalance on test** (e.g. fewer glass crops in the test cap in one run), or **hard class overlap** — I would like guidance on whether to prioritise **better features / small CNN embeddings**, **dataset cleaning**, or **reporting focus** for the FYP.  
- **Crops vs whole images:** Caps apply to **bounding-box crops** (multiple crops per image possible), not necessarily “4 000 unique images per class” — confirm if the module expects **image-level** statistics instead.  
- **Mobile / DL track:** ONNX/TFLite export paths in docs now point to **v3** weights; integration testing on device is a **next** milestone.

---

## 6 · What I would like guidance on

1. Is the **spatial + frequency + comment + ML** narrative in `**REPORT.md`** sufficient for the **analysis chapter**, or should I add **e.g. statistical tests / PCA plots / per-domain confusion**?
2. For the **~80%** expectation: is that intended for **ML-on-features**, **YOLO mAP**, or **end-to-end app accuracy** — so I align metrics and effort correctly?
3. Any **preferred folder or report naming** for formal submission (e.g. all figures under `figures/` for the thesis)?

---

*End of brief — short enough to skim before class; happy to walk through the repo or charts in the session.*