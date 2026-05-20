# Phase 2 Experiment Audit

This audit captures the next project direction after the first round of dataset analysis, ML training, and ANN/CNN baselines. It is not the final report. It is a working checklist for making the experiments cleaner, fairer, and easier to defend.

## Current Project Direction

YOLO is paused. The active story is:

> The original merged dataset was heavily imbalanced, so the project now separates datasets by source, performs EDA, extracts handcrafted features, and uses balanced experiments to compare classical ML, ANN, and CNN models fairly.

The strongest current path is to make the balanced dataset evidence complete and consistent before writing final report/slides.

## Evidence Already Exists

### Dataset EDA and Source Analysis

| Dataset/source | Status | Evidence |
|---|---|---|
| TrashNet | Complete enough for current phase | `runs/external_dataset_eda/trashnet/`, `prepared_datasets/trashnet/` |
| GINI | Complete enough as binary/background-style experiment | `runs/external_dataset_eda/gini/`, `prepared_datasets/gini_binary/` |
| TACO official partial | Complete enough to show difficulty | `runs/external_dataset_eda/taco_official_partial/`, `prepared_datasets/taco_official_partial/` |
| Merged 6-class project dataset | Complete enough for main balanced comparison | `runs/ml/balanced_by_dataset/merged_6class/`, `runs/dl/balanced_ann_cnn/merged_6class/` |

### 637 Feature Explanation

This is already defensible in code and docs. The handcrafted ML feature vector is:

| Feature group | Count | Meaning |
|---|---:|---|
| Spatial | 8 | intensity, gradient, and edge statistics |
| Frequency/FFT | 9 | radial frequency bands plus high-frequency energy |
| Color | 44 | HSV histograms plus BGR/HSV mean and standard deviation |
| HOG | 576 | local gradient orientation / texture / shape descriptor |
| Total | 637 | `8 + 9 + 44 + 576` |

Primary implementation: `scripts/feature_ml_analysis.py`.

## Current Result Snapshot

### Main Balanced Merged 6-Class Experiment

ML result path: `runs/ml/balanced_by_dataset/merged_6class/`

| Model | Accuracy | Macro F1 |
|---|---:|---:|
| XGBoost | 0.6497 | 0.6422 |
| Extra Trees | 0.6110 | 0.6031 |
| Random Forest | 0.5917 | 0.5814 |
| Linear SVM | 0.5352 | 0.5262 |
| Logistic Regression | 0.5234 | 0.5161 |
| Decision Tree | 0.4303 | 0.4253 |

Fair ANN/CNN result path: `runs/dl/balanced_ann_cnn/merged_6class_fair/`

| Model | Accuracy | Macro F1 | Note |
|---|---:|---:|---|
| ANN | 0.3924 | 0.3755 | 5 quick epochs, image size 64 |
| CNN | 0.4614 | 0.4386 | 5 quick epochs, image size 64 |

Important fix completed: the first ANN/CNN balanced run dropped `organic` because `rf_food_waste` was missing from the source filter. The fair rerun adds `rf_food_waste`, keeps the same 6 project classes as ML, and drops only `other` because it has no test support.

### TrashNet Experiment

ML result path: `runs/ml/by_dataset/trashnet/`

| Model | Accuracy | Macro F1 |
|---|---:|---:|
| XGBoost | 0.8103 | 0.7841 |
| Random Forest | 0.7391 | 0.6688 |
| Extra Trees | 0.7115 | 0.6597 |
| Logistic Regression | 0.6166 | 0.5622 |
| Linear SVM | 0.5613 | 0.5155 |
| Decision Tree | 0.5296 | 0.5073 |

ANN/CNN result path: `runs/dl/balanced_ann_cnn/trashnet/`

| Model | Accuracy | Macro F1 | Note |
|---|---:|---:|---|
| ANN | 0.3458 | 0.2792 | 5 quick epochs |
| CNN | 0.4167 | 0.4048 | 5 quick epochs |

Interpretation: TrashNet is clean-background classification data, so handcrafted features work well. The CNN baseline is currently too shallow/short-trained to beat ML.

### GINI Binary Experiment

ML result path: `runs/ml/external_only/gini_binary/`

| Model | Accuracy | Macro F1 |
|---|---:|---:|
| Random Forest | 0.8837 | 0.8516 |
| XGBoost | 0.8760 | 0.8484 |
| Extra Trees | 0.8682 | 0.8348 |
| Decision Tree | 0.8062 | 0.7719 |
| Logistic Regression | 0.8140 | 0.7688 |
| Linear SVM | 0.7829 | 0.7427 |

ANN/CNN result path: `runs/dl/external_only_ann_cnn/gini_binary/`

| Model | Accuracy | Macro F1 | Note |
|---|---:|---:|---|
| ANN | 0.6211 | 0.3831 | class imbalance hurts macro F1 |
| CNN | 0.7895 | 0.7883 | useful binary baseline |

Interpretation: GINI is useful for trash-vs-background behavior, not final multi-class sorting.

### TACO Official Partial Experiment

ML result path: `runs/ml/by_dataset/taco_official_partial/`

| Model | Accuracy | Macro F1 |
|---|---:|---:|
| XGBoost | 0.1399 | 0.0675 |
| Logistic Regression | 0.1145 | 0.0479 |
| Extra Trees | 0.1349 | 0.0457 |
| Random Forest | 0.1552 | 0.0411 |
| Linear SVM | 0.0840 | 0.0368 |
| Decision Tree | 0.0611 | 0.0278 |

ANN/CNN result path: `runs/dl/balanced_ann_cnn/taco_official_partial/`

| Model | Accuracy | Macro F1 | Note |
|---|---:|---:|---|
| ANN | 0.0573 | 0.0028 | many small classes, very low support |
| CNN | 0.0573 | 0.0028 | many small classes, very low support |

Interpretation: this should be used as "real-world TACO-style data is hard" evidence, not as the main success experiment.

## Artifact Completeness

The important artifact types are mostly present:

| Artifact | Current status |
|---|---|
| ML metrics summaries | Present |
| ML classification reports | Present |
| ML confusion matrices | Present |
| ML feature/domain charts | Present |
| ANN/CNN saved models | Present in balanced ANN/CNN folders |
| ANN/CNN training logs | Present as JSON and CSV |
| ANN/CNN accuracy/loss curves | Present |
| ANN/CNN confusion matrices | Present |
| ANN/CNN classification reports | Present |
| Source-level EDA charts | Present for TrashNet, GINI, TACO partial |

## Main Problems To Fix Next

1. ANN/CNN baselines are probably under-trained

   Most balanced ANN/CNN runs use only 5 epochs at image size 64. This is acceptable as a baseline, but if time allows, one stronger CNN run should be trained for the main balanced dataset with the same classes as ML.

2. TACO should not be treated as a balanced success dataset

   TACO partial has many classes with tiny validation/test counts. It is useful for explaining difficulty, noise, and imbalance. It should not be the headline model-performance result.

3. Master comparison table was scattered

   Results existed across multiple folders. A single CSV has now been added at `runs/comparisons/phase2_model_comparison.csv`.

4. Reproducibility entrypoint needs a Phase 2 command order

   `START_HERE.md` has many commands, but the next working phase needs a shorter "run these for balanced comparison" list.

## Recommended Immediate Next Work

### Step 1 - Fix fair main comparison

Status: done.

ANN/CNN was rerun for the merged balanced dataset using the same 6 project classes as the ML experiment. The goal is not maximum accuracy; the goal is a fair model-family comparison.

Expected output:

```text
runs/dl/balanced_ann_cnn/merged_6class_fair/
  REPORT.md
  metrics_summary.json
  class_support.json
  ann/
  cnn/
```

### Step 2 - Build one master comparison table

Status: done.

Create a single table with:

```text
Dataset | Split setup | Class count | Model family | Model | Accuracy | Macro F1 | Evidence path | Notes
```

This should include:

- merged balanced ML
- merged balanced ANN/CNN
- TrashNet ML
- TrashNet ANN/CNN
- GINI binary ML
- GINI binary ANN/CNN
- TACO partial ML
- TACO partial ANN/CNN

Output: `runs/comparisons/phase2_model_comparison.csv`

### Step 3 - Write short error-analysis notes

For the main balanced dataset only, inspect the best ML confusion matrix and best CNN confusion matrix. Summarize:

- easiest classes,
- most confused classes,
- likely reason for confusion,
- whether the issue is feature weakness, visual similarity, or dataset support.

### Step 4 - Freeze Phase 2 evidence

After the fair rerun and master comparison table, avoid adding new data unless it directly fixes a documented problem.

## Decision

Do not restart YOLO. The next engineering work should be:

1. short error analysis,
2. update `START_HERE.md` with the Phase 2 evidence path,
3. optionally train a stronger CNN for the main fair 6-class dataset if time allows.
