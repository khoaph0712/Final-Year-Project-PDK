# Feature + ML Analysis Report

## Scope
- Mobile is intentionally excluded in this stage.
- Focus: feature extraction + classical ML model comparison.

## Pipeline
1. Extract object crops from YOLO labels.
2. Extract spatial-domain and frequency-domain features.
3. Train ML models first (LogReg, SVM-RBF, RandomForest).
4. Compare with accuracy/F1/confusion matrix.

## Data
- Train objects: **8000**
- Test objects: **3000**
- Classes: plastic, glass, metal, paper, cardboard, organic, other

## Model choice rationale
- **Logistic Regression:** simple linear baseline on standardized feature vectors.
- **SVM (RBF):** captures non-linear class boundaries from handcrafted features.
- **Random Forest:** robust tree-based baseline and interpretable feature importance.

## Results
| Model | Accuracy | F1-macro |
|---|---:|---:|
| rf | 0.4747 | 0.4548 |
| svm_rbf | 0.4493 | 0.4215 |
| logreg | 0.3633 | 0.3231 |

## Class difference comments (objects)
Top distinct feature indices are reported to explain which classes differ most in spatial/frequency signatures.
- **organic**: distinctiveness `28.1252`, top feature idx `[4, 6, 5]`
- **paper**: distinctiveness `18.9130`, top feature idx `[3, 0, 5]`
- **other**: distinctiveness `14.8823`, top feature idx `[4, 3, 0]`
- **metal**: distinctiveness `13.1934`, top feature idx `[5, 6, 2]`
- **plastic**: distinctiveness `8.9151`, top feature idx `[5, 6, 4]`
- **glass**: distinctiveness `8.5663`, top feature idx `[3, 6, 2]`
- **cardboard**: distinctiveness `7.1888`, top feature idx `[5, 2, 4]`

## Chart comments
- `chart_domain_importance.png`: compares spatial vs frequency contribution based on model feature importance (not raw magnitude, so it is scale-safe).
- `chart_model_comparison.png`: compares Accuracy/F1 across ML models to justify chosen baseline.
