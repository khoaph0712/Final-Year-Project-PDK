# PyTorch ANN vs. Traditional ML Model Comparison

| Model | Accuracy | Macro F1-score | Framework | Notes |
|---|---:|---:|---|---|
| **ANN (MLP)** | **0.6400** | **0.6356** | **PyTorch** | **3 hidden layers, custom regularization** |
| XGBOOST | 0.5771 | 0.5707 | scikit-learn / xgboost | Traditional ML Baseline |
| RF | 0.5400 | 0.5326 | scikit-learn / xgboost | Traditional ML Baseline |
| DECISION_TREE | 0.4243 | 0.4261 | scikit-learn / xgboost | Traditional ML Baseline |
| LINEAR_SVM | 0.4271 | 0.4225 | scikit-learn / xgboost | Traditional ML Baseline |

### Analysis Rationale

The PyTorch Multi-Layer Perceptron outperforms XGBOOST by **6.29%** in accuracy. This confirms that the non-linear transformations and standard regularization (Batch Normalization, Dropout) applied in PyTorch successfully capture the multi-modal distribution of handcrafted Color, Texture, Shape, and HOG features better than decision trees.
