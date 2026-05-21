# PyTorch ANN vs. Traditional ML Model Comparison

| Model | Accuracy | Macro F1-score | Framework | Notes |
|---|---:|---:|---|---|
| **ANN (MLP)** | **0.8023** | **0.8017** | **PyTorch** | **3 hidden layers, custom regularization** |
| XGBOOST | 0.6526 | 0.6441 | scikit-learn / xgboost | Traditional ML Baseline |
| RF | 0.5880 | 0.5799 | scikit-learn / xgboost | Traditional ML Baseline |
| LINEAR_SVM | 0.5520 | 0.5420 | scikit-learn / xgboost | Traditional ML Baseline |
| DECISION_TREE | 0.3937 | 0.3907 | scikit-learn / xgboost | Traditional ML Baseline |

### Analysis Rationale

The PyTorch Multi-Layer Perceptron outperforms XGBOOST by **14.97%** in accuracy. This confirms that the non-linear transformations and standard regularization (Batch Normalization, Dropout) applied in PyTorch successfully capture the multi-modal distribution of handcrafted Color, Texture, Shape, and HOG features better than decision trees.
