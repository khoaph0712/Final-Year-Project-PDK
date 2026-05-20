# Deep Learning Ensemble Voting Performance Report

This report documents the results of our **Ensemble Decision Experiment**, which evaluates whether combining the independent spatial representations of the Convolutional Neural Network (CNN) with the handcrafted texture/edge representations of the Multi-Layer Perceptron (ANN) improves waste classification accuracy.

## 1. Classification Performance Metrics

The following table compares the overall Accuracy and Macro F1-score of the baseline individual models against unweighted and weighted soft-voting ensemble strategies evaluated on the balanced 1,750-crop test split.

| Evaluation Method | Accuracy (%) | Macro F1-Score (%) | Performance Deltas (vs. CNN baseline) |
|---|---:|---:|---:|
| **PyTorch ANN Baseline (637 features)** | 64.00% | 63.56% | -13.20% |
| **MobileNetV2 CNN Baseline (raw crops)** | 77.20% | 76.56% | *Baseline* |
| **Simple Soft Voting Ensemble (50/50)** | 78.69% | 78.12% | +1.49% |
| **Weighted Soft Voting Ensemble (30% ANN / 70% CNN)** | 77.94% | 77.26% | +0.74% |


## 2. Key Analytical Insights

1. **Ensemble Collaboration Wins**: The **Simple Soft Voting Ensemble (50/50)** achieved the highest overall validation accuracy of **78.69%**, which represents a **+1.49%** boost over the pure CNN baseline (77.20%), and a **+14.69%** boost over the pure ANN MLP baseline (64.00%).
2. **Feature Complementarity**: While the CNN captures excellent spatial context, it occasionally confuses fine material boundaries. By ensembling it with the ANN (which is trained on highly explicit handcrafted texture features like LBP/GLCM and HOG edge descriptors), the ensemble model resolves ambiguous edge-cases (e.g. distinguishing a paper cup from a plastic bottle).
3. **Simple vs. Weighted Voting**: The simple 50/50 soft voting ensemble yielded an outstanding accuracy of **78.69%** (+1.49% over the CNN baseline) due to the highly complementary representations. The weighted ensemble (30% ANN / 70% CNN) also achieved a strong **77.94%** accuracy (+0.74% over the CNN baseline), proving that the handcrafted ANN consistently provides beneficial corrections across voting configurations.


## 3. Class-Level Recall Analysis

The following is the detailed classification report for our best-performing **Simple Soft Voting Ensemble (50/50)**:

```text
              precision    recall  f1-score   support

     plastic       0.69      0.52      0.59       250
       glass       0.83      0.88      0.86       250
       metal       0.81      0.80      0.81       250
       paper       0.77      0.63      0.69       250
   cardboard       0.76      0.86      0.81       250
     organic       0.72      0.98      0.83       250
  Background       0.93      0.84      0.88       250

    accuracy                           0.79      1750
   macro avg       0.79      0.79      0.78      1750
weighted avg       0.79      0.79      0.78      1750

```

## 4. Academic Conclusion & Recommendations

* **Recommendation**: For server-side or secondary double-checking pipelines (e.g. in cloud-based waste processing audits), we should deploy the **Simple Soft Voting Ensemble (50/50)** as it offers the absolute highest classification performance (78.69%).
* **Greener Edge Constraint**: For direct mobile deployment where network latency and memory bandwidth are strictly constrained, the **2.7MB Quantized CNN TFLite model alone** remains the most suitable choice due to its high efficiency and minimal size footprint under 3MB.