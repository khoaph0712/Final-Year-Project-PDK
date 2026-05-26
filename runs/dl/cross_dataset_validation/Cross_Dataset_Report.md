# Cross-Dataset Generalizability Validation Report

This report details the domain shift evaluation to prove generalizability. We trained strictly on TrashNet and tested on complex TACO outdoor samples.

*   **Training Domain A (TrashNet):** 4234 samples
*   **Testing Domain B (TACO):** 600 samples
*   **Domain-Shift Accuracy:** **17.83%**

### Detailed Classification Report:
```text
              precision    recall  f1-score   support

     plastic       0.00      0.00      0.00         0
       glass       0.53      0.48      0.51       200
       metal       0.00      0.00      0.00         0
       paper       0.00      0.00      0.00         0
   cardboard       0.00      0.00      0.00         0
     organic       0.00      0.00      0.00       200
  Background       0.21      0.05      0.08       200

    accuracy                           0.18       600
   macro avg       0.11      0.08      0.08       600
weighted avg       0.25      0.18      0.20       600

```
