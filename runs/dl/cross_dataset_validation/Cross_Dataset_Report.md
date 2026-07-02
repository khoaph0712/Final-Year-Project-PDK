# Cross-Dataset Generalizability Validation Report

Domain-shift evaluation across two acquisition domains that share the same 7 material classes:

* **STUDIO** - curated/lab images (TrashNet + Kaggle Garbage Classification).
* **FIELD**  - real-world detection exports (Roboflow / TACO).

Domains are separated by the Roboflow/TACO filename markers (`_train_`/`_test_`/`_val_`/`rf_`); everything else is STUDIO. Both domains keep all 7 classes, so the earlier broken 3-class / 17.83% result is fixed.

* STUDIO samples used: **4900**
* FIELD samples used:  **4735**

## Results

| Direction | In-domain acc | Cross-domain acc | Gap (pp) | Cross macro-F1 |
|---|---:|---:|---:|---:|
| studio -> field | 80.00% | 39.81% | 40.19 | 0.4023 |
| field -> studio | 72.76% | 44.49% | 28.27 | 0.4074 |

The gap is the honest cost of domain shift: how much accuracy drops when the model meets an acquisition domain it was not trained on.

### STUDIO -> FIELD cross-domain report

```text
              precision    recall  f1-score   support

     plastic       0.27      0.14      0.19       700
       glass       0.58      0.35      0.44       700
       metal       0.56      0.37      0.44       700
       paper       0.56      0.48      0.52       700
   cardboard       0.61      0.62      0.61       700
     organic       0.35      0.37      0.36       700
  Background       0.18      0.47      0.26       535

    accuracy                           0.40      4735
   macro avg       0.44      0.40      0.40      4735
weighted avg       0.45      0.40      0.41      4735
```

### FIELD -> STUDIO cross-domain report

```text
              precision    recall  f1-score   support

     plastic       0.51      0.28      0.36       700
       glass       0.41      0.64      0.50       700
       metal       0.40      0.52      0.46       700
       paper       0.41      0.77      0.54       700
   cardboard       0.69      0.70      0.69       700
     organic       0.55      0.11      0.19       700
  Background       0.17      0.09      0.12       700

    accuracy                           0.44      4900
   macro avg       0.45      0.44      0.41      4900
weighted avg       0.45      0.44      0.41      4900
```