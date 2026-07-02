# Classifier: original vs leakage-quarantined test set

| test set | images | accuracy | macro-F1 |
|---|---:|---:|---:|
| original_test | 5600 | 94.30% | 0.9431 |
| clean_test | 3194 | 91.77% | 0.9114 |

## original_test

```text
              precision    recall  f1-score   support

     plastic       0.96      0.93      0.94       800
       glass       0.96      0.91      0.94       800
       metal       0.91      0.93      0.92       800
       paper       0.92      0.93      0.93       800
   cardboard       0.92      0.96      0.94       800
     organic       0.96      0.97      0.97       800
  Background       0.97      0.97      0.97       800

    accuracy                           0.94      5600
   macro avg       0.94      0.94      0.94      5600
weighted avg       0.94      0.94      0.94      5600
```

## clean_test

```text
              precision    recall  f1-score   support

     plastic       0.94      0.91      0.93       445
       glass       0.95      0.86      0.90       407
       metal       0.87      0.89      0.88       433
       paper       0.86      0.85      0.86       368
   cardboard       0.85      0.92      0.89       402
     organic       0.94      0.97      0.95       503
  Background       0.97      0.97      0.97       636

    accuracy                           0.92      3194
   macro avg       0.91      0.91      0.91      3194
weighted avg       0.92      0.92      0.92      3194
```