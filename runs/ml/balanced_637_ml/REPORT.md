# Balanced 637-Feature Machine Learning Report

## 1. Raw Dataset Class Distribution (Statistical Analysis)

| Class | Train Bboxes | Valid Bboxes | Test Bboxes | Total Bboxes |
|---|---:|---:|---:|---:|
| plastic | 17,786 | 1,792 | 2,214 | 21,792 |
| glass | 7,020 | 2,520 | 203 | 9,743 |
| metal | 12,292 | 2,144 | 770 | 15,206 |
| paper | 7,286 | 1,012 | 2,145 | 10,443 |
| cardboard | 17,377 | 3,386 | 1,042 | 21,805 |
| organic | 22,785 | 2,766 | 2,895 | 28,446 |
| other | 5,361 | 1,099 | 325 | 6,785 |


## 2. Dataset Balancing Strategy

- **Undersampling**: Capped major classes to create a uniform training distribution.

- **Oversampling & Augmentation**: For minority classes, applied flips, rotations, and intensity shifts to achieve a fully balanced set.

- **Negative Samples (Background Class)**: Added real-world empty images and synthetic textures (grass, concrete, carpet) to represent environmental noise, preventing false positives in real-world scenarios.


## 3. Handcrafted Feature Extractor Layout (Exactly 637 Features)

- **Color (30 Features)**: 15-bin RGB Histograms + 15-bin HSV Histograms.

- **Texture (21 Features)**: 10-bin LBP Histogram + 11 statistical Haralick GLCM Descriptors.

- **Shape/Geometric (10 Features)**: 7 Hu Moments + 1 Area + 1 Perimeter + 1 Circularity.

- **Edge/HOG (576 Features)**: Shape orientation orientation histogram.


### Feature Group Importance Contributions

| Feature Group | Random Forest Contribution (%) |
|---|---:|
| Color Features (30) | 14.36% |
| Texture Features (21) | 9.05% |
| Shape/Geometric (10) | 2.43% |
| Edge/HOG Features (576) | 74.15% |


## 4. Machine Learning Baseline Comparison

| Model | Accuracy | Macro F1-score |
|---|---:|---:|
| xgboost | 0.5771 | 0.5707 |
| rf | 0.5400 | 0.5326 |
| decision_tree | 0.4243 | 0.4261 |
| linear_svm | 0.4271 | 0.4225 |


## 5. Noise and Background Performance Assessment

Please inspect the saved confusion matrix files (`confusion_*.png`) to see how the models separate the `Background` class from actual waste items (`plastic`, `glass`, `metal`, etc.). Training models with a high-fidelity `Background` class prevents false positive classifications on floors, tables, and grass in edge deployments.
