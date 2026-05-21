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

- **Color (256 Features)**: 144-bin RGB Histograms (3 channels * 48 bins) + 112-bin HSV Histograms (H:48, S:32, V:32).

- **Texture (47 Features)**: 10-bin Uniform LBP Histogram + 37-bin Gray-Level Co-occurrence Matrix (GLCM/Haralick) Descriptors.

- **Shape/Geometric (10 Features)**: 7 log-transformed scale/rotation-invariant Hu Moments + Area, Perimeter, and Circularity.

- **Edge/HOG (324 Features)**: Gradient orientation histogram (64x64 window, 32x32 block, 16x16 cell/stride, 9 bins).


### Feature Group Importance Contributions

| Feature Group | Random Forest Contribution (%) |
|---|---:|
| Color Features (256) | 49.14% |
| Texture Features (47) | 12.92% |
| Shape/Geometric (10) | 1.42% |
| Edge/HOG Features (324) | 36.52% |


## 4. Machine Learning Baseline Comparison

| Model | Accuracy | Macro F1-score |
|---|---:|---:|
| xgboost | 0.6526 | 0.6441 |
| rf | 0.5880 | 0.5799 |
| linear_svm | 0.5520 | 0.5420 |
| decision_tree | 0.3937 | 0.3907 |


## 5. Noise and Background Performance Assessment

Please inspect the saved confusion matrix files (`confusion_*.png`) to see how the models separate the `Background` class from actual waste items (`plastic`, `glass`, `metal`, etc.). Training models with a high-fidelity `Background` class prevents false positive classifications on floors, tables, and grass in edge deployments.
