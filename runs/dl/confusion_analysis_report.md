# Deep Learning Confusion Matrix & Background Error Analysis

This report evaluates model predictions against the balanced test dataset (250 crops per class) with a specific deep-dive into how PyTorch ANN and MobileNetV2 CNN models interact with the **Background** (negative noise) class.

## 1. Class Performance Overview

### PyTorch ANN (637 Handcrafted Features) Confusion Matrix

![ANN Confusion Matrix](file:///C:/FYP_v2/runs/dl/ann_637/confusion_ann.png)

### MobileNetV2 CNN (Raw Image Crops) Confusion Matrix

![CNN Confusion Matrix](file:///C:/FYP_v2/runs/dl/cnn_mobilenet/confusion_cnn.png)


## 2. Background Class Deep-Dive

The background class represents floor textures, carpets, grass, and table surfaces where waste is absent. Evaluating how other classes leak into the background (False Negatives) or how the background is identified as waste (False Positives) is vital for physical robotic and mobile applications.


### PyTorch ANN (MLP) Error Profile

- **False Positive Rate (Ghost Waste)**: **14.40%** (72 / 500 background samples misclassified as waste items).
  - *Impact*: In a robotic or mobile trash bin setting, this means the AI sees 'Ghost waste' on clear surfaces and triggers sorting mechanisms unnecessarily.
  - *Top False Positive Leakage*: Background textures are most frequently mistaken for **cardboard** (22 instances).
- **False Negative Rate (Waste Blindness)**: **2.77%** (83 / 3000 actual waste items misclassified as background).
  - *Impact*: This represents blind spots where the AI completely ignores waste items, assuming they are simply floor textures.
  - *Top False Negative Leakage*: Waste item **paper** is most frequently ignored as background (20 instances).


### MobileNetV2 CNN Error Profile

- **False Positive Rate (Ghost Waste)**: **6.80%** (34 / 500 background samples misclassified as waste items).
  - *Impact*: In a robotic or mobile trash bin setting, this means the AI sees 'Ghost waste' on clear surfaces and triggers sorting mechanisms unnecessarily.
  - *Top False Positive Leakage*: Background textures are most frequently mistaken for **cardboard** (13 instances).
- **False Negative Rate (Waste Blindness)**: **0.57%** (17 / 3000 actual waste items misclassified as background).
  - *Impact*: This represents blind spots where the AI completely ignores waste items, assuming they are simply floor textures.
  - *Top False Negative Leakage*: Waste item **plastic** is most frequently ignored as background (7 instances).


## 3. Background Leakage Comparison Table

| Model | Ghost Waste FP Rate (%) | Waste Blindness FN Rate (%) | Primary FP Leakage Class | Primary FN Leakage Class |
|---|---:|---:|---|---|
| PyTorch ANN | 14.40% | 2.77% | cardboard | paper |
| MobileNetV2 CNN | 6.80% | 0.57% | cardboard | plastic |

## 4. Engineering Recommendations

1. **Texture Feature Augmentation**: The ANN relies heavily on texture features (LBP/GLCM). Backgrounds with high texture similarity to paper/cardboard are mistaken. We should expand texture training to cover a wider variety of floors.

2. **Context-Aware Scaling**: The CNN performs better due to contextual representations learned via MobileNetV2's deep convolutional layers. However, when transparent waste (glass/plastic) is present, the model can struggle to differentiate it from floors. Adding alpha-mask or edge highlight augmentations will resolve this.
