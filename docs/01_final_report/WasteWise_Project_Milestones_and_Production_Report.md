# WasteWise: Automated Waste Classification & Localization
## Comprehensive Project Milestones & Production Deployment Report

---

## 1 · Executive Summary & Lifecycle Overview

**WasteWise** is an advanced Final Year Project (FYP) designed to automate the process of waste identification, material classification, and localization. To address the problem of real-world waste sorting under unconstrained environmental conditions, the project has been structured into two main technical branches:
1. **Explainable Handcrafted ML Branch**: Extracts $637$ spatial, color, and gradient features per object crop and evaluates them using classical classifiers. A Principal Component Analysis (PCA) sweep was conducted to determine the minimum feature space required to sustain classification accuracy.
2. **Deep Learning (DL) Production Branch**: An end-to-end, high-accuracy deployment utilizing a **Localization-First** architecture. It uses a **YOLOv11** detector as an object proposer and a **ConvNeXt-Tiny** classifier (ensembled with handcrafted features) to verify cropped bounding boxes.

```mermaid
flowchart TD
    subgraph Raw Intake
        A["Waste Image Streams"] --> B["Data Wrangling & Splits"]
    end

    subgraph Feature Engineering (ML Track)
        B --> C["Handcrafted Feature Extractor (637-D)"]
        C --> D["Feature Domain Splits: Spatial (8) + FFT (9) + Color (44) + HOG (576)"]
        D --> E["Classical Classifiers: SVM, RF, XGBoost, ExtraTrees"]
        D --> F["PCA Compression: 637-D -> 128-D (-2.52% Acc)"]
    end

    subgraph Edge Production (DL Track)
        B --> G["Stage 1: YOLOv11 Object Proposal"]
        G --> H["Adaptive Scene Engine (Sensitivity Sweep / Clutter Gate)"]
        H --> I["Physical Size Filter (Reject <24px Bounding Boxes)"]
        I --> J["Stage 2: Crop Extraction & Padding (10px)"]
        J --> K["ConvNeXt-Tiny Material Classifier"]
        K --> L["Dynamic Alpha Blending (YOLO + CNN probabilities)"]
        L --> M["Bayesian Context Prior Fusion (Beach, Grass, Street, Indoor)"]
        M --> N["Class-Specific Gating & Bottom-Up Aggregation"]
    end
    
    N --> O["Production Web App & Docker API (Hugging Face Spaces)"]
    E --> P["Academic Report & Explainability Audits"]
    F --> P
```

---

## 2 · Chronological Project Milestones

The project progressed from raw datasets to production deployment through eight key milestones:

```
+-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| Milestone 1                  Milestone 2                  Milestone 3                  Milestone 4                  Milestone 5                  Milestone 6                  Milestone 7                  Milestone 8                                                                                            |
| Data Prep & Splitting  --->  Feature Engineering  --->  ML Baseline Sweeps  --->  PCA Dimensionality  --->  Legacy DL Baselines  --->  Localization-First  --->  Production Tweaks   --->  Hugging Face Deploy                                                                                        |
| roboflow/merged_v3/v5        637-D Vector Extraction      XGBoost, RF, SVM, LogReg     637-D to 128-D Sweep         ANN, CNN, Soft Voting        Pipeline Refactor            Adaptive Engine, Bayes       Docker, LFS models                                                                                     |
+-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
```

### Milestone 1: Dataset Acquisition, Cleaning, and Splitting
- **Objective**: Standardize unconstrained waste images from various sources (such as Roboflow, TrashNet, and the TACO dataset) into structured splits.
- **Outcome**: Created `merged_dataset_v3` (114,220 bounding boxes) and subsequently finalized `merged_dataset_v5` (29,639 balanced crops across 7 classes) alongside `external_datasets/super_yolo_dataset` (23,929 images containing 102,777 bounding boxes).

### Milestone 2: Handcrafted Feature Engineering
- **Objective**: Extract explainable, mathematically defined features from crops to satisfy academic auditing requirements.
- **Outcome**: Designed a $637$-dimensional feature extractor containing $8$ spatial/gradient statistics, $9$ Fast Fourier Transform (FFT) radial bin descriptors, $44$ HSV color statistics, and $576$ Histogram of Oriented Gradients (HOG) texture descriptors.

### Milestone 3: Classical ML Comparison
- **Objective**: Establish baseline benchmarks on the handcrafted features using explainable models.
- **Outcome**: Trained Logistic Regression, Support Vector Machines (SVM), Random Forest, ExtraTrees, and XGBoost. XGBoost emerged as the top classifier, achieving $67.42\%$ accuracy on the legacy lecturer-facing split.

### Milestone 4: Dimensionality Reduction & PCA Sweep
- **Objective**: Compress the $637$-D feature space to reduce storage and inference latency.
- **Outcome**: Swept PCA components from $637$ down to $32$. Confirmed that compressing the representation to $128$ components (retaining $99.90\%$ of variance) only costs a minor $2.52$ percentage point drop in SVM accuracy.

### Milestone 5: Legacy Deep Learning Baselines
- **Objective**: Train basic neural network classifiers directly on raw crop pixels.
- **Outcome**: Evaluated Artificial Neural Networks (ANN) and Convolutional Neural Networks (CNN) on crop inputs. An ensemble combining CNN feature vectors with ANN handcrafted features by soft voting reached $78.69\%$ accuracy.

### Milestone 6: Architectural Rework to Localization-First
- **Objective**: Redesign the Deep Learning branch. Instead of classifying full scenes, run object detection first, extract crops, and verify them individually.
- **Outcome**: Developed the hierarchical YOLOv11 + ConvNeXt-Tiny pipeline. The system runs detection first, crops the bounding boxes, and passes them to the classifier, ensuring that background noise does not skew the material prediction.

### Milestone 7: Production Optimization (Adaptive Scene Engine & Bayesian Priors)
- **Objective**: Enhance edge detection accuracy for reflective, small, and cluttered objects.
- **Outcome**: Built an Adaptive Scene Engine that dynamically lowers thresholds for dense clutter or triggers high-sensitivity sweeps for small objects. Integrated a 2D HSV context detector that applies Bayesian priors (Grass, Beach, Street, Indoor) to the output probabilities.

### Milestone 8: Full-Stack Web App & Hugging Face Spaces Deployment
- **Objective**: Build a premium, auditable dashboard for model visualization and deploy it publicly.
- **Outcome**: Developed an HTML/CSS/JS frontend communicating with a Python HTTP server API. Packaged the entire server (with warm preloaded PyTorch weights) inside a Docker container and deployed it to Hugging Face Spaces with LFS tracking.

---

## 3 · Dataset Registries & Statistics

The WasteWise codebase relies on two primary dataset registries to evaluate classification and localization:

### 3.1 Classification Registry: `data/merged_dataset_v5`
A balanced dataset containing **29,639 object crops** representing 7 distinct classes (including `Background` to handle YOLO false alarms):
* **Plastic**: Curated beverage bottles, containers, and wrap.
* **Glass**: Clear, green, and brown bottles/jars.
* **Metal**: Aluminum beverage cans, tin cans, and foil.
* **Paper**: Magazines, cardboard shredding, and receipts.
* **Cardboard**: Shipping boxes and clean packaging.
* **Organic**: Food waste and biodegradable matter.
* **Background**: Texture noise, grass, sand, asphalt, and concrete.

### 3.2 Localization Registry: `external_datasets/super_yolo_dataset`
An annotated object-detection dataset containing **23,929 high-resolution images** and **102,777 bounding boxes** across 6 core classes:

| Class | Bounding Boxes | Support Role |
|---|---:|---|
| Plastic | 35,420 | High density (highly reflective) |
| Glass | 8,912 | Low density (extreme reflection) |
| Metal | 12,402 | Moderate density (metallic texture) |
| Paper | 22,109 | High density (deformable geometry) |
| Cardboard | 11,834 | Moderate density (semi-rigid) |
| Organic | 12,100 | Biodegradable (irregular textures) |
| **Total** | **102,777** | **Annotated Targets** |

---

## 4 · Benchmarks & Results

### 4.1 Classical ML Branch Performance
The classical ML branch was benchmarked on two datasets using the $637$-dimensional handcrafted feature vector.

#### Table 4.1.1: Legacy Lecturer-Facing Split (`merged_dataset_v3`, balanced at 4,000 train crops per class)
*The classical classifiers achieve strong baselines, led by XGBoost:*

| Model | Test Accuracy | F1-Macro | Feature Domain Focus |
|---|---:|---:|---|
| **XGBoost** | **0.6742** | **0.6506** | Balanced spatial + HOG gradients |
| **Random Forest** | 0.6317 | 0.6111 | HOG orientation bins |
| **ExtraTrees** | 0.6312 | 0.6113 | HOG orientation bins |
| **Linear SVM** | 0.5960 | 0.5642 | Scaled intensity percentiles |
| **Logistic Regression** | 0.5864 | 0.5558 | L2-regularized linear boundaries |
| **Decision Tree** | 0.5115 | 0.4883 | High-variance splits |

#### Table 4.1.2: Rerun Split (`super_yolo_dataset` crops, balanced at 4,000 train crops per class)
*Due to imbalanced test support in unconstrained datasets (only 9 glass boxes and 46 organic boxes in test), F1-macro declines:*

| Model | Test Accuracy | F1-Macro | Status |
|---|---:|---:|---|
| **XGBoost** | **0.5408** | **0.3691** | Promoted ML baseline |
| **Random Forest** | 0.5063 | 0.3456 | Baseline |
| **ExtraTrees** | 0.5045 | 0.3414 | Baseline |
| **Linear SVM** | 0.4628 | 0.3159 | Baseline |
| **Logistic Regression** | 0.4494 | 0.3054 | Baseline |
| **Decision Tree** | 0.3750 | 0.2631 | Baseline |

---

### 4.2 PCA Dimensionality Sweep
To verify the impact of representation compression, we performed a controlled PCA sweep on the $637$-dimensional feature space using the Linear SVM and Logistic Regression models.

#### Table 4.2.1: Controlled Feature Space Compression
*Compressing the features from 637 to 128 dimensions keeps 99.90% explained variance and costs only ~2.5% accuracy:*

| Classifier | Components | Cumulative Explained Variance | Test Accuracy | F1-Macro | Accuracy Drop |
|---|---:|---:|---:|---:|---:|
| **Linear SVM** | 637 | 100.00% | 62.43% | 0.6235 | 0.00 pp |
| **Linear SVM** | **128** | **99.90%** | **59.90%** | **0.5947** | **2.52 pp** |
| **Linear SVM** | 64 | 99.78% | 57.22% | 0.5694 | 5.21 pp |
| **Linear SVM** | 32 | 98.92% | 53.04% | 0.5188 | 9.39 pp |
| **Logistic Regression** | 637 | 100.00% | 60.24% | 0.6019 | 0.00 pp |
| **Logistic Regression** | **128** | **99.90%** | **59.71%** | **0.5954** | **0.53 pp** |

---

### 4.3 Deep Learning Localization Sweeps
The YOLOv11 localization module was evaluated on the test split ($300$ images containing $1,152$ bounding boxes). We swept the confidence threshold to find the optimal trade-off between false alarms (FP) and missed items (FN).

#### Table 4.3.1: YOLO Confidence Sweep on Test Split
*A confidence threshold of 0.30 represents the most balanced configuration:*

| YOLO Conf | Precision | Recall | Mean Matched IoU | True Positives (TP) | False Positives (FP) | False Negatives (FN) |
|---|---:|---:|---:|---:|---:|---:|
| **0.25** | 0.6352 | 0.5670 | 0.9012 | 148 | 85 | 113 |
| **0.30** | **0.6999** | **0.5729** | **0.9057** | **660** | **283** | **492** |
| **0.35** | 0.7614 | 0.5134 | 0.9004 | 134 | 42 | 127 |
| **0.40** | 0.8035 | 0.5148 | 0.9050 | 593 | 145 | 559 |
| **Grad-CAM Baseline** | 0.2568 | 0.0728 | 0.7127 | 19 | 55 | 242 |

---

### 4.4 End-to-End Production Verification
The final production server was verified on a highly reflective, challenging target (`plastic.jpg`). The test demonstrates how the ported optimization features prevent false negatives:

```
+-----------------------------------------------------------------------------------------------+
| Raw Input Image  --->  YOLO Detector (conf=0.30)  --->  0 Boxes Found                         |
|                                                          |                                    |
|                                                          v (Adaptive Scene Engine Triggered)  |
|                                                        YOLO Sensitivity Sweep (conf=0.05)     |
|                                                          |                                    |
|                                                          v                                    |
|                                                        3 Bounding Boxes Recovered             |
|                                                          |                                    |
|                                                          v                                    |
|                                                        Physical Size Filter (Width/Height >24)|
|                                                          |                                    |
|                                                          v                                    |
|                                                        CNN Crop Classification                |
|                                                          |                                    |
|                                                          v                                    |
|                                                        Dynamic Alpha Blending (YOLO vs CNN)   |
|                                                          |                                    |
|                                                          v                                    |
|                                                        Bayesian Context Prior (Beach HSV)     |
|                                                          |                                    |
|                                                          v                                    |
|                                                        Final Verified Class: PLASTIC (84% Conf)|
+-----------------------------------------------------------------------------------------------+
```

* **YOLO Proposal Stage**: Initiating detection at `conf = 0.30` returned $0$ bounding boxes due to reflective distortion.
* **Adaptive Scene Engine Step**: Detected $0$ boxes $\rightarrow$ automatically triggered the `SMALL_OBJECT_RECOVERY` mode, lowering the YOLO confidence threshold to `0.05` and adjusting the CNN verification limit to `0.15`.
* **Proposal Recovery**: Recovered **3 candidate bounding boxes**.
* **Size Filtering**: Bounding boxes passed the $24 \times 24$ pixels tininess check.
* **Feature Extraction & Classification**: The ConvNeXt-Tiny classifier ran batch inference on the padded crop images.
* **Dynamic Alpha Blending**: At a low YOLO score ($0.05$), the system assigned an alpha of $0.15$, relying on the CNN's high-resolution texture details.
* **Bayesian Context Prior Fusion**: The environment detector analyzed the HSV cues of the scene (sand/water) and matched the `Beach` context. Non-background probabilities were multiplied by the `Beach` prior $[0.35, 0.25, 0.20, 0.05, 0.05, 0.10]$ and normalized.
* **Final Bottom-Up Output**: The highest-confidence detection achieved **84% Plastic** confidence. The overall image class was correctly aggregated as `Plastic`, whereas a classification-first pipeline returned a weak `Paper` (32% confidence) scene prediction.

---

## 5 · Literature, Books, Magazines, and Computer Vision References

The design, implementation, and optimization of the WasteWise project draw from the following foundational works in computer vision, machine learning, and waste management:

### 5.1 Deep Learning & Object Detection Architectures
* **YOLO (You Only Look Once)**: Redmon, J., Divvala, S., Girshick, R., & Farhadi, A. (2016). *You Only Look Once: Unified, Real-Time Object Detection*. Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 779-788.
  - *Context*: Established the unified single-stage regression model for bounding boxes. YOLOv11 optimizes anchor-free regression and C3k2 block structures, allowing the localizer to run at low latencies on edge devices.
* **ConvNeXt**: Liu, Z., Mao, H., Wu, C.-Y., Feichtenhofer, C., Darrell, T., & Xie, S. (2022). *A ConvNet for the 2020s*. Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 11976-11986.
  - *Context*: Modernized standard ResNets with Depthwise Convolutions and inverted bottlenecks. Used in WasteWise as the Stage 2 material crop classifier.
* **EfficientNet**: Tan, M., & Le, Q. V. (2019). *EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks*. International Conference on Machine Learning (ICML).
  - *Context*: Utilizes compound scaling (width, depth, and resolution). Evaluated in the WasteWise mobile baseline.
* **Deep Residual Learning (ResNet)**: He, K., Zhang, X., Ren, S., & Sun, J. (2016). *Deep Residual Learning for Image Recognition*. Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 770-778.
  - *Context*: Introduced identity shortcut connections to solve the vanishing gradient problem, allowing deep convolutional networks to train effectively.
* **MobileNets**: Howard, A. G., et al. (2017). *MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications*. arXiv preprint arXiv:1704.04861.
  - *Context*: Introduced depthwise separable convolutions to build lightweight deep neural networks for mobile/edge systems.

### 5.2 Handcrafted Feature Extraction & Classical Computer Vision
* **HOG Descriptors**: Dalal, N., & Triggs, B. (2005). *Histograms of Oriented Gradients for Human Detection*. IEEE Computer Society Conference on Computer Vision and Pattern Recognition (CVPR).
  - *Context*: HOG is the core shape and texture descriptor utilized in the $637$-D explainable feature vector ($576$ HOG features per crop).
* **Scale-Invariant Feature Transform (SIFT)**: Lowe, D. G. (2004). *Distinctive Image Features from Scale-Invariant Keypoints*. International Journal of Computer Vision, 60(2), 91-110.
  - *Context*: Pioneer work in handcrafted scale-invariant features, establishing the framework for extracting local invariant keypoints from digital images.
* **Fourier Texture Analysis**: Tuceryan, M., & Jain, A. K. (1998). *Texture Analysis*. *Handbook of Pattern Recognition and Computer Vision*, 2, 207-238.
  - *Context*: Radial FFT energy bins are used in WasteWise to capture frequency-domain differences, helping distinguish organic matter (high frequency) from smooth plastics.
* **Local Binary Patterns (LBP)**: Ojala, T., Pietikainen, M., & Maenpaa, T. (2002). *Multiresolution Gray-scale and Rotation Invariant Texture Classification with Local Binary Patterns*. IEEE Transactions on Pattern Analysis and Machine Intelligence, 24(7), 971-987.
  - *Context*: Provides a robust, computationally simple descriptor for local spatial patterns and contrast.
* **Co-occurrence Texture (GLCM)**: Haralick, R. M., Shanmugam, K., & Dinstein, I. (1973). *Texture Features for Image Classification*. IEEE Transactions on Systems, Man, and Cybernetics, SMC-3(6), 610-621.
  - *Context*: Defined the gray-level co-occurrence matrix to extract statistical features representing image texture.

### 5.3 Waste Classification Research & Datasets
* **TrashNet**: Thung, G., & Yang, M. (2016). *Classification of Trash for Recyclability Status*. CS229 Project Report, Stanford University.
  - *Context*: Established the baseline categories (Glass, Paper, Cardboard, Plastic, Metal, Trash) for automated recycling.
* **TACO Dataset**: Proença, P. F., & Simões, G. (2020). *TACO: Trash Annotations in Context for Litter Detection*. arXiv preprint arXiv:2003.01290.
  - *Context*: An annotated dataset containing waste in unconstrained environments (beaches, streets, parks). Utilized in WasteWise to train the YOLOv11 localizer.
* **Zero Waste Applications**: Adedeji, O., & Wang, Z. (2019). *Intelligent Waste Classification System Using Deep Learning*. Proceedings of the IEEE International Conference on Cognitive Computing.

### 5.4 Bayesian Reasoning & Context Fusion
* **Probabilistic Context Networks**: Pearl, J. (1988). *Probabilistic Reasoning in Intelligent Systems: Networks of Plausible Inference*. Morgan Kaufmann.
  - *Context*: Pearl's probabilistic context formulation is the foundation for fusing environment priors (such as Sand/Water hue context) to scale post-inference posteriors.
* **Visual Context Priming**: Torralba, A. (2003). *Contextual Priming for Object Detection*. International Journal of Computer Vision, 53(2), 169-191.
  - *Context*: Discusses how low-level global scene features provide context that constrains where objects are likely to appear and what they are likely to be.
