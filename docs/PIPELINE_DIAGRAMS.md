# WasteWise Pipeline Diagrams

These diagrams match the current project positioning: the ML branch stays as the
explainable finalized pipeline, while the DL branch is presented as a
classification-to-localization workflow.

## Overall Project Workflow

```mermaid
flowchart TD
    A["Raw waste image datasets"] --> B["Dataset cleaning and split"]
    B --> C["Object crops and labels"]

    C --> ML1["ML branch"]
    ML1 --> ML2["637 handcrafted features"]
    ML2 --> ML3["Model sweep"]
    ML3 --> ML4["Accuracy, F1, confusion matrix"]
    ML2 --> ML5["PCA compression"]
    ML5 --> ML6["Compact feature evaluation"]
    ML4 --> ML7["Final explainable ML evidence"]
    ML6 --> ML7

    B --> DL1["DL branch"]
    DL1 --> DL2["Stage 1: classification gate"]
    DL2 --> DL3["Stage 2: localization module"]
    DL3 --> DL4["Boxes / heatmaps"]
    DL4 --> DL5["Precision, recall, IoU"]
    DL5 --> DL6["Final localization evidence"]
```

## Current DL Pipeline

```mermaid
flowchart LR
    A["Input image"] --> B["Stage 1: image/classification gate"]
    B --> C["Predicted class / visual evidence"]
    C --> D{"Stage 2 localizer"}
    D --> E["Grad-CAM baseline"]
    D --> F["YOLO localization-only module"]
    E --> G["Heatmap-derived boxes"]
    F --> H["YOLO boxes without final class decision"]
    G --> I["Localization evaluation"]
    H --> I
    I --> J["Precision, recall, matched IoU"]
```

## Classical ML Pipeline

```mermaid
flowchart LR
    A["YOLO-labelled images"] --> B["Crop objects"]
    B --> C["Resize crop to 64x64"]
    C --> D["Extract 637 features"]
    D --> E["Spatial + FFT + color + HOG"]
    E --> F["Train ML models"]
    F --> G["Decision Tree / SVM / RF / ExtraTrees / XGBoost / LogReg"]
    G --> H["Select best model"]
    E --> I["PCA sweep"]
    I --> J["32 / 64 / 128 / 256 / 512 components"]
    H --> K["Report ML evidence"]
    J --> K
```

## Legacy DL Pipeline

```mermaid
flowchart LR
    A["Input image"] --> B["YOLOv11 detects boxes first"]
    B --> C["Crop each detected object"]
    C --> D["EfficientNetB0 classifies crop"]
    D --> E["Soft-voting / threshold logic"]
    E --> F["Accepted annotated output"]
    F --> G["Legacy experiment evidence"]
```

## Final Report Positioning

```mermaid
flowchart TD
    A["Final project story"] --> B["ML: finalized explainable classification"]
    A --> C["DL: redesigned classification-to-localization"]
    B --> D["Report accuracy, F1, PCA, feature importance"]
    C --> E["Report precision, recall, IoU, visual localization evidence"]
    D --> F["WasteWise final evidence"]
    E --> F
```
