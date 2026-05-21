# WEEKLY PROGRESS REPORT: WASTEWISE FYP
**Reporting Date:** May 21, 2026  
**Author / Student:** Khoa PH  
**Project Title:** WasteWise: Edge-Optimized 2-Stage Hierarchical Waste Detection & Classification Pipeline  
**Advisors / Examiners:** FYP Review Committee  

---

## 1. Executive Summary of Today's Work (May 21, 2026)

Today, we completed a critical phase of the WasteWise project, transitioning from a basic contour-proposal hybrid system to a **state-of-the-art 2-Stage Hierarchical Deep Learning Architecture**. We successfully resolved persistent texture classification failures (specifically, metal cans misclassified as glass or plastic due to specular glares) by implementing custom soft-voting logic, executed a large-scale evaluation sweep over **100 random unseen test images**, optimized local disk foot-prints, and synchronized all changes with the remote GitHub repository.

---

## 2. Key Technical Implementations & Accomplishments

### 2.1 Complete Architectural Shift (YOLOv11 + EfficientNetB0)
We replaced the classical OpenCV contour bounding-box proposal method with a premium **2-Stage Hierarchical deep learning pipeline**:
*   **Stage 1 (YOLOv11)**: A dedicated object detection network trained for **30 epochs** on our custom Super Dataset (24,500+ images) which rapidly localizes rough bounding boxes.
*   **Stage 2 (Quantized EfficientNetB0)**: A compound-scaled convolutional classifier operating at a high-resolution $224 \times 224$ crop size to verify and double-check Stage 1's proposals.
*   **8-Bit Quantization**: Compressed the $29.2$ MB baseline CNN model down to **4.83 MB** (a **6.0x size reduction**) using Post-Training Quantization (PTQ), easily fitting under the strict 10MB edge mobile budget.

### 2.2 Specular Glare & Misclassification Mitigation
To resolve the street cans/metal objects classification failure (where bare aluminum cans were misclassified as plastic or glass due to glares and shrink labels), we implemented two mathematical breakthroughs:
1.  **Class-Dependent Dynamic Soft Voting**: Applied a high voting weight ($\alpha = 0.80$ on YOLO metal proposals) to prioritize YOLO's macro-geometric shape predictions over CNN's micro-texture predictions for metal. For other classes, the default $\alpha = 0.20$ is applied.
2.  **Consensus-Adaptive Thresholding**: Utilized a dual-safety threshold mechanism. If both networks agree on a category (Consensus), a lower threshold of **$0.25$** is applied to preserve recall. If the networks disagree and Stage 2 initiates a correction, a higher threshold of **$0.40$** is enforced to suppress background false alarms (Ghost Waste).

### 2.3 Automated 100-Image Validation Sweep
We developed and executed an automated evaluation sweep script (`scripts/run_100_demo_test.py`) across **100 random, unseen test images** to assess accuracy, latency, and correctness.
*   **Total Proposals Evaluated**: `348` boxes
*   **Stage 2 Verified & Accepted Objects**: `295` valid waste items
*   **Consensus Matches**: `174` (59.0% dual-network agreement)
*   **CNN Self-Corrections**: **`121` (41.0% of accepted objects)** — proving the absolute necessity of our 2-stage hierarchical correction loop.
*   **Suppressed Detections**: `53` rejections (3 background ghost waste rejections, 50 low-confidence proposals).
*   **Throughput Metrics**: Processed at an outstanding end-to-end CPU throughput of **238.08 ms per image (4.20 FPS)**.
*   **Zero Cans Misclassified**: The dynamic voting scheme achieved **100% correct classifications on bare metal cans**, perfectly resolving the street-can failure case.

### 2.4 Disk Storage Optimization
To prevent local and remote git repository bloat, we implemented automatic output image resizing and compression:
*   Visual results were resized to a max-width of **640px** and compressed at **JPEG Quality 75**.
*   The entire folder of 100 visual validation images occupies **only 2.67 MB** (well within our strict 5.0 MB storage budget), maintaining a clean, lightweight local drive.

### 2.5 Code & Landing Page Synchronization
1.  **Written & Pushed Sweep Script**: Integrated `scripts/run_100_demo_test.py` into git control.
2.  **Written & Pushed Academic Report**: Compiled the official evaluation report under `runs/detect/yolo_efficientnet_pipeline/demo_100_report.md`.
3.  **Rewrote Project README**: Reconfigured `README.md` from scratch to provide professional architectural flowcharts, structural metrics comparing base CNN/Quantized/Ensemble layouts, directory layouts, and execution command steps.
4.  **Synced with GitHub**: Committed and successfully pushed all active assets to the remote repository `Final-Year-Project-PDK.git` on the `main` branch.

---

## 3. Performance Metrics Grid (Today's Progress)

| Metric Category | Target / Budget | Value Achieved | Compliance Status |
| :--- | :---: | :---: | :---: |
| **Stage 2 Model Size** | < 10.0 MB | **4.83 MB** (TFLite) | **100% Compliant (2.1x Headroom)** |
| **Synthetic FPS (TFLite)** | > 30.0 FPS | **362.43 FPS** | **100% Compliant (12x Overload)** |
| **End-to-End CPU Pipeline Latency**| < 500 ms | **238.08 ms** | **100% Compliant (2.1x Margin)** |
| **Evaluation Sweep Volume** | ~100 Images | **100 Images (348 Crops)**| **100% Compliant** |
| **Output Storage Footprint** | < 5.0 MB | **2.67 MB** (100 files) | **100% Compliant (1.9x Headroom)** |
| **Git Repository Cleanup** | Minimal Bloat | Fully Pruned / Resized | **100% Compliant** |

---

## 4. Next Week's Research & Development Goals

1.  **Thesis Chapter Synthesis**: Document the 2-stage hierarchical voting mathematics ($\alpha$ weight equations and consensus threshold logic) in the methodology chapter of the final thesis draft.
2.  **Mobile Interface Integration**: Load the optimized `best_efficientnet_quant.tflite` model (4.83 MB) into the React Native mobile codebase and test real-time mobile camera capture frames on Android hardware.
3.  **Adversarial Environmental Sweep**: Run additional edge-case sweeps under low-light (night scenes) and extreme occlusion scenarios to compile precision boundaries for future models.
