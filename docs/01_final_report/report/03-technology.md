# 3. Technology and Tools

The stack is pinned rather than floating. `requirements.txt` records the exact
versions installed in the training environment and deployed to the inference
service, so that local sweeps measure the same inference code that production
runs.
<!-- src: requirements.txt#header comment -->

## 3.1 Training and Modelling

| Component | Version | Role |
|---|---|---|
| Python | 3.11 | Runtime |
| PyTorch | 2.4.1 | Deep learning framework |
| torchvision | 0.19.1 | Faster R-CNN, RetinaNet, FCOS |
| Ultralytics | 8.4.92 | YOLOv8 / YOLO11 / YOLO26 / RT-DETR |
| scikit-learn | ≥1.4.0 | Classical models, PCA, metrics |
| XGBoost | ≥2.0.0 | Gradient-boosted classical baseline |
| NumPy / pandas | ≥1.26 / ≥2.1 | Numerical and tabular processing |
| OpenCV | — | Image decode, resize, perceptual hashing |

<!-- src: requirements.txt -->

*Table 3.1 — Training stack.*

PyTorch is pinned at 2.4.1 deliberately: torch 2.11 forces a new ONNX exporter
via `onnxscript`/`onnx_ir` that conflicts with the `onnx` 1.16.2 and
`ml_dtypes` 0.3.2 versions the TFLite export chain requires. This is a recorded
constraint, not an oversight.
<!-- src: requirements.txt#torch pin comment -->

## 3.2 Export and Deployment

ONNX 1.16.2 with ONNX Runtime 1.18.1 provides the portable inference path, and
the Ultralytics TFLite export chain (`onnx2tf`, `onnx_graphsurgeon`,
`tflite-support`, TensorFlow 2.16.1) supports mobile targets. The web application
is served from Vercel, with the model service deployed alongside; the
front end is a static application communicating with the inference endpoint.
<!-- src: requirements.txt; scripts/deploy_hf_space.py; web/ -->

## 3.3 Compute

Two distinct compute environments were used, and conflating them would misreport
the experiments.

**Local development and deployed-model training.** An RTX 3060 with 12 GB. This
card constrains the training recipe directly: YOLO26n at 640 px with batch 32
exhausts its memory during the pin-memory stage, so batch 16 with four workers is
the stable configuration, and RAM caching is disabled because the 20,000-image
dataset requires roughly 24 GB against 17 GB available. These constraints are
recorded as Failure F7.
<!-- src: docs/01_final_report/FAILURES_AND_FIXES.md#F7 -->

**Rented GPU for the controlled architecture comparison.** The seven-architecture
sweep of §7.4 was provisioned on Vast.ai using an **RTX 4090** on a
`pytorch/pytorch:2.4.0-cuda12.1-cudnn9-runtime` image, with 60 GB of disk to hold
the 6.2 GB packed dataset and seven sets of run artefacts.
<!-- src: scripts/vast/README.md#search offers gpu_name=RTX_4090, create instance --image -->

A 24 GB card was required because Faster R-CNN at batch 16 and 512 px does not
fit in less. `provision.sh` additionally carries a Blackwell (sm_120)
compatibility guard for RTX 5090 instances; that guard is defensive and the sweep
was specified against the 4090.

## 3.4 Development and Reproducibility Tooling

Git provides version control, with dataset builds, audits and evaluations driven
by scripts under `scripts/` rather than notebooks, so that every reported figure
traces to a re-runnable command. Dataset ingestion uses hardlinks where possible,
so images are not silently re-encoded between generations — a property that
matters given that re-encoding is precisely what defeated byte-level
deduplication in §6.4.

Matplotlib and seaborn generate all figures. Evaluation artefacts are written as
JSON alongside human-readable Markdown summaries in `runs/`, which is what makes
the evidence-tracing convention used throughout this report possible.
