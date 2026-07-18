# WasteWise Web App

This is the standalone web app for the WasteWise FYP.

The recommended local server runs the real local model pipeline:

- Tuned PyTorch ConvNeXt + 637-feature classifier: `runs/dl/convnext_ensemble_tuned/best_convnext_ensemble_tuned.pth`
- Handcrafted feature scaler: `runs/dl/convnext_ensemble_tuned/handcrafted_scaler.npz`
- YOLO26m hard-case localizer: `models/trained/yolov11_detector/best.pt` (directory
  name is historical; the checkpoint is the promoted YOLO26m detector).
- Crop verification: every YOLO box is classified again and fused with the YOLO
  material vote (alpha-capped) and a scene context prior.
- Bin routing: detections are aggregated (area x confidence) into a dominant material, which is routed to its bin by a direct material->bin lookup. (The earlier S6 waste-state gate that marked each object `waste`/`not_waste`/`review` was removed 2026-07-18 - there was no dataset to train it as a real decision, and it added complexity for no measured gain.)
- Localization settings: `conf=0.04` (candidate generation), objectness gate `0.30`,
  `imgsz=640`, `max_det=80`. The constants at the top of `web/server.py` are the
  source of truth - each carries the audit trail for why it holds its value.
- Result display: `/api/predict` returns a top-level `topPredictions` array for the top three material classes, and the page renders those beside all seven class score bars and detection boxes.

The API now prefers the tuned PyTorch classifier when both the `.pth` model and scaler are present. The older Keras classifier path remains only as a fallback. The app reports the material and its bin route; it does not judge whether an item should actually be discarded. A trained waste-state decision (`waste` / `not_waste` / `review`) was scoped but removed 2026-07-18 for lack of a dataset to train it - a future system could add a dedicated state head if that data is collected.

## Run Locally

From the repository root:

```powershell
.\.venv311\Scripts\python.exe web\server.py --port 4178
```

Then open:

```text
http://localhost:4178
```

## Deploy

The UI is plain HTML, CSS, and JavaScript. The real-model endpoint is a local Python API, so static hosts can only run the browser fallback unless you deploy an equivalent API.

Recommended static publish directory:

```text
web
```

## Scope

- Uses real user uploads as the scanner input.
- Uses the local model API for uploads when `web/server.py` is running.
- Falls back to the browser-side adapter only when the model API is unavailable.
- Saves recent scan results in local browser History for review.

