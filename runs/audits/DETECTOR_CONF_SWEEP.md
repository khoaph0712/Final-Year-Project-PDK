# Detector confidence sweep (post F4c) - clean val + test, deployed weights

`yolo26n_hardcase_v2_long/weights/best.pt` (the deployed baseline; F4c's tiled
retrain lost, so no new weights). No retraining - just the `conf` operating
point (see F5). Swept at BOTH imgsz 640 (eval convention used elsewhere in this
repo) and imgsz 960 (the actual `web/server.py` `YOLO_IMG_SIZE` default) since
they don't agree well enough to assume one from the other.
`scripts/sweep_detector_conf_clean.py`.

## imgsz 960 (matches production)

| split | conf | P | R | organic R | small-box R |
|---|---:|---:|---:|---:|---:|
| val | 0.10 | 0.759 | 0.603 | 0.532 | 0.417 |
| val | 0.20 | 0.759 | 0.603 | 0.447 | 0.325 |
| val | **0.30 (old)** | 0.731 | 0.624 | 0.383 | 0.263 |
| val | 0.50 | 0.840 | 0.542 | 0.278 | 0.168 |
| test | 0.10 | 0.502 | 0.429 | 0.239 | 0.360 |
| test | 0.20 | 0.502 | 0.429 | 0.174 | 0.306 |
| test | **0.30 (old)** | 0.525 | 0.399 | 0.087 | 0.270 |
| test | 0.50 | 0.589 | 0.305 | 0.022 | 0.214 |

Full grid (0.10/0.15/0.20/0.25/0.30/0.40/0.50) in
`runs/audits/detector_conf_sweep_baseline_v2long_imgsz960.json` (imgsz 640
counterpart alongside it for reference - same direction, smaller magnitude).

## Decision: conf 0.30 -> 0.10

At the real serving resolution, dropping conf from 0.30 to 0.10:
- test precision -2.3pp (0.525 -> 0.502), test recall **+3.0pp** (0.399 -> 0.429)
- **organic recall nearly triples** (0.087 -> 0.239)
- **small-box recall +9pp** (0.270 -> 0.360)
- val precision is *higher* at 0.10 than at 0.30 (0.759 vs 0.731)

This is the opposite of the F4c tiled-training result: no retraining, a real
measured gain on the exact metric the project has been chasing (organic /
small-box recall), for a precision cost within noise on one split and a
*gain* on the other. Deployed in `web/server.py`: `YOLO_CONF` 0.30 -> 0.10.

The "confident waste" decision gate (`estimate_detection_waste_state`, used to
label something "waste" outright vs. route to "review") is deliberately kept
at the old threshold via a new `YOLO_GATE_CONF = 0.30` constant - only the
candidate-box generation pass got more permissive, not the bar for a
no-review-needed answer. Weak new detections in the 0.10-0.30 band still have
to clear the classifier + gate logic; if they don't, they land in "review"
instead of a wrong confident label.
