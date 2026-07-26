#!/usr/bin/env python
"""WasteWise web server with real local model inference.

Serves the static files in this folder and exposes POST /api/predict.
The API runs the hard-case ConvNeXt classifier and YOLO26n localizer.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time
from email.parser import BytesParser
from email.policy import HTTP as EMAIL_HTTP_POLICY
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any


os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

ROOT = Path(__file__).resolve().parent.parent
WEB_DIR = ROOT / "web"
CLASSIFIER_PATH = ROOT / "models" / "trained" / "efficientnet_classifier" / "best_efficientnet_tuned.h5"
TUNED_CLASSIFIER_PATH = ROOT / "runs" / "dl" / "convnext_ensemble_tuned" / "best_convnext_ensemble_tuned.pth"
TUNED_SCALER_PATH = ROOT / "runs" / "dl" / "convnext_ensemble_tuned" / "handcrafted_scaler.npz"
YOLO_PATH = ROOT / "models" / "trained" / "yolov11_detector" / "best.pt"
LOCALIZER_LABEL = "YOLO26m hard-case localizer"

CLASSIFIER_CLASSES = ["plastic", "glass", "metal", "paper", "cardboard", "organic", "Background"]
# Detector candidate-generation confidence. Lowered 0.10 -> 0.04 (F13): sweeping the deployed
# 6-class detector's conf on the clean held-out field test (runs/audits/detector_conf_sweep_field_6class.json)
# lifts class-agnostic field recall 0.505@0.10 -> 0.557@0.05 -> 0.586@0.04 AND studio clean-val
# recall 0.584 -> 0.637, small-box +11pp - a strict recall gain across BOTH domains from a pure
# threshold change (the F12 class-agnostic retrain was reverted after this showed ~75% of its
# apparent field gain was just this conf drop, at a -10pp studio cost). YOLO_GATE_CONF keeps the
# "confident waste" decision on the box objectness score, so the extra low-conf boxes only
# widen the classifier's candidate pool; they cannot be auto-labeled waste (F5 mechanism).
#
# 2026-07-12: promoted YOLO26n hard-neg fine-tune -> YOLO26s (100-epoch retrain, same 6-class
# hardcase dataset). Re-swept conf/gate/alpha on the new backbone (runs/audits/yolo26s_conf_gate_alpha_sweep.json):
# YOLO_CONF=0.04 still holds (going lower keeps buying recall at a fast-eroding rate: +1.1pp
# field recall per step past 0.04, while candidate volume balloons). YOLO_GATE_CONF raised
# 0.30 -> 0.40: yolo26s is less locally-precise than yolo26n at the same gate threshold
# (field precision 0.762 vs yolo26n's documented 0.812 @ gate=0.30); gate=0.40 restores that
# field-precision guarantee (0.811) at a small recall cost (0.549 -> 0.512), while raw field
# recall at conf=0.04 is still +14.3pp over the old deployed model either way. Alpha cap kept
# at 0.40 (unchanged): yolo26s predicts "plastic" for 84.2% of field boxes (vs yolo26n's 78%),
# so the detector's material vote is equally or more unreliable on field images - no basis to
# trust it more.
#
# 2026-07-15: promoted YOLO26s -> YOLO26m (100-epoch retrain resumed from epoch 61 after a
# Kaggle/Colab quota chain forced a move to local GPU; same 6-class hardcase dataset). Re-swept
# conf/gate/alpha on the new backbone (runs/audits/yolo26m_conf_gate_alpha_sweep.json, rebuilt
# from the same deterministic taco_field_clean_v1 test set + clean_eval studio val used for the
# yolo26s sweep). YOLO_CONF=0.04 still holds - and this time it's a clean win, not just a
# recall/precision trade: field recall 0.709->0.781 (+7.2pp) AND field precision 0.394->0.467
# (+7.3pp) over yolo26s at the identical operating point; small-box field recall 0.586->0.682
# (+9.6pp), the historically hardest number to move. YOLO_GATE_CONF lowered 0.40 -> 0.30:
# yolo26m is MORE locally-precise than yolo26s at the same gate (field precision 0.829 @
# gate=0.30 vs yolo26s's 0.762 there), so gate=0.30 already clears the ~0.81 field-precision
# guarantee the old gate=0.40 was raised to hit - and recovers a lot of gated recall doing it
# (field recall 0.512 @ old gate=0.40 -> 0.631 @ new gate=0.30). Alpha cap kept at 0.40
# (unchanged): yolo26m predicts "plastic" for 84.7% of field boxes (vs yolo26s's 84.2%,
# statistically the same bias) - still no basis to trust the material vote more.
YOLO_CONF = 0.04
YOLO_GATE_CONF = 0.30
# Serve at the training resolution. The model was trained at imgsz=640; running it at 960
# measurably hurts on BOTH eval domains (F9 in docs/01_final_report/FAILURES_AND_FIXES.md):
# clean test mAP50 0.474@640 vs 0.409@960, recall 0.456 vs 0.429; realworld_v2 test mAP50
# 0.499@640 vs 0.475@960 - and 960 costs ~2.25x the CPU inference time on the HF Space.
YOLO_IMG_SIZE = int(os.environ.get("WASTEWISE_YOLO_IMG_SIZE", "640"))
YOLO_IOU = 0.55
YOLO_MAX_DETECTIONS = int(os.environ.get("WASTEWISE_YOLO_MAX_DETECTIONS", "80"))
MAX_CROP_VERIFICATIONS = int(os.environ.get("WASTEWISE_MAX_CROP_VERIFICATIONS", str(YOLO_MAX_DETECTIONS)))
SCENE_TOP_K = 3
CROP_PAD_PX = 10
# Per-detection confidence bar for counting a box as a reliable material vote (headline
# material and the mixed-load bin hint). Not a waste decision - just noise rejection.
MIN_RELIABLE_CONF = 0.50
MAX_UPLOAD_BYTES = 15 * 1024 * 1024
DOMINANT_OVERRIDE_MARGIN = 20
# 2026-07-26 real-world fusion audit: 13 non-dataset litter photos through the full
# pipeline + 8 fusion variants swept on two independent eval sets (rf_garbage_cls test
# stride-2, n=521, plastic/paper-heavy; rf_garbage_cls valid stride-4, n=524,
# organic/glass/cardboard-heavy; archived in runs/audits/realworld_fusion_sweep_2026-07-26.json).
# Findings and decisions:
#   * CONTEXT PRIOR OFF. infer_context mislabels asphalt/benches as "Beach" (the sand HSV
#     mask overlaps warm grey pavement), and the hand-written priors tilt every ambiguous
#     crop toward plastic (organic x0.05-0.15 vs plastic x0.20-0.35). Measured: prior-off
#     costs -1.5pp on the plastic-heavy test split but GAINS +0.2pp on the organic-heavy
#     valid split - the prior's "value" was a plastic subsidy, not scene understanding.
#     Real-world evidence: banana peel -> Plastic 65%; a cardboard-sleeved coffee cup's
#     correct Cardboard-50 crop call flipped to Plastic by the Grass prior alone.
#     infer_context stays for the UI context field; only the prior multiply is disabled.
#   * PLASTIC-VOTE DISCOUNT, disagree-gated. The detector votes plastic on 84.7% of field
#     boxes (see above), so a plastic vote that CONTRADICTS the crop classifier's top
#     material carries ~no information: alpha 0.40 -> 0.15 for exactly those votes.
#     Corroborating plastic votes keep 0.40 (blanket discount measured -2.9pp extra
#     plastic on the test split; the disagree gate recovers +1.2pp of it).
#   * UNSURE-CROP ZOOM RETRY ON. Loose detector boxes dilute crops with background
#     (banana-peel crop: organic 0.324 in the detector box vs 0.575 in a tight crop).
#     When crop top-1 < 0.55, a 22%-per-side center-shrunk view is averaged in.
#   * REJECTED by measurement: sand-mask sat>=50 (didn't fix contexts, -0.4pp), organic
#     prior floor (no effect - the prior tilt dominates), blanket alpha 0.15 (above).
# Net on the combined independent sets (n=1045): macro 0.9692 -> 0.9546, entirely the
# plastic subsidy coming off (plastic 0.965 -> 0.873; every other class flat or up, and
# the organic/glass/cardboard-heavy valid split hits its best value of any variant at
# 0.9810). Real-world: 7 of 12 non-dataset litter photos pass spanning all six classes -
# the old config passed 6 with zero paper and zero organic.
SAND_SAT_MIN = int(os.environ.get("WASTEWISE_SAND_SAT_MIN", "20"))
ALPHA_PLASTIC_VOTE = float(os.environ.get("WASTEWISE_ALPHA_PLASTIC_VOTE", "0.15"))
ALPHA_PLASTIC_MODE = os.environ.get("WASTEWISE_ALPHA_PLASTIC_MODE", "disagree")  # blanket | disagree
UNSURE_ZOOM = os.environ.get("WASTEWISE_UNSURE_ZOOM", "1") == "1"
UNSURE_ZOOM_CONF = float(os.environ.get("WASTEWISE_UNSURE_ZOOM_CONF", "0.55"))
UNSURE_ZOOM_SHRINK = 0.22  # fraction removed from each side of the box
ORGANIC_PRIOR_FLOOR = float(os.environ.get("WASTEWISE_ORGANIC_PRIOR_FLOOR", "0"))
CONTEXT_PRIOR_MODE = os.environ.get("WASTEWISE_CONTEXT_PRIOR", "off")  # on | off
# YOLO26 is end-to-end/NMS-free: the iou= arg to predict() is silently ignored, and at
# conf 0.04 near-duplicate proposals of one object flow through - drawn twice in the UI
# and voting twice in the material aggregation. Class-agnostic greedy dedup fixes both.
BOX_DEDUP_IOU = float(os.environ.get("WASTEWISE_BOX_DEDUP_IOU", "0.55"))  # 0 disables
# Dense-scene detector retry (see comment at the call site in predict_image).
DENSE_RETRY_MIN = int(os.environ.get("WASTEWISE_DENSE_RETRY_MIN", "8"))  # 0 disables
DENSE_RETRY_IMG_SIZE = int(os.environ.get("WASTEWISE_DENSE_RETRY_IMG_SIZE", "960"))
# API-only mode (set in the HF Space Dockerfile): serve /api/* only; the Vercel
# frontend is the one website. Local dev keeps serving web/ as before.
API_ONLY = os.environ.get("WASTEWISE_API_ONLY", "0") == "1"

# web/app.js keeps a fallback-only copy of this table; keep the two in sync.
ROUTES = {
    "plastic": "Recycling",
    "glass": "Recycling",
    "metal": "Recycling",
    "paper": "Recycling",
    "cardboard": "Recycling",
    "organic": "Compost",
    "Background": "Review",
}

_models: dict[str, Any] | None = None
_model_preload_started = False
_models_lock = threading.Lock()
# Ultralytics YOLO.predict() mutates internal predictor state on the shared model object and
# is not safe to call concurrently from multiple threads (ThreadingHTTPServer gives each
# request its own thread). Serializing at the request level costs only queueing delay -
# inference already runs ~1s/image - in exchange for not risking crossed/corrupted results.
_inference_lock = threading.Lock()


def title_class(value: str) -> str:
    return value if value == "Background" else value.capitalize()


def normalize_class_key(value: str | None) -> str | None:
    if not value:
        return None

    clean = value.strip().lower().replace("_", " ").replace("-", " ")
    aliases = {
        "bottle": "plastic",
        "plastic bottle": "plastic",
        "can": "metal",
        "tin": "metal",
        "aluminium": "metal",
        "aluminum": "metal",
        "carton": "cardboard",
        "box": "cardboard",
        "cardboard box": "cardboard",
        "food": "organic",
        "food waste": "organic",
        "trash": None,
        "waste": None,
        "garbage": None,
    }

    if clean in aliases:
        return aliases[clean]

    for class_name in CLASSIFIER_CLASSES:
        class_clean = class_name.lower()
        if clean == class_clean or class_clean in clean:
            return class_name
    return None


def class_index(class_key: str) -> int:
    return CLASSIFIER_CLASSES.index(class_key)


def top_k_classes(probs: Any, k: int = SCENE_TOP_K) -> list[dict[str, Any]]:
    ranked = sorted(
        (
            (index, class_name, float(probs[index]))
            for index, class_name in enumerate(CLASSIFIER_CLASSES)
            if class_name != "Background"
        ),
        key=lambda item: item[2],
        reverse=True,
    )
    return [
        {
            "className": title_class(class_name),
            "key": class_name,
            "confidence": round(confidence * 100),
        }
        for index, class_name, confidence in ranked[:k]
    ]


def load_tuned_classifier() -> dict[str, Any]:
    import numpy as np
    import torch

    scripts_dir = ROOT / "scripts"
    if str(scripts_dir) not in sys.path:
        sys.path.append(str(scripts_dir))
    from custom_feature_extractor import extract_637_features
    from stage2_model import Stage3EnsembleClassifier

    if not TUNED_CLASSIFIER_PATH.exists():
        raise FileNotFoundError(f"Tuned classifier not found: {TUNED_CLASSIFIER_PATH}")
    if not TUNED_SCALER_PATH.exists():
        raise FileNotFoundError(f"Tuned scaler not found: {TUNED_SCALER_PATH}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    classifier = Stage3EnsembleClassifier(num_classes=len(CLASSIFIER_CLASSES))
    state_dict = torch.load(TUNED_CLASSIFIER_PATH, map_location=device, weights_only=True)
    classifier.load_state_dict(state_dict)
    classifier.to(device)
    classifier.eval()

    scaler = np.load(TUNED_SCALER_PATH)
    scale = scaler["scale"].astype("float32")
    scale[scale == 0] = 1.0

    return {
        "classifierType": "convnext_637_tuned",
        "classifier": classifier,
        "classifierPath": TUNED_CLASSIFIER_PATH,
        "classifierLabel": "Hard-case ConvNeXt classifier",
        "device": device,
        "torch": torch,
        "extractFeatures": extract_637_features,
        "scalerMean": scaler["mean"].astype("float32"),
        "scalerScale": scale,
    }


def get_models() -> dict[str, Any]:
    global _models
    if _models is not None:
        return _models

    with _models_lock:
        if _models is not None:
            return _models

        from ultralytics import YOLO

        patch_torchvision_nms()

        if not YOLO_PATH.exists():
            raise FileNotFoundError(f"YOLO model not found: {YOLO_PATH}")

        if TUNED_CLASSIFIER_PATH.exists() and TUNED_SCALER_PATH.exists():
            classifier_bundle = load_tuned_classifier()
        else:
            try:
                import tf_keras as keras
            except ImportError:
                from tensorflow import keras  # type: ignore

            if not CLASSIFIER_PATH.exists():
                raise FileNotFoundError(f"Classifier model not found: {CLASSIFIER_PATH}")
            classifier_bundle = {
                "classifierType": "efficientnet_keras",
                "classifier": keras.models.load_model(str(CLASSIFIER_PATH), compile=False),
                "classifierPath": CLASSIFIER_PATH,
                "classifierLabel": "EfficientNet",
                "keras": keras,
            }

        localizer = YOLO(str(YOLO_PATH))
        _models = {**classifier_bundle, "localizer": localizer}
        return _models


def preload_models_background() -> None:
    global _model_preload_started
    if _model_preload_started or _models is not None:
        return
    _model_preload_started = True

    def worker() -> None:
        try:
            get_models()
            print("[OK] Models preloaded in background.")
        except Exception as exc:  # noqa: BLE001
            print(f"[WARN] Background model preload failed: {exc}", file=sys.stderr)

    threading.Thread(target=worker, name="wastewise-model-preload", daemon=True).start()


def patch_torchvision_nms() -> None:
    """Patch broken Windows torchvision NMS builds with a small torch fallback."""
    try:
        import torch
        import torchvision.ops
    except Exception:
        return

    def nms(boxes: Any, scores: Any, iou_threshold: float) -> Any:
        if boxes.numel() == 0:
            return torch.empty((0,), dtype=torch.long, device=boxes.device)

        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 2]
        y2 = boxes[:, 3]
        areas = (x2 - x1).clamp(min=0) * (y2 - y1).clamp(min=0)
        order = scores.argsort(descending=True)
        keep = []

        while order.numel() > 0:
            current = order[0]
            keep.append(current)
            if order.numel() == 1:
                break

            rest = order[1:]
            xx1 = torch.maximum(x1[current], x1[rest])
            yy1 = torch.maximum(y1[current], y1[rest])
            xx2 = torch.minimum(x2[current], x2[rest])
            yy2 = torch.minimum(y2[current], y2[rest])

            inter_w = (xx2 - xx1).clamp(min=0)
            inter_h = (yy2 - yy1).clamp(min=0)
            inter = inter_w * inter_h
            union = areas[current] + areas[rest] - inter
            iou = inter / union.clamp(min=1e-7)
            order = rest[iou <= iou_threshold]

        return torch.stack(keep).to(dtype=torch.long, device=boxes.device)

    try:
        torchvision.ops.nms(torch.zeros((0, 4)), torch.zeros((0,)), 0.5)
    except Exception:
        torchvision.ops.nms = nms


def infer_context(image_bgr: Any, exclude_boxes: list | None = None) -> str:
    import cv2
    import numpy as np

    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    h, w = image_bgr.shape[:2]

    # Score the background, not the litter item itself - a close-up crop is mostly the
    # detected object, and tan/brown litter (cardboard, banana peel, rust) matches the
    # "sand" hue just as well as an actual beach, which used to tag most photos "Beach"
    # regardless of the real scene.
    bg_mask = np.ones((h, w), dtype=bool)
    for x1, y1, x2, y2 in exclude_boxes or []:
        bg_mask[max(0, int(y1)):min(h, int(y2)), max(0, int(x1)):min(w, int(x2))] = False
    if not bg_mask.any():  # box covers the whole frame - no background left to read
        bg_mask[:] = True
    total = max(1, int(np.sum(bg_mask)))

    green = cv2.inRange(hsv, np.array([35, 40, 40]), np.array([85, 255, 255])) > 0
    sand = cv2.inRange(hsv, np.array([10, SAND_SAT_MIN, 80]), np.array([28, 150, 255])) > 0
    water = cv2.inRange(hsv, np.array([90, 30, 50]), np.array([130, 255, 255])) > 0

    green_pct = float(np.sum(green & bg_mask) / total)
    sand_pct = float(np.sum(sand & bg_mask) / total)
    water_pct = float(np.sum(water & bg_mask) / total)

    if green_pct >= 0.25:
        return "Grass"
    # Water hue (cyan/blue) also matches blue paint, tarps, and bins - a real beach needs a
    # visible sand expanse AND a water band together, not just one blue or tan surface.
    if sand_pct >= 0.30 and water_pct >= 0.12:
        return "Beach"
    if float(np.mean(hsv[:, :, 2][bg_mask])) < 95:
        return "Indoor"
    return "Street"


def scene_empty_cues(image_bgr: Any) -> dict[str, Any]:
    import cv2
    import numpy as np

    resized = cv2.resize(image_bgr, (320, 320), interpolation=cv2.INTER_AREA)
    hsv = cv2.cvtColor(resized, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 60, 160)
    edge_density = float(np.mean(edges > 0))
    value_std = float(np.std(hsv[:, :, 2]))
    saturation_mean = float(np.mean(hsv[:, :, 1]))
    is_empty = edge_density < 0.006 and value_std < 7.5 and saturation_mean < 12.0

    return {
        "isEmpty": is_empty,
        "edgeDensity": round(edge_density, 4),
        "valueStd": round(value_std, 2),
        "saturationMean": round(saturation_mean, 2),
    }


def preprocess_image(image_bgr: Any, keras: Any) -> Any:
    import cv2
    import numpy as np

    resized = cv2.resize(image_bgr, (224, 224), interpolation=cv2.INTER_LINEAR)
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32)
    arr = keras.applications.efficientnet.preprocess_input(rgb)
    return np.expand_dims(arr, axis=0)


def classify_bgr_keras(image_bgr: Any, models: dict[str, Any]) -> Any:
    classifier_input = preprocess_image(image_bgr, models["keras"])
    return models["classifier"].predict(classifier_input, verbose=0)[0]


def classify_bgr_tuned(image_bgr: Any, models: dict[str, Any]) -> Any:
    import cv2
    import numpy as np

    torch = models["torch"]
    device = models["device"]
    resized = cv2.resize(image_bgr, (224, 224), interpolation=cv2.INTER_CUBIC)
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    rgb = (rgb - np.array([0.485, 0.456, 0.406], dtype=np.float32)) / np.array([0.229, 0.224, 0.225], dtype=np.float32)
    image_tensor = torch.from_numpy(np.transpose(rgb, (2, 0, 1))).unsqueeze(0).to(device)

    features = models["extractFeatures"](image_bgr).astype(np.float32)
    features = (features - models["scalerMean"]) / models["scalerScale"]
    feature_tensor = torch.from_numpy(features).unsqueeze(0).to(device)

    with torch.inference_mode():
        logits = models["classifier"](image_tensor, feature_tensor)
        probs = torch.softmax(logits, dim=1).detach().cpu().numpy()[0]
    return probs


def classify_bgr_tuned_batch(images_bgr: list[Any], models: dict[str, Any]) -> Any:
    import cv2
    import numpy as np

    if not images_bgr:
        return np.empty((0, len(CLASSIFIER_CLASSES)), dtype=np.float32)

    torch = models["torch"]
    device = models["device"]
    image_tensors = []
    handcrafted_vectors = []
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    for image_bgr in images_bgr:
        resized = cv2.resize(image_bgr, (224, 224), interpolation=cv2.INTER_CUBIC)
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        rgb = (rgb - mean) / std
        image_tensors.append(np.transpose(rgb, (2, 0, 1)))

        features = models["extractFeatures"](image_bgr).astype(np.float32)
        handcrafted_vectors.append((features - models["scalerMean"]) / models["scalerScale"])

    image_tensor = torch.from_numpy(np.stack(image_tensors)).to(device)
    feature_tensor = torch.from_numpy(np.stack(handcrafted_vectors)).to(device)

    with torch.inference_mode():
        logits = models["classifier"](image_tensor, feature_tensor)
        probs = torch.softmax(logits, dim=1).detach().cpu().numpy()
    return probs


def classify_bgr_batch(images_bgr: list[Any], models: dict[str, Any]) -> Any:
    import numpy as np

    if not images_bgr:
        return np.empty((0, len(CLASSIFIER_CLASSES)), dtype=np.float32)
    if models["classifierType"] == "convnext_637_tuned":
        return classify_bgr_tuned_batch(images_bgr, models)
    return np.stack([classify_bgr_keras(image_bgr, models) for image_bgr in images_bgr])


def classify_bgr(image_bgr: Any, models: dict[str, Any]) -> Any:
    if models["classifierType"] == "convnext_637_tuned":
        return classify_bgr_tuned(image_bgr, models)
    return classify_bgr_keras(image_bgr, models)


def encode_crop_thumb(crop: Any, max_side: int = 120) -> str:
    """Small base64 JPEG thumbnail of a crop, for the GUI workflow stepper."""
    import base64

    import cv2

    h, w = crop.shape[:2]
    scale = max_side / float(max(h, w))
    if scale < 1.0:
        crop = cv2.resize(crop, (max(1, int(w * scale)), max(1, int(h * scale))))
    ok, buf = cv2.imencode(".jpg", crop, [int(cv2.IMWRITE_JPEG_QUALITY), 70])
    if not ok:
        return ""
    return "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode("ascii")


def predict_image(image_bytes: bytes) -> dict[str, Any]:
    import cv2
    import numpy as np

    started = time.perf_counter()
    models = get_models()
    localizer = models["localizer"]

    raw = np.frombuffer(image_bytes, dtype=np.uint8)
    image = cv2.imdecode(raw, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("Could not decode image.")

    height, width = image.shape[:2]
    empty_cues = scene_empty_cues(image)

    # Removed 2026-07-16 (audit): a "recovery" re-predict at conf=0.05 when the primary pass
    # found zero boxes was provably a no-op after F13 lowered YOLO_CONF to 0.04 (a stricter
    # threshold on the same image can only return a subset of nothing), and the adaptive
    # per-mode CNN thresholds it set were shadowed by CLASS_THRESHOLDS covering every
    # material class. If a real recovery pass is wanted, it must run BELOW YOLO_CONF and be
    # re-measured against runs/audits/pipeline_bin_decision_eval_*.json first.
    def run_detector(imgsz: int) -> tuple[list, int, int]:
        result = localizer.predict(
            image,
            conf=YOLO_CONF,
            iou=YOLO_IOU,
            imgsz=imgsz,
            max_det=YOLO_MAX_DETECTIONS,
            verbose=False,
        )[0]
        boxes = list(result.boxes[:YOLO_MAX_DETECTIONS])

        # Sort YOLO proposals by confidence and size
        boxes.sort(
            key=lambda box: (
                float(box.conf[0]) if box.conf is not None else 0.0,
                float((box.xyxy[0][2] - box.xyxy[0][0]) * (box.xyxy[0][3] - box.xyxy[0][1])),
            ),
            reverse=True,
        )

        size_filtered = 0
        records = []
        for raw_box in boxes:
            x1, y1, x2, y2 = [float(v) for v in raw_box.xyxy[0].tolist()]
            cls_idx = int(raw_box.cls[0]) if raw_box.cls is not None else -1
            yolo_label = localizer.names.get(cls_idx, str(cls_idx)) if hasattr(localizer, "names") else str(cls_idx)
            yolo_key = normalize_class_key(str(yolo_label))
            yolo_score = float(raw_box.conf[0]) if raw_box.conf is not None else 0.0

            w_box = x2 - x1
            h_box = y2 - y1

            # Physical Size Filter: reject tiny blurry crops (width or height < 24 px)
            if w_box < 24 or h_box < 24:
                size_filtered += 1
                continue

            ix1 = max(0, int(round(x1)) - CROP_PAD_PX)
            iy1 = max(0, int(round(y1)) - CROP_PAD_PX)
            ix2 = min(width, int(round(x2)) + CROP_PAD_PX)
            iy2 = min(height, int(round(y2)) + CROP_PAD_PX)
            crop = image[iy1:iy2, ix1:ix2]
            if crop.size == 0 or crop.shape[0] < 16 or crop.shape[1] < 16:
                continue

            records.append(
                {
                    "x1": x1,
                    "y1": y1,
                    "x2": x2,
                    "y2": y2,
                    "crop": crop,
                    "yoloLabel": yolo_label,
                    "yoloKey": yolo_key,
                    "yoloScore": yolo_score,
                }
            )
        return records, len(boxes), size_filtered

    crop_records, proposed_box_count, size_filtered_count = run_detector(YOLO_IMG_SIZE)

    # Dense-scene retry: many proposals at 640 means a multi-object scene where most items
    # shrink below detectable size; rerun at high res for real recall (34 -> 204 proposals
    # on the floating-bottles demo). Close-up single-object photos yield 1-6 proposals and
    # never trigger, which matters: blanket 960 shattered close-ups into spurious fragments
    # and dropped real-photo pass rate 7/12 -> 4/12 (sweep_v11 vs v10, 2026-07-27).
    if (
        DENSE_RETRY_MIN > 0
        and len(crop_records) >= DENSE_RETRY_MIN
        and DENSE_RETRY_IMG_SIZE > YOLO_IMG_SIZE
    ):
        dense_records, dense_proposed, dense_filtered = run_detector(DENSE_RETRY_IMG_SIZE)
        if len(dense_records) > len(crop_records):
            crop_records, proposed_box_count, size_filtered_count = (
                dense_records, dense_proposed, dense_filtered,
            )

    context = infer_context(
        image, exclude_boxes=[(r["x1"], r["y1"], r["x2"], r["y2"]) for r in crop_records]
    )

    verified_crop_count = min(MAX_CROP_VERIFICATIONS, len(crop_records))
    if verified_crop_count > 0:
        verified_probs = classify_bgr_batch(
            [record["crop"] for record in crop_records[:verified_crop_count]],
            models,
        )
        if UNSURE_ZOOM:
            retry_idx = [
                i for i in range(verified_crop_count)
                if float(np.max(verified_probs[i])) < UNSURE_ZOOM_CONF
            ]
            if retry_idx:
                zoom_crops = []
                for i in retry_idx:
                    r = crop_records[i]
                    bw = r["x2"] - r["x1"]
                    bh = r["y2"] - r["y1"]
                    sx1 = max(0, int(round(r["x1"] + UNSURE_ZOOM_SHRINK * bw)))
                    sy1 = max(0, int(round(r["y1"] + UNSURE_ZOOM_SHRINK * bh)))
                    sx2 = min(width, int(round(r["x2"] - UNSURE_ZOOM_SHRINK * bw)))
                    sy2 = min(height, int(round(r["y2"] - UNSURE_ZOOM_SHRINK * bh)))
                    crop = image[sy1:sy2, sx1:sx2]
                    if crop.size == 0 or crop.shape[0] < 16 or crop.shape[1] < 16:
                        crop = r["crop"]
                    zoom_crops.append(crop)
                zoom_probs = classify_bgr_batch(zoom_crops, models)
                bg_idx = class_index("Background")
                for j, i in enumerate(retry_idx):
                    # keep the view that better frames a material object (less Background);
                    # averaging instead lets the emptier view's Background veto real objects
                    if float(zoom_probs[j][bg_idx]) < float(verified_probs[i][bg_idx]):
                        verified_probs[i] = zoom_probs[j]

        # Class-agnostic duplicate-box removal, AFTER classification so the survivor of a
        # duplicate pair is the most decisive view (highest top-1 prob), not whichever one
        # YOLO happened to score higher - near-identical crops can classify differently
        # and the higher-YOLO-conf view is not reliably the better-classified one.
        if BOX_DEDUP_IOU > 0 and verified_crop_count > 1:

            def box_iou(a: dict, b: dict) -> float:
                ix = max(0.0, min(a["x2"], b["x2"]) - max(a["x1"], b["x1"]))
                iy = max(0.0, min(a["y2"], b["y2"]) - max(a["y1"], b["y1"]))
                inter = ix * iy
                area_a = max(0.0, a["x2"] - a["x1"]) * max(0.0, a["y2"] - a["y1"])
                area_b = max(0.0, b["x2"] - b["x1"]) * max(0.0, b["y2"] - b["y1"])
                union = area_a + area_b - inter
                return inter / union if union > 0 else 0.0

            order = sorted(
                range(verified_crop_count),
                key=lambda i: float(np.max(verified_probs[i])),
                reverse=True,
            )
            kept_idx: list[int] = []
            for i in order:
                if all(
                    box_iou(crop_records[i], crop_records[j]) <= BOX_DEDUP_IOU
                    for j in kept_idx
                ):
                    kept_idx.append(i)
            kept_idx.sort()  # restore YOLO-confidence ordering for downstream indexing
            # ponytail: unverified tail only exists if MAX_CROP_VERIFICATIONS is lowered
            # below max_det (never in the deployed config); dedup it against kept boxes only.
            tail = [
                r for r in crop_records[verified_crop_count:]
                if all(box_iou(r, crop_records[j]) <= BOX_DEDUP_IOU for j in kept_idx)
            ]
            verified_probs = [verified_probs[i] for i in kept_idx]
            crop_records = [crop_records[i] for i in kept_idx] + tail
            verified_crop_count = len(kept_idx)

        # Bottom-up scene classification aggregation
        scene_probs = np.mean(verified_probs, axis=0)
    else:
        verified_probs = []
        scene_probs = classify_bgr(image, models)

    scene_pred_idx = int(np.argmax(scene_probs))
    scene_pred_key = CLASSIFIER_CLASSES[scene_pred_idx]
    scene_pred_conf = float(scene_probs[scene_pred_idx])
    scene_top = top_k_classes(scene_probs)

    detections = []
    material_counts: dict[str, int] = {}
    material_weights: dict[str, float] = {}

    CONTEXT_PRIORS = {
        "Beach":    [0.35, 0.25, 0.20, 0.05, 0.05, 0.10],
        "Grass":    [0.30, 0.15, 0.15, 0.20, 0.05, 0.15],
        "Indoor":   [0.20, 0.10, 0.15, 0.30, 0.20, 0.05],
        "Street":   [0.30, 0.15, 0.15, 0.20, 0.10, 0.10],
        "default":  [1/6,  1/6,  1/6,  1/6,  1/6,  1/6]
    }

    for box_index, record in enumerate(crop_records):
        x1 = record["x1"]
        y1 = record["y1"]
        x2 = record["x2"]
        y2 = record["y2"]
        yolo_label = record["yoloLabel"]
        yolo_key = record["yoloKey"]
        yolo_score = record["yoloScore"]
        should_verify_crop = box_index < MAX_CROP_VERIFICATIONS

        if should_verify_crop:
            crop_probs = verified_probs[box_index]
            crop_idx = int(np.argmax(crop_probs))
            crop_key = CLASSIFIER_CLASSES[crop_idx]
            crop_conf = float(crop_probs[crop_idx])

            # Dynamic Alpha Blending. Alpha caps at 0.40: the detector's material vote
            # can nudge but NOT override a confident crop-classifier call. The detector is
            # material-reliable on studio but plastic-biased on field (predicts plastic for
            # ~78% of field boxes), so at the old cap 0.70 a confident field "plastic" vote
            # flipped correct glass/metal crops to plastic. Cap 0.40 keeps the 92.9% material
            # classifier as the authority on confident calls; costs -0.7pp studio macro-F1.
            # Re-checked 2026-07-12 after the YOLO26s promotion: yolo26s is equally biased
            # (84.2% field plastic-vote rate) so the cap still applies unchanged. Re-checked
            # again 2026-07-15 after YOLO26m: 84.7% field plastic-vote rate, same story.
            # ponytail: cap 0.40, revisit only with a material-reliable field detector.
            yolo_probs = np.zeros(7)
            if yolo_key in CLASSIFIER_CLASSES:
                yidx = class_index(yolo_key)
                yolo_probs[yidx] = yolo_score

            if yolo_score >= 0.30:
                # A plastic vote from this detector carries little information on field
                # boxes (84.7% plastic-vote base rate); trust it less than other materials.
                # "disagree" mode discounts it only when the crop classifier's top material
                # contradicts it, so corroborating votes keep the full boost.
                plastic_discount = yolo_key == "plastic" and (
                    ALPHA_PLASTIC_MODE == "blanket"
                    or CLASSIFIER_CLASSES[int(np.argmax(crop_probs[:6]))] != "plastic"
                )
                alpha = ALPHA_PLASTIC_VOTE if plastic_discount else 0.40
            else:
                alpha = 0.15

            combined_probs = alpha * yolo_probs + (1.0 - alpha) * crop_probs

            # Apply Multi-Modal Bayesian Context Fusion
            # ponytail: tried damping this prior toward uniform (infer_context is an HSV
            # color heuristic with no ground truth to fit CONTEXT_PRIORS against) - measured
            # -0.5 to -0.6pp material/bin/waste-state accuracy on
            # runs/audits/pipeline_bin_decision_eval_*.json with no offsetting gain, so kept
            # the raw prior. Revisit only with an eval set that labels true scene context.
            if CONTEXT_PRIOR_MODE == "off":
                prior = CONTEXT_PRIORS["default"]
            else:
                prior = CONTEXT_PRIORS.get(context, CONTEXT_PRIORS["default"])
            if ORGANIC_PRIOR_FLOOR > 0:
                floored = list(prior)
                floored[5] = max(floored[5], ORGANIC_PRIOR_FLOOR)  # index 5 = organic
                total_prior = sum(floored)
                prior = [v / total_prior for v in floored]
            cv_probs = combined_probs[:6]
            bg_prob = combined_probs[6]
            
            fused_unnormalized = cv_probs * np.array(prior)
            sum_fused = np.sum(fused_unnormalized)
            if sum_fused > 0:
                fused_probs = fused_unnormalized / sum_fused
            else:
                fused_probs = cv_probs
            
            fused_probs_scaled = fused_probs * (1.0 - bg_prob)
            fused_combined_probs = np.append(fused_probs_scaled, bg_prob)

            label_idx = int(np.argmax(fused_combined_probs))
            label_key = CLASSIFIER_CLASSES[label_idx]
            label_conf = float(fused_combined_probs[label_idx])
            label_source = "yolo-crop-bayes-fusion"
        else:
            alpha = None
            crop_key = yolo_key or scene_pred_key
            crop_conf = yolo_score if yolo_key else scene_pred_conf
            if yolo_key:
                label_key = yolo_key
                label_source = "yolo-fast"
                label_conf = yolo_score
            else:
                label_key = scene_pred_key
                label_source = "scene-fast"
                label_conf = scene_pred_conf

        # Every non-Background box is a material vote, weighted by frame area x confidence.
        # No waste gate any more: the headline material and bin route come straight from
        # these votes (S6 waste-state decision removed 2026-07-18 - never had the data to
        # train it as a real decision layer).
        area_weight = max(1.0, (x2 - x1) * (y2 - y1)) * max(0.04, yolo_score)
        label_title = title_class(label_key)
        if label_key != "Background":
            material_counts[label_title] = material_counts.get(label_title, 0) + 1
            material_weights[label_title] = material_weights.get(label_title, 0.0) + area_weight * max(0.10, label_conf)

        detections.append(
            {
                "x": max(0.0, min(100.0, x1 / width * 100.0)),
                "y": max(0.0, min(100.0, y1 / height * 100.0)),
                "w": max(0.0, min(100.0, (x2 - x1) / width * 100.0)),
                "h": max(0.0, min(100.0, (y2 - y1) / height * 100.0)),
                "label": label_title,
                "confidence": round(label_conf * 100),
                "yoloLabel": title_class(str(yolo_label)),
                "yoloConfidence": round(yolo_score * 100),
                "cropClass": title_class(crop_key),
                "cropConfidence": round(crop_conf * 100),
                "cropThumb": encode_crop_thumb(record["crop"]),
                "alpha": None if alpha is None else round(float(alpha), 2),
                "verified": should_verify_crop,
                "labelSource": label_source,
            }
        )

    if detections:
        detections.sort(key=lambda item: (item["confidence"], item["w"] * item["h"]), reverse=True)
        decision_source = f"YOLO {YOLO_IMG_SIZE}px -> top {MAX_CROP_VERIFICATIONS} crop verify -> bottom-up aggregation"
    else:
        decision_source = "scene classifier fallback; no boxes found"

    scores = {
        title_class(class_name): round(float(scene_probs[index]) * 100)
        for index, class_name in enumerate(CLASSIFIER_CLASSES)
    }

    detected_materials = [
        {
            "className": class_name,
            "count": material_counts[class_name],
            "weight": round(material_weights.get(class_name, 0.0), 2),
        }
        for class_name in sorted(material_counts, key=lambda key: material_weights.get(key, 0.0), reverse=True)
    ]
    # Headline material: the area x confidence material vote (detected_materials) wins by
    # default - a blanket switch to "single most confident detection" measured -0.7pp material
    # accuracy overall (runs/audits/pipeline_bin_decision_eval_dominant_fix.json). But
    # sample-can.jpg shows a real failure mode: a correct 80%-confident box can lose to several
    # wrong boxes at lower confidence just because their area adds up. Only override when a
    # single detection beats the vote winner's own confidence by a wide margin - cheap enough
    # to not fire on ordinary noise, wide enough to catch a clear miss like this one.
    dominant_material = detected_materials[0]["className"] if detected_materials else title_class(scene_pred_key)
    if detections:
        non_background_detections = [item for item in detections if item["label"] != "Background"]
        if non_background_detections:
            best_detection = max(non_background_detections, key=lambda item: (item["confidence"], item["w"] * item["h"]))
            current_conf = max((item["confidence"] for item in detections if item["label"] == dominant_material), default=0)
            if best_detection["label"] != dominant_material and best_detection["confidence"] >= current_conf + DOMINANT_OVERRIDE_MARGIN:
                dominant_material = best_detection["label"]
    dominant_key = dominant_material if dominant_material == "Background" else dominant_material.lower()
    possible_confidence = round(scene_pred_conf * 100)
    possible_source = "scene classifier"
    if detections:
        matching_detections = [item for item in detections if item["label"] == dominant_material]
        candidate_detection = max(
            matching_detections or detections,
            key=lambda item: (item["confidence"], item["w"] * item["h"]),
        )
        possible_confidence = int(candidate_detection["confidence"])
        possible_source = candidate_detection.get("labelSource", "localized detection")

    # Mixed-load bin hint: a real secondary object is often physically smaller in frame than
    # the dominant one, so an area-weighted share test (tried: >=15% of total scene weight)
    # almost never fires - measured 0/18 true positives on the mixed-class eval images,
    # because e.g. a small organic scrap next to a big metal can is ~8% of scene weight even
    # though it's clearly present. Presence uses a flat per-detection confidence bar
    # (MIN_RELIABLE_CONF) instead: any reliably-labeled, non-Background detection counts toward
    # its bin, regardless of how much of the frame it occupies. Purely informational.
    bins_present: set[str] = set()
    for item in detections:
        if item["label"] == "Background" or item["confidence"] < int(MIN_RELIABLE_CONF * 100):
            continue
        bin_name = ROUTES.get(item["label"].lower(), "Review")
        if bin_name != "Review":
            bins_present.add(bin_name)
    mixed_bins = sorted(bins_present)
    # Informational only, deliberately not overriding final_bin: forcing a "Mixed - X + Y"
    # bin string on this signal was tried and measured net-negative (runs/audits/
    # pipeline_bin_decision_eval_final.json vs _final2.json) - it barely improves true-positive
    # catch rate on genuinely mixed frames (1/18) while misrouting several single-material
    # frames whose secondary "material" is just detector/classifier noise. Surface it as
    # metadata (mixedLoadDetected, materialsPresent below) so the UI can hint at it without
    # the core bin decision inheriting that noise.
    mixed_load_detected = len(mixed_bins) > 1

    # Bin route is a plain material -> bin lookup now. "Review" is only the routing default
    # for an unknown/no-object material, not a waste verdict.
    final_bin = ROUTES.get(dominant_key, "Review")
    possible_class_name = dominant_material

    latency_ms = int(round((time.perf_counter() - started) * 1000))

    return {
        "className": dominant_material,
        "confidence": round(scene_pred_conf * 100),
        "topPredictions": scene_top,
        "context": context,
        "latency": latency_ms,
        "mode": f"Localization-First ({models['classifierLabel']} + {LOCALIZER_LABEL})",
        "bin": final_bin,
        "possibleMaterial": {
            "className": possible_class_name,
            "confidence": possible_confidence,
            "source": possible_source,
        },
        "detections": detections,
        "scores": scores,
        "model": {
            "classifier": str(models["classifierPath"].relative_to(ROOT)),
            "classifierType": models["classifierType"],
            "localizer": str(YOLO_PATH.relative_to(ROOT)),
            "localizerType": LOCALIZER_LABEL,
            "yoloConf": YOLO_CONF,
            "yoloIou": YOLO_IOU,
            "yoloImageSize": YOLO_IMG_SIZE,
            "maxDetections": YOLO_MAX_DETECTIONS,
            "maxCropVerifications": MAX_CROP_VERIFICATIONS,
            "decisionSource": decision_source,
            "proposedBoxes": proposed_box_count,
            "sizeFilteredBoxes": size_filtered_count,
            "keptBoxes": len(crop_records),
            "verifiedCrops": verified_crop_count,
            "contextPrior": CONTEXT_PRIORS.get(context, CONTEXT_PRIORS["default"]),
            "fullImageClass": title_class(scene_pred_key),
            "fullImageConfidence": round(scene_pred_conf * 100),
            "sceneTopK": scene_top,
            "emptySceneCues": empty_cues,
            "detectedObjectCount": len(detections),
            "detectedMaterials": detected_materials,
            "mixedLoadDetected": mixed_load_detected,
            "materialsPresent": sorted(material_weights, key=lambda k: material_weights[k], reverse=True),
        },
    }


class WasteWiseHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, directory=str(WEB_DIR), **kwargs)

    def end_headers(self) -> None:
        # Deliberately open CORS: this is a public demo API and the Vercel mirror /
        # any static rehost of the frontend must be able to call it cross-origin.
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET,POST,OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        # No Cache-Control previously meant browsers could apply heuristic freshness to
        # static files (index.html/app.js/assets) and skip revalidation for a while after a
        # deploy - a returning visitor could keep running old JS against new HTML (exactly
        # what happened testing the sample picker: fresh index.html + stale cached app.js
        # silently no-op'd on sample ids the old SAMPLES array didn't have). no-cache forces
        # a conditional GET (If-Modified-Since) every time - still gets 304s for unchanged
        # files, but a real edit is never missed.
        self.send_header("Cache-Control", "no-cache")
        super().end_headers()

    def do_OPTIONS(self) -> None:
        self.send_response(204)
        self.end_headers()

    def do_GET(self) -> None:
        if self.path == "/api/health":
            classifier_path = TUNED_CLASSIFIER_PATH if TUNED_CLASSIFIER_PATH.exists() else CLASSIFIER_PATH
            classifier_type = "convnext_637_tuned" if TUNED_CLASSIFIER_PATH.exists() and TUNED_SCALER_PATH.exists() else "efficientnet_keras"
            self.write_json(
                {
                    "ok": True,
                    "classifier": str(classifier_path.relative_to(ROOT)),
                    "classifierType": classifier_type,
                    "localizer": str(YOLO_PATH.relative_to(ROOT)),
                    "localizerType": LOCALIZER_LABEL,
                    "yoloConf": YOLO_CONF,
                    "yoloGateConf": YOLO_GATE_CONF,
                    "yoloIou": YOLO_IOU,
                    "yoloImageSize": YOLO_IMG_SIZE,
                    "maxDetections": YOLO_MAX_DETECTIONS,
                    "maxCropVerifications": MAX_CROP_VERIFICATIONS,
                    "sceneTopK": SCENE_TOP_K,
                    "minReliableConf": MIN_RELIABLE_CONF,
                }
            )
            return
        if API_ONLY:
            # Backend-only deployment (HF Space): the frontend lives on Vercel, so any
            # non-API GET answers with a service card instead of mirroring the site.
            self.write_json(
                {
                    "service": "WasteWise model API",
                    "frontend": "https://wastewise-fyp.vercel.app",
                    "endpoints": ["/api/health", "POST /api/predict"],
                }
            )
            return
        super().do_GET()

    def do_POST(self) -> None:
        if self.path.split("?", 1)[0] != "/api/predict":
            self.send_error(404, "Unknown endpoint")
            return

        content_length = int(self.headers.get("Content-Length", 0) or 0)
        if content_length > MAX_UPLOAD_BYTES:
            self.write_json({"error": f"Upload too large (max {MAX_UPLOAD_BYTES // (1024 * 1024)}MB)."}, status=413)
            return

        try:
            image_bytes = self.read_multipart_image(content_length)
            if not image_bytes:
                raise ValueError("Empty image upload.")
            if len(image_bytes) > MAX_UPLOAD_BYTES:
                raise ValueError(f"Upload too large (max {MAX_UPLOAD_BYTES // (1024 * 1024)}MB).")
            with _inference_lock:
                result = predict_image(image_bytes)
            self.write_json(result)
        except ValueError as exc:
            self.write_json({"error": str(exc)}, status=400)
        except Exception as exc:  # noqa: BLE001
            print(f"[ERROR] /api/predict failed: {exc!r}", file=sys.stderr)
            self.write_json({"error": "Internal error while processing the image."}, status=500)

    def read_multipart_image(self, content_length: int) -> bytes:
        """Extract the 'image' part from a multipart/form-data body.

        Stdlib email parser instead of the cgi module (deprecated, removed in
        Python 3.13): prefix the raw body with the request's Content-Type header
        so BytesParser sees a well-formed MIME message.
        """
        content_type = self.headers.get("Content-Type", "")
        if "multipart/form-data" not in content_type.lower():
            raise ValueError("Expected multipart/form-data upload.")
        body = self.rfile.read(content_length)
        message = BytesParser(policy=EMAIL_HTTP_POLICY).parsebytes(
            f"Content-Type: {content_type}\r\nMIME-Version: 1.0\r\n\r\n".encode("ascii", "ignore") + body
        )
        for part in message.iter_parts():
            if part.get_param("name", header="content-disposition") == "image":
                return part.get_payload(decode=True) or b""
        raise ValueError("Missing multipart field: image")

    def write_json(self, payload: dict[str, Any], status: int = 200) -> None:
        data = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


def main() -> None:
    parser = argparse.ArgumentParser(description="Serve WasteWise web app with real model API.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=4175)
    parser.add_argument("--warmup", action="store_true", help="Load models before accepting requests.")
    parser.add_argument("--no-preload", action="store_true", help="Disable background model preload.")
    args = parser.parse_args()

    if args.warmup:
        get_models()
    elif not args.no_preload and os.environ.get("WASTEWISE_PRELOAD_MODELS", "1") != "0":
        preload_models_background()

    server = ThreadingHTTPServer((args.host, args.port), WasteWiseHandler)
    print(f"[OK] WasteWise web + model API running at http://{args.host}:{args.port}/")
    print(f"[OK] Health check: http://{args.host}:{args.port}/api/health")
    server.serve_forever()


if __name__ == "__main__":
    main()
