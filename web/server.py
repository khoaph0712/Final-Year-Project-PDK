#!/usr/bin/env python
"""WasteWise web server with real local model inference.

Serves the static files in this folder and exposes POST /api/predict.
The API runs the hard-case ConvNeXt classifier and YOLO26n localizer.
"""

from __future__ import annotations

import argparse
import cgi
import json
import os
import sys
import threading
import time
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
YOLO_RECOVERY_CONF = 0.10
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
WASTE_GATE_CONF = 0.55
WASTE_REVIEW_CONF = 0.50
SCENE_REVIEW_CONF = 0.55
WEAK_REVIEW_MATERIAL_CONF = 0.35

ROUTES = {
    "plastic": "Recycling",
    "glass": "Recycling",
    "metal": "Recycling",
    "paper": "Recycling",
    "cardboard": "Recycling",
    "organic": "Compost",
    "Background": "Review",
}

WASTE_STATE_LABELS = {
    "waste": "Waste",
    "not_waste": "Not waste",
    "review": "Review",
}

_models: dict[str, Any] | None = None
_model_preload_started = False
_models_lock = threading.Lock()


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


def choose_detection_label(
    crop_key: str,
    crop_conf: float,
    yolo_key: str | None,
    yolo_score: float,
    scene_top_keys: set[str],
    scene_probs: Any,
) -> tuple[str, str]:
    if crop_key == "Background" and yolo_key:
        return yolo_key, "yolo-background-crop"

    if crop_conf >= 0.72:
        return crop_key, "crop-classifier"

    if yolo_key and yolo_score >= 0.45:
        return yolo_key, "yolo-localizer"

    if yolo_key and yolo_key in scene_top_keys and crop_conf < 0.68:
        return yolo_key, "scene-guided-yolo"

    if crop_key in scene_top_keys or crop_conf >= 0.50:
        return crop_key, "scene-guided-crop"

    if yolo_key:
        return yolo_key, "low-conf-yolo"

    scene_best = max(scene_top_keys, key=lambda key: float(scene_probs[class_index(key)]))
    return scene_best, "scene-fallback"


def detection_confidence(label_key: str, crop_probs: Any, yolo_key: str | None, yolo_score: float, scene_probs: Any) -> float:
    label_idx = class_index(label_key)
    crop_label_conf = float(crop_probs[label_idx])
    scene_label_conf = float(scene_probs[label_idx])
    yolo_label_conf = yolo_score if yolo_key == label_key else 0.0

    if yolo_label_conf:
        return max(crop_label_conf, min(0.95, 0.55 * yolo_label_conf + 0.30 * crop_label_conf + 0.15 * scene_label_conf))
    return max(crop_label_conf, min(0.90, 0.70 * crop_label_conf + 0.30 * scene_label_conf))


def estimate_detection_waste_state(
    label_key: str,
    label_conf: float,
    crop_key: str,
    crop_conf: float,
    yolo_key: str | None,
    yolo_score: float,
    scene_is_empty: bool,
) -> tuple[str, str]:
    """Conservative waste-state gate until a trained state model exists."""
    if scene_is_empty:
        return "not_waste", "empty or near-uniform scene"

    if label_key == "Background":
        if crop_conf >= 0.70 and not yolo_key:
            return "not_waste", "background-dominant crop"
        return "review", "background or unknown object"

    if label_conf >= WASTE_GATE_CONF and yolo_score >= YOLO_GATE_CONF:
        return "waste", "localized material candidate"

    if yolo_key == label_key and yolo_score >= 0.45:
        return "waste", "strong localizer agreement"

    if crop_key == "Background" or label_conf < WASTE_REVIEW_CONF:
        return "review", "weak waste-state evidence"

    return "review", "material known, disposal state unknown"


def choose_final_decision(
    detections: list[dict[str, Any]],
    scene_pred_key: str,
    scene_pred_conf: float,
    scene_is_empty: bool,
) -> tuple[str, str]:
    if scene_is_empty:
        return "not_waste", "empty or near-uniform scene"

    if detections:
        if any(item["wasteState"] == "waste" for item in detections):
            return "waste", "at least one localized object is gated as waste"
        if all(item["wasteState"] == "not_waste" for item in detections):
            return "not_waste", "localized objects look non-disposable"
        return "review", "objects found but disposal state is uncertain"

    if scene_pred_key == "Background" and scene_pred_conf >= 0.70:
        return "not_waste", "no trash-like object localized"
    return "review", "no object localized with enough evidence"


def load_tuned_classifier() -> dict[str, Any]:
    import numpy as np
    import torch
    import torch.nn as nn
    import torchvision.models as models

    feature_dir = ROOT / "scripts" / "archive"
    if str(feature_dir) not in sys.path:
        sys.path.append(str(feature_dir))
    from custom_feature_extractor import extract_637_features

    class ConvNeXtFeatureExtractor(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.backbone = models.convnext_tiny(weights=None)
            self.backbone.classifier = nn.Identity()

        def forward(self, x: Any) -> Any:
            features = self.backbone(x)
            return torch.flatten(features, 1)

    class Stage3EnsembleClassifier(nn.Module):
        def __init__(self, num_classes: int = 7, dropout_rate: float = 0.3) -> None:
            super().__init__()
            self.convnext_extractor = ConvNeXtFeatureExtractor()
            self.classifier = nn.Sequential(
                nn.Linear(768 + 637, 256),
                nn.BatchNorm1d(256),
                nn.ReLU(),
                nn.Dropout(p=dropout_rate),
                nn.Linear(256, 64),
                nn.BatchNorm1d(64),
                nn.ReLU(),
                nn.Dropout(p=0.2),
                nn.Linear(64, num_classes),
            )

        def forward(self, image_tensor: Any, handcrafted_features_tensor: Any) -> Any:
            deep_features = self.convnext_extractor(image_tensor)
            fused_features = torch.cat((deep_features, handcrafted_features_tensor), dim=1)
            return self.classifier(fused_features)

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


def infer_context(image_bgr: Any) -> str:
    import cv2
    import numpy as np

    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    total = max(1, image_bgr.shape[0] * image_bgr.shape[1])

    green = cv2.inRange(hsv, np.array([35, 40, 40]), np.array([85, 255, 255]))
    sand = cv2.inRange(hsv, np.array([10, 20, 80]), np.array([28, 150, 255]))
    water = cv2.inRange(hsv, np.array([90, 30, 50]), np.array([130, 255, 255]))

    green_pct = float(np.sum(green > 0) / total)
    beach_pct = float((np.sum(sand > 0) + np.sum(water > 0)) / total)

    if green_pct >= 0.25:
        return "Grass"
    if beach_pct >= 0.20:
        return "Beach"
    if float(np.mean(hsv[:, :, 2])) < 95:
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
    context = infer_context(image)

    # 1. Adaptive Scene Engine & YOLO localizer run first
    adaptive_cnn_conf = WASTE_REVIEW_CONF  # Default 0.50
    adaptive_mode = "STANDARD"

    yolo_result = localizer.predict(
        image,
        conf=YOLO_CONF,
        iou=YOLO_IOU,
        imgsz=YOLO_IMG_SIZE,
        max_det=YOLO_MAX_DETECTIONS,
        verbose=False,
    )[0]
    yolo_conf_used = YOLO_CONF

    boxes = list(yolo_result.boxes[:YOLO_MAX_DETECTIONS])

    # Scenario 1: Extreme Small Object / Low Recall recovery sensitivity sweep
    if len(boxes) == 0:
        yolo_result = localizer.predict(
            image,
            conf=0.05,
            iou=YOLO_IOU,
            imgsz=YOLO_IMG_SIZE,
            max_det=YOLO_MAX_DETECTIONS,
            verbose=False,
        )[0]
        yolo_conf_used = 0.05
        boxes = list(yolo_result.boxes[:YOLO_MAX_DETECTIONS])
        if len(boxes) > 0:
            adaptive_cnn_conf = 0.15
            adaptive_mode = "SMALL_OBJECT_RECOVERY"

    # Scenario 2: Dense Clutter & Overlap Mitigation
    elif len(boxes) >= 8:
        adaptive_cnn_conf = 0.20
        adaptive_mode = "DENSE_CLUTTER_MITIGATION"

    # Sort YOLO proposals by confidence and size
    boxes.sort(
        key=lambda box: (
            float(box.conf[0]) if box.conf is not None else 0.0,
            float((box.xyxy[0][2] - box.xyxy[0][0]) * (box.xyxy[0][3] - box.xyxy[0][1])),
        ),
        reverse=True,
    )

    proposed_box_count = len(boxes)
    size_filtered_count = 0

    crop_records = []
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
            size_filtered_count += 1
            continue

        ix1 = max(0, int(round(x1)) - CROP_PAD_PX)
        iy1 = max(0, int(round(y1)) - CROP_PAD_PX)
        ix2 = min(width, int(round(x2)) + CROP_PAD_PX)
        iy2 = min(height, int(round(y2)) + CROP_PAD_PX)
        crop = image[iy1:iy2, ix1:ix2]
        if crop.size == 0 or crop.shape[0] < 16 or crop.shape[1] < 16:
            continue

        crop_records.append(
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

    verified_crop_count = min(MAX_CROP_VERIFICATIONS, len(crop_records))
    if verified_crop_count > 0:
        verified_probs = classify_bgr_batch(
            [record["crop"] for record in crop_records[:verified_crop_count]],
            models,
        )
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

    CLASS_THRESHOLDS = {
        "plastic": 0.25,
        "glass": 0.30,
        "metal": 0.18,
        "paper": 0.20,
        "cardboard": 0.18,
        "organic": 0.22
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
                alpha = 0.40
            else:
                alpha = 0.15

            combined_probs = alpha * yolo_probs + (1.0 - alpha) * crop_probs

            # Apply Multi-Modal Bayesian Context Fusion
            prior = CONTEXT_PRIORS.get(context, CONTEXT_PRIORS["default"])
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

        waste_state, waste_reason = estimate_detection_waste_state(
            label_key,
            label_conf,
            crop_key,
            crop_conf,
            yolo_key,
            yolo_score,
            bool(empty_cues["isEmpty"]),
        )

        # Apply Class-Specific Threshold Rejection
        current_threshold = CLASS_THRESHOLDS.get(label_key, adaptive_cnn_conf)
        if label_key == "Background":
            if waste_state == "waste":
                waste_state = "review"
                waste_reason = "background-dominant crop"
        elif label_conf < current_threshold:
            waste_state = "review"
            waste_reason = f"low confidence ({label_conf*100:.1f}% < {current_threshold*100:.0f}%)"

        area_weight = max(1.0, (x2 - x1) * (y2 - y1)) * max(0.04, yolo_score)
        label_title = title_class(label_key)
        if waste_state == "waste":
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
                "wasteState": waste_state,
                "wasteStateLabel": WASTE_STATE_LABELS[waste_state],
                "wasteReason": waste_reason,
                "needsReview": waste_state == "review",
            }
        )

    if detections:
        detections.sort(key=lambda item: (item["confidence"], item["w"] * item["h"]), reverse=True)
        decision_source = f"YOLO {YOLO_IMG_SIZE}px ({adaptive_mode}) -> top {MAX_CROP_VERIFICATIONS} crop verify -> bottom-up aggregation"
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
    final_state, final_reason = choose_final_decision(
        detections,
        scene_pred_key,
        scene_pred_conf,
        bool(empty_cues["isEmpty"]),
    )
    dominant_material = detected_materials[0]["className"] if detected_materials else title_class(scene_pred_key)
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

    strong_waste_detections = [
        item
        for item in detections
        if item["wasteState"] == "waste" and int(item["confidence"]) >= int(WASTE_REVIEW_CONF * 100)
    ]
    if final_state == "waste" and not strong_waste_detections:
        final_state = "review"
        final_reason = "localized objects exist, but material confidence is below review threshold"
    if final_state != "not_waste" and scene_pred_key == "Background" and not strong_waste_detections:
        final_state = "review"
        final_reason = "scene classifier sees background or unknown object; manual review required"
    if final_state != "not_waste" and scene_pred_conf < SCENE_REVIEW_CONF and not strong_waste_detections:
        final_state = "review"
        final_reason = "low classifier confidence; possible material is not reliable enough"

    final_bin = ROUTES.get(dominant_key, "Review") if final_state == "waste" else ("Do not bin" if final_state == "not_waste" else "Review")
    possible_class_name = dominant_material
    if final_state == "review" and possible_confidence < int(WEAK_REVIEW_MATERIAL_CONF * 100):
        possible_class_name = "Unclear"
        possible_source = "weak localizer/classifier evidence"

    latency_ms = int(round((time.perf_counter() - started) * 1000))
    needs_review = final_state == "review" or any(item["needsReview"] for item in detections)

    return {
        "className": dominant_material,
        "confidence": round(scene_pred_conf * 100),
        "topPredictions": scene_top,
        "context": context,
        "latency": latency_ms,
        "mode": f"Localization-First ({models['classifierLabel']} + {LOCALIZER_LABEL})",
        "bin": final_bin,
        "wasteState": final_state,
        "wasteStateLabel": WASTE_STATE_LABELS[final_state],
        "wasteDecision": final_reason,
        "needsReview": needs_review,
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
            "yoloConf": yolo_conf_used,
            "yoloIou": YOLO_IOU,
            "yoloImageSize": YOLO_IMG_SIZE,
            "maxDetections": YOLO_MAX_DETECTIONS,
            "maxCropVerifications": MAX_CROP_VERIFICATIONS,
            "decisionSource": decision_source,
            "adaptiveMode": adaptive_mode,
            "adaptiveCnnConf": adaptive_cnn_conf,
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
        },
    }


class WasteWiseHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, directory=str(WEB_DIR), **kwargs)

    def end_headers(self) -> None:
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET,POST,OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
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
                    "recoveryYoloConf": YOLO_RECOVERY_CONF,
                    "yoloIou": YOLO_IOU,
                    "yoloImageSize": YOLO_IMG_SIZE,
                    "maxDetections": YOLO_MAX_DETECTIONS,
                    "maxCropVerifications": MAX_CROP_VERIFICATIONS,
                    "sceneTopK": SCENE_TOP_K,
                    "wasteGateConf": WASTE_GATE_CONF,
                    "wasteReviewConf": WASTE_REVIEW_CONF,
                    "sceneReviewConf": SCENE_REVIEW_CONF,
                    "weakReviewMaterialConf": WEAK_REVIEW_MATERIAL_CONF,
                }
            )
            return
        super().do_GET()

    def do_POST(self) -> None:
        if self.path.split("?", 1)[0] != "/api/predict":
            self.send_error(404, "Unknown endpoint")
            return

        try:
            form = cgi.FieldStorage(
                fp=self.rfile,
                headers=self.headers,
                environ={
                    "REQUEST_METHOD": "POST",
                    "CONTENT_TYPE": self.headers.get("Content-Type", ""),
                },
            )
            if "image" not in form:
                raise ValueError("Missing multipart field: image")
            item = form["image"]
            image_bytes = item.file.read()
            if not image_bytes:
                raise ValueError("Empty image upload.")
            self.write_json(predict_image(image_bytes))
        except Exception as exc:  # noqa: BLE001
            self.write_json({"error": str(exc)}, status=500)

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
