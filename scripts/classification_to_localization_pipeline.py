#!/usr/bin/env python
"""Classification-first localization pipeline.

Stage 1: run an image-level classifier on the full image.
Stage 2: convert the classifier's Grad-CAM evidence into localization boxes.

This intentionally reverses the old YOLO-first flow. YOLO labels are used only
as ground truth for localization evaluation.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
from dataclasses import dataclass
from pathlib import Path

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import cv2
import numpy as np
import yaml

try:
    import tf_keras as keras
except ImportError:
    from tensorflow import keras

import tensorflow as tf


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATA = ROOT / "external_datasets" / "super_yolo_dataset" / "data.yaml"
DEFAULT_MODEL = ROOT / "models" / "trained" / "efficientnet_classifier" / "best_efficientnet_tuned.h5"
DEFAULT_YOLO_WEIGHTS = ROOT / "models" / "trained" / "yolov11_detector" / "best.pt"
DEFAULT_OUT = ROOT / "runs" / "dl" / "localization_rework" / "default_run"
CLASSIFIER_CLASSES = ["plastic", "glass", "metal", "paper", "cardboard", "organic", "Background"]


@dataclass
class Box:
    x1: int
    y1: int
    x2: int
    y2: int
    cls: int | None = None
    score: float = 1.0


def load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve_split_images(data_yaml: Path, split: str) -> tuple[Path, list[str]]:
    cfg = load_yaml(data_yaml)
    dataset_root = Path(cfg.get("path", data_yaml.parent))
    if not dataset_root.exists():
        dataset_root = data_yaml.parent

    split_value = cfg.get(split)
    if split_value is None:
        raise ValueError(f"Split '{split}' not found in {data_yaml}")

    images_dir = Path(split_value)
    if not images_dir.is_absolute():
        images_dir = dataset_root / images_dir
    if not images_dir.exists():
        raise FileNotFoundError(f"Images directory not found: {images_dir}")

    names = cfg.get("names", [])
    return images_dir, list(names)


def primary_label_id(image_path: Path) -> int:
    label_path = label_path_for_image(image_path)
    if not label_path.exists():
        return -1
    for raw in label_path.read_text(encoding="utf-8").splitlines():
        parts = raw.strip().split()
        if parts:
            return int(float(parts[0]))
    return -1


def image_paths(images_dir: Path, max_images: int | None, sample_mode: str, seed: int) -> list[Path]:
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    paths = sorted(p for p in images_dir.iterdir() if p.suffix.lower() in exts)
    rng = random.Random(seed)

    if sample_mode == "random":
        rng.shuffle(paths)
        return paths[:max_images] if max_images is not None else paths

    if sample_mode == "stratified":
        groups: dict[int, list[Path]] = {}
        for path in paths:
            groups.setdefault(primary_label_id(path), []).append(path)
        for group_paths in groups.values():
            rng.shuffle(group_paths)

        selected: list[Path] = []
        class_ids = sorted(groups)
        while any(groups.values()) and (max_images is None or len(selected) < max_images):
            for class_id in class_ids:
                if groups[class_id]:
                    selected.append(groups[class_id].pop())
                    if max_images is not None and len(selected) >= max_images:
                        break
        return selected

    if max_images is not None:
        paths = paths[:max_images]
    return paths


def label_path_for_image(image_path: Path) -> Path:
    return image_path.parent.parent / "labels" / f"{image_path.stem}.txt"


def read_yolo_boxes(label_path: Path, width: int, height: int) -> list[Box]:
    boxes: list[Box] = []
    if not label_path.exists():
        return boxes
    for raw in label_path.read_text(encoding="utf-8").splitlines():
        parts = raw.strip().split()
        if len(parts) < 5:
            continue
        cls = int(float(parts[0]))
        cx, cy, bw, bh = [float(v) for v in parts[1:5]]
        x1 = int(round((cx - bw / 2.0) * width))
        y1 = int(round((cy - bh / 2.0) * height))
        x2 = int(round((cx + bw / 2.0) * width))
        y2 = int(round((cy + bh / 2.0) * height))
        boxes.append(
            Box(
                x1=max(0, min(width - 1, x1)),
                y1=max(0, min(height - 1, y1)),
                x2=max(0, min(width - 1, x2)),
                y2=max(0, min(height - 1, y2)),
                cls=cls,
            )
        )
    return boxes


def find_nested_backbone(model: keras.Model) -> keras.Model:
    for layer in model.layers:
        if isinstance(layer, keras.Model) and len(getattr(layer, "layers", [])) > 10:
            return layer
    raise ValueError("Could not find nested CNN backbone inside classifier model.")


def find_last_feature_layer(backbone: keras.Model):
    preferred = ["top_activation", "top_conv", "out_relu"]
    for name in preferred:
        try:
            return backbone.get_layer(name)
        except ValueError:
            pass

    for layer in reversed(backbone.layers):
        shape = getattr(layer, "output_shape", None)
        if shape is not None and len(shape) == 4:
            return layer
        out = getattr(layer, "output", None)
        if out is not None and len(out.shape) == 4:
            return layer
    raise ValueError("Could not find a 4D feature layer for Grad-CAM.")


def build_grad_model(model: keras.Model) -> tuple[keras.Model, str]:
    backbone = find_nested_backbone(model)
    feature_layer = find_last_feature_layer(backbone)

    x = backbone.output
    base_index = model.layers.index(backbone)
    for layer in model.layers[base_index + 1 :]:
        x = layer(x)

    grad_model = keras.models.Model(backbone.inputs, [feature_layer.output, x])
    return grad_model, feature_layer.name


def preprocess_image(image_bgr: np.ndarray, target_size: tuple[int, int]) -> np.ndarray:
    resized = cv2.resize(image_bgr, target_size, interpolation=cv2.INTER_LINEAR)
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32)
    arr = keras.applications.efficientnet.preprocess_input(rgb)
    return np.expand_dims(arr, axis=0)


def compute_gradcam(grad_model: keras.Model, image_input: np.ndarray, class_idx: int) -> tuple[np.ndarray, np.ndarray]:
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(image_input)
        loss = predictions[:, class_idx]

    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0.0)
    max_val = tf.reduce_max(heatmap)
    heatmap = heatmap / tf.maximum(max_val, 1e-10)
    return heatmap.numpy(), predictions.numpy()[0]


def boxes_from_heatmap(
    heatmap: np.ndarray,
    width: int,
    height: int,
    threshold: float,
    min_area_ratio: float,
    max_boxes: int,
    score: float,
    cls: int,
) -> list[Box]:
    resized = cv2.resize(heatmap, (width, height), interpolation=cv2.INTER_CUBIC)
    mask = (resized >= threshold).astype(np.uint8) * 255
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    min_area = max(8, int(width * height * min_area_ratio))
    candidates: list[tuple[int, Box]] = []
    for label_id in range(1, num_labels):
        x, y, w, h, area = stats[label_id]
        if area < min_area:
            continue
        candidates.append((int(area), Box(x, y, x + w, y + h, cls=cls, score=score)))

    candidates.sort(key=lambda item: item[0], reverse=True)
    return [box for _, box in candidates[:max_boxes]]


def boxes_from_yolo(yolo_model, image: np.ndarray, conf: float) -> list[Box]:
    result = yolo_model.predict(image, conf=conf, verbose=False)[0]
    pred_boxes: list[Box] = []
    for raw_box in result.boxes:
        x1, y1, x2, y2 = [int(round(v)) for v in raw_box.xyxy[0].tolist()]
        cls = int(raw_box.cls[0]) if raw_box.cls is not None else None
        score = float(raw_box.conf[0]) if raw_box.conf is not None else 1.0
        pred_boxes.append(Box(x1, y1, x2, y2, cls=cls, score=score))
    return pred_boxes


def heatmap_from_boxes(width: int, height: int, boxes: list[Box]) -> np.ndarray:
    heatmap = np.zeros((height, width), dtype=np.float32)
    for box in boxes:
        x1 = max(0, min(width - 1, box.x1))
        y1 = max(0, min(height - 1, box.y1))
        x2 = max(0, min(width, box.x2))
        y2 = max(0, min(height, box.y2))
        if x2 <= x1 or y2 <= y1:
            continue
        heatmap[y1:y2, x1:x2] = np.maximum(heatmap[y1:y2, x1:x2], box.score)
    max_val = float(np.max(heatmap))
    return heatmap / max_val if max_val > 0 else heatmap


def iou(a: Box, b: Box) -> float:
    ix1 = max(a.x1, b.x1)
    iy1 = max(a.y1, b.y1)
    ix2 = min(a.x2, b.x2)
    iy2 = min(a.y2, b.y2)
    iw = max(0, ix2 - ix1)
    ih = max(0, iy2 - iy1)
    inter = iw * ih
    area_a = max(0, a.x2 - a.x1) * max(0, a.y2 - a.y1)
    area_b = max(0, b.x2 - b.x1) * max(0, b.y2 - b.y1)
    union = area_a + area_b - inter
    return float(inter / union) if union > 0 else 0.0


def match_boxes(preds: list[Box], gts: list[Box], iou_threshold: float) -> tuple[int, int, int, float]:
    pairs: list[tuple[float, int, int]] = []
    for pi, pred in enumerate(preds):
        for gi, gt in enumerate(gts):
            pairs.append((iou(pred, gt), pi, gi))
    pairs.sort(reverse=True, key=lambda item: item[0])

    used_preds: set[int] = set()
    used_gts: set[int] = set()
    matched_ious: list[float] = []
    for score, pi, gi in pairs:
        if score < iou_threshold or pi in used_preds or gi in used_gts:
            continue
        used_preds.add(pi)
        used_gts.add(gi)
        matched_ious.append(score)

    tp = len(matched_ious)
    fp = max(0, len(preds) - tp)
    fn = max(0, len(gts) - tp)
    mean_iou = float(np.mean(matched_ious)) if matched_ious else 0.0
    return tp, fp, fn, mean_iou


def draw_visual(
    image: np.ndarray,
    heatmap: np.ndarray,
    pred_boxes: list[Box],
    gt_boxes: list[Box],
    class_name: str,
    confidence: float,
    out_path: Path,
) -> None:
    height, width = image.shape[:2]
    heat = cv2.resize(heatmap, (width, height), interpolation=cv2.INTER_CUBIC)
    heat_color = cv2.applyColorMap(np.uint8(255 * heat), cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(image, 0.62, heat_color, 0.38, 0)

    for box in gt_boxes:
        cv2.rectangle(overlay, (box.x1, box.y1), (box.x2, box.y2), (0, 220, 0), 2)
    for box in pred_boxes:
        cv2.rectangle(overlay, (box.x1, box.y1), (box.x2, box.y2), (0, 0, 255), 2)

    label = f"{class_name} {confidence * 100:.1f}%"
    cv2.putText(overlay, label, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), overlay)


def write_report(out_dir: Path, summary: dict, rows: list[dict], layer_name: str, args: argparse.Namespace) -> None:
    report = out_dir / "REPORT.md"
    box_source = "YOLO localization boxes" if args.localizer == "yolo" else "Grad-CAM localization boxes"
    heatmap_source = "YOLO objectness map" if args.localizer == "yolo" else "Grad-CAM heatmap"
    lines = [
        "# Classification-First Localization Report",
        "",
        "This run implements the revised DL workflow: Stage 1 classification, Stage 2 localization.",
        "",
        "## Configuration",
        "",
        f"- Data: `{args.data}`",
        f"- Split: `{args.split}`",
        f"- Model: `{args.model}`",
        f"- Stage 2 localizer: `{args.localizer}`",
        f"- Grad-CAM feature layer: `{layer_name}`",
        f"- Heatmap threshold: `{args.heatmap_threshold}`",
        f"- YOLO weights: `{args.yolo_weights}`",
        f"- YOLO confidence: `{args.yolo_conf}`",
        f"- IoU threshold: `{args.iou_threshold}`",
        f"- Sample mode: `{args.sample_mode}`",
        f"- Seed: `{args.seed}`",
        "",
        "## Localization Metrics",
        "",
        "| Metric | Value |",
        "|---|---:|",
        f"| Images evaluated | {summary['images']} |",
        f"| Ground-truth boxes | {summary['gt_boxes']} |",
        f"| Predicted boxes | {summary['pred_boxes']} |",
        f"| True positives | {summary['tp']} |",
        f"| False positives | {summary['fp']} |",
        f"| False negatives | {summary['fn']} |",
        f"| Precision | {summary['precision']:.4f} |",
        f"| Recall | {summary['recall']:.4f} |",
        f"| Mean matched IoU | {summary['mean_matched_iou']:.4f} |",
        f"| Classification gate hit-rate | {summary['classification_gate_hit_rate']:.4f} |",
        "",
        "## Notes",
        "",
        f"- Visual overlay uses a `{heatmap_source}`.",
        "- Green boxes in visual outputs are ground truth.",
        f"- Red boxes are `{box_source}`.",
        "- Classification gate hit-rate is diagnostic only; final DL evaluation is localization-first.",
        "",
        "## Artifacts",
        "",
        "- `predictions.csv`",
        "- `summary.json`",
        "- `visuals/*.jpg`",
    ]
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")

    with (out_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump({"summary": summary, "rows": rows}, f, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run classification-first Grad-CAM localization.")
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--yolo-weights", type=Path, default=DEFAULT_YOLO_WEIGHTS)
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--max-images", type=int, default=50)
    parser.add_argument("--max-visuals", type=int, default=16)
    parser.add_argument("--localizer", default="yolo", choices=["gradcam", "yolo"])
    parser.add_argument("--yolo-conf", type=float, default=0.35)
    parser.add_argument("--sample-mode", default="stratified", choices=["first", "random", "stratified"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--heatmap-threshold", type=float, default=0.45)
    parser.add_argument("--min-area-ratio", type=float, default=0.001)
    parser.add_argument("--max-boxes", type=int, default=3)
    parser.add_argument("--iou-threshold", type=float, default=0.5)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    visuals_dir = args.out_dir / "visuals"
    visuals_dir.mkdir(parents=True, exist_ok=True)

    images_dir, dataset_classes = resolve_split_images(args.data, args.split)
    paths = image_paths(images_dir, args.max_images, args.sample_mode, args.seed)
    if not paths:
        raise FileNotFoundError(f"No images found in {images_dir}")

    print(f"[INFO] Loading classifier: {args.model}")
    model = keras.models.load_model(str(args.model), compile=False)
    grad_model, layer_name = build_grad_model(model)
    yolo_model = None
    if args.localizer == "yolo":
        from ultralytics import YOLO

        print(f"[INFO] Loading Stage 2 YOLO localizer: {args.yolo_weights}")
        yolo_model = YOLO(str(args.yolo_weights))
    print(f"[INFO] Grad-CAM layer: {layer_name}")
    print(f"[INFO] Evaluating {len(paths)} images from {images_dir}")

    rows: list[dict] = []
    tp_total = fp_total = fn_total = 0
    gt_total = pred_total = 0
    matched_ious: list[float] = []
    gate_hits = 0

    csv_path = args.out_dir / "predictions.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "image",
                "pred_class",
                "pred_conf",
                "gt_classes",
                "pred_boxes",
                "gt_boxes",
                "tp",
                "fp",
                "fn",
                "mean_matched_iou",
            ],
        )
        writer.writeheader()

        for index, image_path in enumerate(paths):
            image = cv2.imread(str(image_path))
            if image is None:
                continue
            height, width = image.shape[:2]
            gt_boxes = read_yolo_boxes(label_path_for_image(image_path), width, height)

            image_input = preprocess_image(image, (224, 224))
            probs_raw = model.predict(image_input, verbose=0)[0]
            pred_idx = int(np.argmax(probs_raw))
            pred_conf = float(probs_raw[pred_idx])
            pred_class = CLASSIFIER_CLASSES[pred_idx] if pred_idx < len(CLASSIFIER_CLASSES) else str(pred_idx)

            heatmap = np.zeros((height, width), dtype=np.float32)
            pred_boxes: list[Box] = []
            if args.localizer == "yolo":
                if yolo_model is None:
                    raise RuntimeError("YOLO localizer was not loaded.")
                pred_boxes = boxes_from_yolo(yolo_model, image, args.yolo_conf)
                heatmap = heatmap_from_boxes(width, height, pred_boxes)
            elif pred_class != "Background":
                heatmap, _ = compute_gradcam(grad_model, image_input, pred_idx)
                pred_boxes = boxes_from_heatmap(
                    heatmap,
                    width,
                    height,
                    args.heatmap_threshold,
                    args.min_area_ratio,
                    args.max_boxes,
                    pred_conf,
                    pred_idx,
                )

            tp, fp, fn, mean_iou = match_boxes(pred_boxes, gt_boxes, args.iou_threshold)
            tp_total += tp
            fp_total += fp
            fn_total += fn
            gt_total += len(gt_boxes)
            pred_total += len(pred_boxes)
            if mean_iou > 0:
                matched_ious.append(mean_iou)

            gt_class_names = []
            for box in gt_boxes:
                if box.cls is not None and 0 <= box.cls < len(dataset_classes):
                    gt_class_names.append(dataset_classes[box.cls])
            if pred_class in set(gt_class_names):
                gate_hits += 1

            row = {
                "image": image_path.name,
                "pred_class": pred_class,
                "pred_conf": round(pred_conf, 6),
                "gt_classes": ",".join(sorted(set(gt_class_names))),
                "pred_boxes": len(pred_boxes),
                "gt_boxes": len(gt_boxes),
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "mean_matched_iou": round(mean_iou, 6),
            }
            rows.append(row)
            writer.writerow(row)

            if index < args.max_visuals:
                out_path = visuals_dir / f"{image_path.stem}_{args.localizer}.jpg"
                draw_visual(image, heatmap, pred_boxes, gt_boxes, pred_class, pred_conf, out_path)

            print(
                f"[{index + 1:03d}/{len(paths):03d}] {image_path.name}: "
                f"{pred_class} {pred_conf:.3f}, boxes pred/gt={len(pred_boxes)}/{len(gt_boxes)}, "
                f"tp/fp/fn={tp}/{fp}/{fn}, IoU={mean_iou:.3f}"
            )

    precision = tp_total / max(1, tp_total + fp_total)
    recall = tp_total / max(1, tp_total + fn_total)
    summary = {
        "images": len(rows),
        "gt_boxes": gt_total,
        "pred_boxes": pred_total,
        "tp": tp_total,
        "fp": fp_total,
        "fn": fn_total,
        "precision": precision,
        "recall": recall,
        "mean_matched_iou": float(np.mean(matched_ious)) if matched_ious else 0.0,
        "classification_gate_hit_rate": gate_hits / max(1, len(rows)),
    }
    write_report(args.out_dir, summary, rows, layer_name, args)
    print(f"[SUCCESS] Report saved to: {args.out_dir / 'REPORT.md'}")


if __name__ == "__main__":
    main()
