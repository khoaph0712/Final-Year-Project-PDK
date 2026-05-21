"""Run YOLO waste-sorting predictions on your own images/folders.

Examples:
    python scripts/predict_images.py --source path\to\image.jpg
    python scripts/predict_images.py --source path\to\folder --conf 0.35
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ultralytics import YOLO


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_WEIGHTS = ROOT / "runs" / "dl" / "trash_yolov8n_tuned_v1" / "weights" / "best.pt"
DEFAULT_OUT = ROOT / "runs" / "manual_tests" / "yolo_predictions"
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
UNCERTAIN_OUTPUT = "uncertain - retake photo closer/brighter"
CLASS_MIN_CONF = {
    "plastic": 0.35,
    "glass": 0.35,
    "metal": 0.35,
    "paper": 0.35,
    "cardboard": 0.40,
    "organic": 0.30,
    "other": 0.50,
}
SORTING_RULES = {
    "plastic": {
        "bin": "recyclable",
        "instruction": "Put in the recyclable bin if it is clean; rinse first if contaminated.",
    },
    "glass": {
        "bin": "recyclable",
        "instruction": "Put in the recyclable bin; keep broken glass separated if required locally.",
    },
    "metal": {
        "bin": "recyclable",
        "instruction": "Put in the recyclable bin if it is empty and reasonably clean.",
    },
    "paper": {
        "bin": "recyclable",
        "instruction": "Put in the recyclable bin if dry and clean.",
    },
    "cardboard": {
        "bin": "recyclable",
        "instruction": "Flatten and put in the recyclable bin if dry and clean.",
    },
    "organic": {
        "bin": "organic",
        "instruction": "Put in the organic/compost bin.",
    },
    "other": {
        "bin": "general waste",
        "instruction": "Put in the general waste bin unless local rules say otherwise.",
    },
}


def collect_sources(source: Path) -> list[Path]:
    if source.is_file():
        if source.suffix.lower() not in IMAGE_EXTS:
            raise SystemExit(f"Unsupported image type: {source.suffix}")
        return [source]
    if source.is_dir():
        images = [p for p in source.rglob("*") if p.suffix.lower() in IMAGE_EXTS]
        if not images:
            raise SystemExit(f"No supported images found under {source}")
        return sorted(images)
    raise SystemExit(f"Source does not exist: {source}")


def box_area_ratio(xyxy: list[float], image_shape: tuple[int, int]) -> float:
    h, w = image_shape
    x1, y1, x2, y2 = xyxy
    box_w = max(0.0, x2 - x1)
    box_h = max(0.0, y2 - y1)
    return (box_w * box_h) / max(1.0, float(w * h))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True, help="Image file or folder to predict.")
    parser.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--conf", type=float, default=0.35, help="Base confidence threshold. Use 0.35+ for real-world demos.")
    parser.add_argument("--iou", type=float, default=0.5)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--min-box-area-ratio", type=float, default=0.005, help="Reject boxes smaller than this image-area ratio.")
    parser.add_argument("--top1", action=argparse.BooleanOptionalAction, default=True, help="Use only the highest-confidence kept detection for sorting.")
    parser.add_argument("--class-thresholds", action=argparse.BooleanOptionalAction, default=True, help="Use stricter per-class minimum confidence rules.")
    args = parser.parse_args()

    images = collect_sources(args.source)
    args.out.mkdir(parents=True, exist_ok=True)

    model = YOLO(str(args.weights))
    results = model.predict(
        source=[str(p) for p in images],
        imgsz=args.imgsz,
        conf=args.conf,
        iou=args.iou,
        save=True,
        project=str(args.out.parent),
        name=args.out.name,
        exist_ok=True,
        verbose=False,
    )

    rows: list[dict] = []
    for image_path, result in zip(images, results):
        detections: list[dict] = []
        rejected_detections: list[dict] = []
        names = result.names
        if result.boxes is not None:
            for box in result.boxes:
                cls_id = int(box.cls.item())
                class_name = names[cls_id]
                confidence = float(box.conf.item())
                xyxy = [round(float(v), 2) for v in box.xyxy[0].tolist()]
                area_ratio = box_area_ratio(xyxy, result.orig_shape)
                min_conf = max(args.conf, CLASS_MIN_CONF.get(class_name, args.conf)) if args.class_thresholds else args.conf
                sorting = SORTING_RULES.get(
                    class_name,
                    {
                        "bin": "unknown",
                        "instruction": "No sorting rule configured for this class.",
                    },
                )
                row = {
                    "class_id": cls_id,
                    "class_name": class_name,
                    "confidence": round(confidence, 4),
                    "min_conf_used": round(min_conf, 4),
                    "box_area_ratio": round(area_ratio, 6),
                    "sorting_bin": sorting["bin"],
                    "sorting_instruction": sorting["instruction"],
                    "xyxy": xyxy,
                }
                if confidence < min_conf:
                    row["reject_reason"] = "low_confidence"
                    rejected_detections.append(row)
                    continue
                if area_ratio < args.min_box_area_ratio:
                    row["reject_reason"] = "tiny_box"
                    rejected_detections.append(row)
                    continue
                detections.append(row)
        detections = sorted(detections, key=lambda d: d["confidence"], reverse=True)
        sorting_detections = detections[:1] if args.top1 else detections
        if detections:
            bins = sorted({d["sorting_bin"] for d in sorting_detections})
            sorting_output = ", ".join(bins)
        else:
            sorting_output = UNCERTAIN_OUTPUT
        rows.append(
            {
                "image": str(image_path),
                "sorting_output": sorting_output,
                "detections": detections,
                "sorting_detections": sorting_detections,
                "rejected_detections": rejected_detections,
            }
        )

    summary_path = args.out / "predictions_summary.json"
    summary_path.write_text(json.dumps(rows, indent=2), encoding="utf-8")

    print(f"[OK] Predicted {len(images)} image(s)")
    print(f"[OK] Annotated images: {args.out}")
    print(f"[OK] JSON summary: {summary_path}")
    for row in rows:
        print(f"- {Path(row['image']).name}: {row['sorting_output']}")


if __name__ == "__main__":
    main()
