"""Run YOLO waste-sorting predictions on your own images/folders.

Examples:
    python scripts/predict_images.py --source path\to\image.jpg
    python scripts/predict_images.py --source path\to\folder --conf 0.25
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ultralytics import YOLO


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_WEIGHTS = ROOT / "runs" / "dl" / "trash_yolov8n_v3" / "weights" / "best.pt"
DEFAULT_OUT = ROOT / "runs" / "manual_tests" / "yolo_predictions"
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True, help="Image file or folder to predict.")
    parser.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold. Try 0.15 for higher recall.")
    parser.add_argument("--iou", type=float, default=0.5)
    parser.add_argument("--imgsz", type=int, default=640)
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
        names = result.names
        if result.boxes is not None:
            for box in result.boxes:
                cls_id = int(box.cls.item())
                class_name = names[cls_id]
                sorting = SORTING_RULES.get(
                    class_name,
                    {
                        "bin": "unknown",
                        "instruction": "No sorting rule configured for this class.",
                    },
                )
                detections.append(
                    {
                        "class_id": cls_id,
                        "class_name": class_name,
                        "confidence": round(float(box.conf.item()), 4),
                        "sorting_bin": sorting["bin"],
                        "sorting_instruction": sorting["instruction"],
                        "xyxy": [round(float(v), 2) for v in box.xyxy[0].tolist()],
                    }
                )
        if detections:
            bins = sorted({d["sorting_bin"] for d in detections})
            sorting_output = ", ".join(bins)
        else:
            sorting_output = "no detection"
        rows.append({"image": str(image_path), "sorting_output": sorting_output, "detections": detections})

    summary_path = args.out / "predictions_summary.json"
    summary_path.write_text(json.dumps(rows, indent=2), encoding="utf-8")

    print(f"[OK] Predicted {len(images)} image(s)")
    print(f"[OK] Annotated images: {args.out}")
    print(f"[OK] JSON summary: {summary_path}")
    for row in rows:
        print(f"- {Path(row['image']).name}: {row['sorting_output']}")


if __name__ == "__main__":
    main()
