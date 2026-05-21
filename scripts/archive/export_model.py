"""Export the trained YOLOv8n model to mobile-friendly formats.

Produces (next to the input weights):
    best.onnx                            ONNX opset 12, simplified
    best_saved_model/                    TF SavedModel (intermediate)
    best_float32.tflite                  TFLite, float32 (good accuracy, bigger)
    best_float16.tflite                  TFLite, float16 (balanced)
    best_int8.tflite                     TFLite, int8 quantised (fastest, smallest)
    best_metadata.json                   Class names + input/output tensor shapes

Usage:
    python scripts/export_model.py
    python scripts/export_model.py --weights runs/dl/trash_yolov8n_v3/weights/best.pt
    python scripts/export_model.py --imgsz 320           # smaller = faster on phone

Notes:
    - int8 quantisation uses a calibration sample from the val split.
    - If ONNX fails with `ml_dtypes` / `float4_e2m1fn`, run:
        .venv311\\Scripts\\python.exe -m pip install "ml_dtypes>=0.5.0" --upgrade
      Or skip ONNX only: --skip-onnx (TFLite still needs a working export stack).
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from ultralytics import YOLO


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_WEIGHTS = ROOT / "runs" / "dl" / "trash_yolov8n_v3" / "weights" / "best.pt"
DEFAULT_DATA = ROOT / "merged_dataset_v3" / "data.yaml"


def export_onnx(weights: Path, imgsz: int) -> Path:
    print("\n[>] Exporting ONNX…")
    model = YOLO(str(weights))
    path = model.export(format="onnx", imgsz=imgsz, opset=12, simplify=True, dynamic=False, half=False)
    print(f"    ONNX:   {path}")
    return Path(path)


def export_tflite(
    weights: Path,
    imgsz: int,
    data: Path,
    int8: bool,
    half: bool,
    suffix: str,
    fraction: float = 0.02,
) -> Path:
    tag = "int8" if int8 else ("float16" if half else "float32")
    print(f"\n[>] Exporting TFLite ({tag})…")
    model = YOLO(str(weights))
    kwargs: dict = {"format": "tflite", "imgsz": imgsz}
    if int8:
        kwargs.update({"int8": True, "data": str(data), "fraction": fraction})
    elif half:
        kwargs["half"] = True
    path = Path(model.export(**kwargs))
    target = weights.parent / f"best_{suffix}.tflite"
    if path != target:
        shutil.copy2(path, target)
    print(f"    TFLite: {target}")
    return target


def write_metadata(weights: Path, data: Path, imgsz: int) -> Path:
    import yaml

    with open(data, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    names = cfg["names"]

    meta = {
        "model": "yolov8n",
        "task": "detect",
        "input": {
            "name": "images",
            "shape": [1, 3, imgsz, imgsz],
            "dtype": "float32",
            "layout": "NCHW (ONNX) / NHWC (TFLite)",
            "normalization": "x / 255.0",
        },
        "output": {
            "description": "YOLOv8 raw predictions: [batch, 4+nc, num_anchors]",
            "num_classes": len(names),
            "anchors_at_imgsz_640": 8400,
            "needs_nms": True,
            "suggested_conf": 0.25,
            "suggested_iou": 0.45,
        },
        "classes": {str(i): n for i, n in enumerate(names)},
        "bin_mapping": {
            "plastic": "recycling",
            "glass": "recycling",
            "metal": "recycling",
            "paper": "recycling",
            "cardboard": "recycling",
            "organic": "compost",
            "other": "general",
        },
    }
    path = weights.parent / "best_metadata.json"
    path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"\n[OK] Metadata: {path}")
    return path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--skip-onnx", action="store_true")
    parser.add_argument("--skip-tflite", action="store_true")
    parser.add_argument("--skip-int8", action="store_true")
    parser.add_argument(
        "--int8-fraction",
        type=float,
        default=0.02,
        help="Fraction of val images used for INT8 calibration (avoids OOM). Default 0.02 ≈ 100 imgs.",
    )
    args = parser.parse_args()

    if not args.weights.exists():
        raise SystemExit(f"Weights not found: {args.weights}")

    if not args.skip_onnx:
        export_onnx(args.weights, args.imgsz)

    if not args.skip_tflite:
        export_tflite(args.weights, args.imgsz, args.data, int8=False, half=False, suffix="float32")
        export_tflite(args.weights, args.imgsz, args.data, int8=False, half=True, suffix="float16")
        if not args.skip_int8:
            export_tflite(
                args.weights,
                args.imgsz,
                args.data,
                int8=True,
                half=False,
                suffix="int8",
                fraction=args.int8_fraction,
            )

    write_metadata(args.weights, args.data, args.imgsz)
    print("\n[DONE] Drop the .tflite file + best_metadata.json into mobile/assets/model/")


if __name__ == "__main__":
    main()
