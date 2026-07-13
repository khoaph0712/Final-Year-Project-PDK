from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

from ultralytics import YOLO


def collect_images(dataset_yaml: Path, limit: int) -> list[Path]:
    root = None
    val = None
    for line in dataset_yaml.read_text(encoding="utf-8").splitlines():
        if line.startswith("path:"):
            root = Path(line.split(":", 1)[1].strip())
        elif line.startswith("val:"):
            val = line.split(":", 1)[1].strip()
    if root is None or val is None:
        raise ValueError(f"Could not parse dataset path/val from {dataset_yaml}")

    image_dir = root / val
    images = sorted(
        path
        for ext in ("*.jpg", "*.jpeg", "*.png", "*.webp")
        for path in image_dir.glob(ext)
    )
    return images[:limit]


def metric_value(metrics, path: str) -> float:
    value = metrics
    for key in path.split("."):
        value = getattr(value, key)
    return float(value)


def benchmark_model(name: str, weights: Path, data: Path, images: list[Path], imgsz: int) -> dict:
    model = YOLO(str(weights))
    val_metrics = model.val(
        data=str(data),
        imgsz=imgsz,
        batch=16,
        device="cpu",
        plots=False,
        verbose=False,
        project="runs/detect",
        name=f"benchmark_{name}",
        exist_ok=True,
    )

    latencies = []
    for image in images:
        started = time.perf_counter()
        model.predict(str(image), imgsz=imgsz, device="cpu", conf=0.30, iou=0.55, verbose=False)
        latencies.append((time.perf_counter() - started) * 1000)

    return {
        "name": name,
        "weights": str(weights),
        "map50": metric_value(val_metrics.box, "map50"),
        "map50_95": metric_value(val_metrics.box, "map"),
        "precision": metric_value(val_metrics.box, "mp"),
        "recall": metric_value(val_metrics.box, "mr"),
        "latency_images": len(latencies),
        "latency_ms_mean": statistics.mean(latencies),
        "latency_ms_median": statistics.median(latencies),
        "latency_ms_p95": sorted(latencies)[int(len(latencies) * 0.95) - 1],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path("external_datasets/yolo26_hardcase_dataset_v1/data.yaml"))
    parser.add_argument("--old", type=Path, default=Path("models/trained/yolov11_detector/best_before_.pt"))
    parser.add_argument("--new", type=Path, default=Path("runs/detect/yolo26n_hardcase_dataset_v1/weights/best.pt"))
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--latency-images", type=int, default=80)
    parser.add_argument("--out", type=Path, default=Path("runs/detect/yolo11_vs_yolo26_benchmark.json"))
    args = parser.parse_args()

    images = collect_images(args.data, args.latency_images)
    results = {
        "data": str(args.data),
        "imgsz": args.imgsz,
        "models": [
            benchmark_model("old_yolo11", args.old, args.data, images, args.imgsz),
            benchmark_model("new_yolo26n", args.new, args.data, images, args.imgsz),
        ],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps(results, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
