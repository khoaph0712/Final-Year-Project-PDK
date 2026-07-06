#!/usr/bin/env python
"""Class-agnostic (nc=1) field-expansion retrain from the deployed backbone.

nc changes 6->1, so Ultralytics reinitializes the detect head (backbone+neck transfer
from the deployed 6-class weights). Slightly higher lr0 than the F11 fine-tune to let the
fresh head learn; mosaic on, mixup/copy_paste off. Early-stops on field-val mAP50.

Run: .venv311\\Scripts\\python.exe scripts/finetune_class_agnostic.py
"""
from pathlib import Path
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]

if __name__ == "__main__":
    model = YOLO(str(ROOT / "models" / "trained" / "yolov11_detector" / "best.pt"))
    model.train(
        data=str(ROOT / "external_datasets" / "class_agnostic_field_v1" / "data.yaml"),
        project=str(ROOT / "runs" / "detect"),
        name="class_agnostic_field_v1", exist_ok=True,
        epochs=40, imgsz=640, batch=16, workers=2, cache=False,
        optimizer="AdamW", lr0=2e-4, lrf=0.1, cos_lr=True, warmup_epochs=2.0,
        mosaic=1.0, close_mosaic=10, mixup=0.0, copy_paste=0.0,
        hsv_h=0.015, hsv_s=0.6, hsv_v=0.5, degrees=8.0, scale=0.5, fliplr=0.5,
        patience=12, seed=42, verbose=True, plots=False,
    )
    print("TRAIN_DONE", model.trainer.best)
