#!/usr/bin/env python
"""Field-rebalance fine-tune of the deployed YOLO26n (Step 1 execution).

Fine-tunes from the deployed detector on the 26.6%-field rebalanced set. Low LR to
protect studio features; mosaic on for context diversity; mixup/copy_paste off
(no synthetic artifacts). Early-stops on field-val mAP50.

Run: .venv311\\Scripts\\python.exe scripts/finetune_field_rebalance.py
"""
from pathlib import Path
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]

if __name__ == "__main__":
    model = YOLO(str(ROOT / "models" / "trained" / "yolov11_detector" / "best.pt"))
    model.train(
        data=str(ROOT / "external_datasets" / "field_rebalance_v1" / "data.yaml"),
        project=str(ROOT / "runs" / "detect"),
        name="field_rebalance_v1_k6",
        exist_ok=True,
        epochs=30, imgsz=640, batch=16, workers=2, cache=False,
        optimizer="AdamW", lr0=1e-4, lrf=0.1, cos_lr=True, warmup_epochs=1.0,
        mosaic=1.0, close_mosaic=10, mixup=0.0, copy_paste=0.0,
        hsv_h=0.015, hsv_s=0.6, hsv_v=0.5, degrees=8.0, scale=0.5, fliplr=0.5,
        patience=10, seed=42, verbose=True, plots=False,
    )
    print("TRAIN_DONE", model.trainer.best)
