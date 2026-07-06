#!/usr/bin/env python
"""Hard-negative fine-tune for small-object recall. From deployed 6-class weights.
Small-object aug: mosaic (tiling shrinks objects) + aggressive scale (zoom-out synthesizes
small instances). bbox-only labels so no copy_paste/mixup. Early-stop on field-val mAP50.
Run: python scripts/finetune_hardneg.py"""
from pathlib import Path
from ultralytics import YOLO
ROOT=Path(__file__).resolve().parents[1]
if __name__=="__main__":
    m=YOLO(str(ROOT/"models/trained/yolov11_detector/best.pt"))
    m.train(
        data=str(ROOT/"external_datasets/hardneg_smallobj_v1/data.yaml"),
        project=str(ROOT/"runs/detect"), name="hardneg_smallobj_v1", exist_ok=True,
        epochs=25, imgsz=640, batch=16, workers=2, cache=False,
        optimizer="AdamW", lr0=1e-4, lrf=0.1, cos_lr=True, warmup_epochs=1.0,
        mosaic=1.0, close_mosaic=8, scale=0.6, degrees=5.0, fliplr=0.5,
        mixup=0.0, copy_paste=0.0, hsv_h=0.015, hsv_s=0.6, hsv_v=0.5,
        patience=8, seed=42, verbose=True, plots=False,
    )
    print("TRAIN_DONE", m.trainer.best)
