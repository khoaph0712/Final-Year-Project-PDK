#!/usr/bin/env python
"""Train YOLO26n localizer on the super dataset plus TACO hard cases."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA = ROOT / "external_datasets" / "yolo26_hardcase_dataset_v1" / "data.yaml"
DEFAULT_PROJECT = ROOT / "runs" / "detect"
DEFAULT_MODEL = ROOT / "models" / "pretrained" / "yolo26n.pt"
DEFAULT_NAME = "yolo26n_hardcase_dataset_v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--model", default=str(DEFAULT_MODEL))
    parser.add_argument("--project", type=Path, default=DEFAULT_PROJECT)
    parser.add_argument("--name", default=DEFAULT_NAME)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--cache", default="disk")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--lr0", type=float, default=None)
    parser.add_argument("--lrf", type=float, default=None)
    parser.add_argument("--patience", type=int, default=None)
    parser.add_argument("--cos-lr", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.data.exists():
        print(f"[ERROR] Dataset config not found: {args.data}")
        sys.exit(1)

    import torch
    from ultralytics import YOLO

    if not torch.cuda.is_available():
        print("[ERROR] CUDA GPU not available; YOLO26 training on CPU is not practical.")
        sys.exit(1)

    run_dir = args.project / args.name
    last_checkpoint = run_dir / "weights" / "last.pt"
    model_name = str(last_checkpoint) if args.resume and last_checkpoint.exists() else args.model

    print("=====================================================================")
    print("                  YOLO26n HARD-CASE LOCALIZER TRAINING               ")
    print("=====================================================================")
    print(f"[INFO] Dataset: {args.data}")
    print(f"[INFO] Model:   {model_name}")
    print(f"[INFO] Run:     {run_dir}")
    print(f"[INFO] Epochs:  {args.epochs}")
    print(f"[INFO] Batch:   {args.batch}")
    print(f"[INFO] Image:   {args.imgsz}")
    print(f"[INFO] Device:  {torch.cuda.get_device_name(0)}")

    model = YOLO(model_name)
    train_kwargs = {
        "data": str(args.data),
        "epochs": args.epochs,
        "batch": args.batch,
        "imgsz": args.imgsz,
        "device": 0,
        "project": str(args.project),
        "name": args.name,
        "exist_ok": True,
        "resume": args.resume and last_checkpoint.exists(),
        "val": True,
        "cache": args.cache,
        "workers": args.workers,
        "plots": True,
    }
    if args.lr0 is not None:
        train_kwargs["lr0"] = args.lr0
    if args.lrf is not None:
        train_kwargs["lrf"] = args.lrf
    if args.patience is not None:
        train_kwargs["patience"] = args.patience
    if args.cos_lr:
        train_kwargs["cos_lr"] = True

    model.train(
        **train_kwargs,
    )
    print("=====================================================================")
    print(f"[SUCCESS] YOLO26n training complete: {run_dir / 'weights' / 'best.pt'}")
    print("=====================================================================")


if __name__ == "__main__":
    main()
