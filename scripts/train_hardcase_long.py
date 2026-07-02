#!/usr/bin/env python
"""Longer YOLO26n hard-case retrain.

Must be a real module (not `python -c`) because YOLO's DataLoader uses spawn
multiprocessing on Windows, which requires an ``if __name__ == "__main__"`` guard.
Run with the CUDA env: ``.venv311\\Scripts\\python.exe scripts/train_hardcase_long.py``
"""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    import torch
    from ultralytics import YOLO

    assert torch.cuda.is_available(), "CUDA not available - use .venv311 python"
    print(f"[INFO] Device: {torch.cuda.get_device_name(0)}")

    checkpoint_path = ROOT / "runs" / "detect" / "yolo26n_hardcase_v2_long" / "weights" / "last.pt"
    resume_training = checkpoint_path.exists()

    if resume_training:
        print(f"[INFO] Resuming training from checkpoint: {checkpoint_path}")
        model = YOLO(str(checkpoint_path))
    else:
        print("[INFO] Starting fresh training from pretrained weights...")
        model = YOLO(str(ROOT / "models" / "pretrained" / "yolo26n.pt"))

    model.train(
        data=str(ROOT / "external_datasets" / "yolo26_hardcase_dataset_v1" / "data.yaml"),
        epochs=100,
        imgsz=640,
        batch=16,
        cache=False,          # RAM too small for ram-cache, disk too full for disk-cache
        cos_lr=True,
        patience=40,
        device=0,
        workers=4,
        project=str(ROOT / "runs" / "detect"),
        name="yolo26n_hardcase_v2_long",
        exist_ok=True,
        plots=True,
        resume=resume_training,
    )
    print("TRAIN_DONE")


if __name__ == "__main__":
    main()
