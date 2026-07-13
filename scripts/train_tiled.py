#!/usr/bin/env python
"""Train yolo26n on the tile-augmented dataset (step 1 of the tiny-object work).

Same recipe as the deployed baseline (scripts/train_hardcase_long.py: 100 epochs,
imgsz 640, batch 16, cos_lr, cache off, fresh from pretrained yolo26n) so the ONLY
difference vs the baseline is the tiled training data - a fair before/after.
Validation points at the ORIGINAL hardcase val (via the tiled data.yaml), matching
the baseline's model-selection signal. Final judgment is the clean test split.

Run: .venv311\\Scripts\\python.exe scripts/train_tiled.py
"""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUN = "yolo26n_hardcase_tiled_v1"


def main() -> None:
    import torch
    from ultralytics import YOLO

    assert torch.cuda.is_available(), "CUDA not available - use .venv311 python"
    print(f"[INFO] Device: {torch.cuda.get_device_name(0)}")

    last = ROOT / "runs" / "detect" / RUN / "weights" / "last.pt"
    resume = last.exists()
    model = YOLO(str(last)) if resume else YOLO(str(ROOT / "models" / "pretrained" / "yolo26n.pt"))
    print(f"[INFO] {'Resuming' if resume else 'Fresh from pretrained'}")

    model.train(
        data=str(ROOT / "external_datasets" / "yolo26_hardcase_tiled_v1" / "data.yaml"),
        epochs=100,
        imgsz=640,
        batch=16,
        cache=False,
        cos_lr=True,
        patience=40,
        device=0,
        workers=4,
        project=str(ROOT / "runs" / "detect"),
        name=RUN,
        exist_ok=True,
        plots=True,
        resume=resume,
    )
    print("TRAIN_DONE")


if __name__ == "__main__":
    main()
