#!/usr/bin/env python
"""Fine-tune the retrained YOLO26n at 960px (from v2_long best.pt).

Aligns training resolution with the web server's imgsz=960 inference and is the
primary tiny-object lever (organic median box ~74px at 640 -> ~110px at 960).
batch=8 because 960px at batch 16 would OOM the 12GB RTX 3060; cache off (RAM/disk
limits). Must be a script file: Windows spawn DataLoader needs the __main__ guard.

Run: .venv311\\Scripts\\python.exe scripts/train_hardcase_960.py
"""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    import torch
    from ultralytics import YOLO

    assert torch.cuda.is_available(), "CUDA not available - use .venv311 python"
    print(f"[INFO] Device: {torch.cuda.get_device_name(0)}")

    run_dir = ROOT / "runs" / "detect" / "yolo26n_hardcase_v3_960"
    last = run_dir / "weights" / "last.pt"
    resume = last.exists()
    source = last if resume else ROOT / "runs" / "detect" / "yolo26n_hardcase_v2_long" / "weights" / "best.pt"
    print(f"[INFO] {'Resuming' if resume else 'Fine-tuning from'}: {source}")

    model = YOLO(str(source))
    model.train(
        data=str(ROOT / "external_datasets" / "yolo26_hardcase_dataset_v1" / "data.yaml"),
        epochs=40,
        imgsz=960,
        batch=8,
        cache=False,
        cos_lr=True,
        lr0=0.003,           # fine-tune, not from-scratch
        patience=15,
        device=0,
        workers=4,
        project=str(ROOT / "runs" / "detect"),
        name="yolo26n_hardcase_v3_960",
        exist_ok=True,
        plots=True,
        resume=resume,
    )
    print("TRAIN_DONE")


if __name__ == "__main__":
    main()
