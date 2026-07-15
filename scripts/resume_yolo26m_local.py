from ultralytics import YOLO

CKPT = r"C:\kaggle\working\runs\yolo26m_hardcase_v1\weights\last.pt"

if __name__ == "__main__":
    model = YOLO(CKPT)
    model.train(resume=True, cache=False, workers=2, batch=8)

    print("[SUCCESS] Training complete:", CKPT.replace("last.pt", "best.pt"))
