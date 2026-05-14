# Tuned YOLO Training Progress

Training was started on the tuned dataset:

```text
tuned_dataset_v1\data.yaml
```

Command:

```powershell
yolo detect train `
  model=runs\dl\trash_yolov8n_v3\weights\best.pt `
  data=tuned_dataset_v1\data.yaml `
  epochs=50 imgsz=800 batch=16 `
  project=runs\dl name=trash_yolov8n_tuned_v1 exist_ok=True
```

The run was interrupted before completing all 50 epochs, but it reached epoch 33 and already passed the target line on validation mAP@0.5.

Latest recorded row from `runs\dl\trash_yolov8n_tuned_v1\results.csv`:

| Epoch | Precision | Recall | mAP@0.5 | mAP@0.5:0.95 |
|---:|---:|---:|---:|---:|
| 33 | 0.84904 | 0.71189 | 0.81111 | 0.60179 |

This should be reported as **training progress**, not final test performance. The next step is to let training finish to epoch 50, then run `scripts\evaluate.py` on both validation and test splits.

Resume command:

```powershell
yolo detect train resume=True model=runs\dl\trash_yolov8n_tuned_v1\weights\last.pt
```

Evaluation command after training completes:

```powershell
.\.venv311\Scripts\python.exe scripts\evaluate.py `
  --weights runs\dl\trash_yolov8n_tuned_v1\weights\best.pt `
  --data tuned_dataset_v1\data.yaml `
  --out runs\dl\trash_yolov8n_tuned_v1\quality_check `
  --split both
```
