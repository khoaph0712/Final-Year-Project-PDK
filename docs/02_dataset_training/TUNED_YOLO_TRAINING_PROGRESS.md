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

The run was resumed and completed all 50 epochs.

Latest recorded row from `runs\dl\trash_yolov8n_tuned_v1\results.csv`:

| Epoch | Precision | Recall | mAP@0.5 | mAP@0.5:0.95 |
|---:|---:|---:|---:|---:|
| 50 | 0.84819 | 0.72737 | 0.81999 | 0.61276 |

Final evaluation was then run on validation and test splits.

Evaluation command:

```powershell
.\.venv311\Scripts\python.exe scripts\evaluate.py `
  --weights runs\dl\trash_yolov8n_tuned_v1\weights\best.pt `
  --data tuned_dataset_v1\data.yaml `
  --out runs\dl\trash_yolov8n_tuned_v1\quality_check `
  --split both
```

Final evaluation results:

- Validation mAP@0.5: `0.825`
- Validation mAP@0.5:0.95: `0.624`
- Test mAP@0.5: `0.810`
- Test mAP@0.5:0.95: `0.622`

Report the tuned YOLO result as **81.0% mAP@0.5 on the tuned test split**. This is object detection mAP, not the same metric as classical ML accuracy.
