# ConvNeXt + 637-Feature Fine-Tuning Result

Run date: 2026-06-07

Command:

```powershell
.\.venv311\Scripts\python.exe scripts\train_convnext_ensemble_tuned.py
```

Hardware:

- Device: CUDA
- GPU: NVIDIA GeForce RTX 3060

Dataset:

- Source: `data/merged_dataset_v5`
- Train split used: 1,000 images per class x 7 classes = 7,000 balanced training images
- Test split used: 300 images per class x 7 classes = 2,100 balanced validation images
- Fused features: ConvNeXt-Tiny image embedding + 637 handcrafted features

Training:

- Phase 1: frozen ConvNeXt backbone, classifier-head warmup for 3 epochs
- Phase 2: progressive unfreezing of ConvNeXt final stage for 7 epochs

Result:

- Best tuned validation accuracy: 92.52%
- Approximate correct validation predictions: 1,943 / 2,100
- Best model artifact: `runs/dl/convnext_ensemble_tuned/best_convnext_ensemble_tuned.pth`

Epoch trace:

| Phase | Epoch | Train accuracy | Validation accuracy |
|---|---:|---:|---:|
| Warmup | 1 | 70.30% | n/a |
| Warmup | 2 | 84.49% | n/a |
| Warmup | 3 | 87.83% | n/a |
| Fine-tune | 1 | 92.75% | 90.81% |
| Fine-tune | 2 | 94.75% | 91.57% |
| Fine-tune | 3 | 95.57% | 92.33% |
| Fine-tune | 4 | 96.07% | 92.48% |
| Fine-tune | 5 | 96.57% | 92.48% |
| Fine-tune | 6 | 96.75% | 92.48% |
| Fine-tune | 7 | 97.38% | 92.52% |

Deployment note:

The current web API still serves the Keras EfficientNet classifier plus YOLO localizer. Deploying this tuned PyTorch classifier requires adding a PyTorch inference adapter to `web/server.py` or exporting the model before replacing the deployed classifier.

## Update: superseded by hard-case retrain + F4b fine-tune (2026-07-03)

`best_convnext_ensemble_tuned.pth` in this directory no longer holds the 92.52%-val weights described above. It has been overwritten twice since:

1. **Hard-case retrain** (2026-06-09, same warmup+unfreeze recipe, bigger split) — see `runs/dl/convnext_hardcase_tuned/RESULT.md`. Val macro-F1 0.9322, test macro-F1 0.9398.
2. **F4b detector-crop fine-tune** (2026-07-03, `scripts/finetune_stage2_on_detector_crops.py`, 8 epochs) — fixes train/serve skew (classifier trained on clean GT crops, served with imperfect detector-predicted crops). Promoted (`promote: true` in `runs/dl/convnext_detector_crops_ft/finetune_result.json`) and is the weight file currently loaded by `web/server.py` (`TUNED_CLASSIFIER_PATH`).

| | before F4b | after F4b (deployed now) |
|---|---:|---:|
| detector-crop val acc | 76.91% | 88.88% |
| detector-crop val F1 | — | 0.7989 |
| clean GT test acc | 92.93% | 93.77% |

Pre-F4b weights backed up at `best_convnext_ensemble_tuned_pre_ft_backup.pth` in this same directory.
