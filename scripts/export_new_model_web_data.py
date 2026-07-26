"""Export the newly trained model runs into web/assets/data/new_models.json.

Collects, for the presentation site:
  * six Stage-1 backbone benchmarks (RESULT.json + a freshly computed
    confusion matrix from the cached test features and saved MLP head),
  * the deployed ConvNeXt+637 eval on the same split as baseline,
  * the seven-architecture detector sweep (metrics + training curves),
  * the torch ResNet50 / MobileNetV2 retrain comparison (+ history CSVs).

Also copies the torch confusion-matrix PNGs and training logs into web/assets.
Static export only; nothing here retrains or promotes a model.
"""
from __future__ import annotations

import csv
import json
import shutil
from pathlib import Path

import joblib
import numpy as np
from sklearn.metrics import confusion_matrix

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "web" / "assets" / "data"
FIG_DIR = ROOT / "web" / "assets" / "figures"
CLASSES = ["plastic", "glass", "metal", "paper", "cardboard", "organic", "Background"]
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

STAGE1 = [
    # (run dir, file label, display name)
    ("vit_small_stage1_benchmark", "vit_small", "ViT-Small/16"),
    ("swin_tiny_stage1_benchmark", "swin_tiny", "Swin-Tiny"),
    ("convnextv2_material_stage1_benchmark", "convnextv2", "ConvNeXtV2-Tiny"),
    ("efficientnetv2_s_stage1_benchmark", "efficientnetv2_s", "EfficientNetV2-S"),
    ("resnet50_stage1_benchmark", "resnet50", "ResNet50"),
    ("mobilenetv3_large_stage1_benchmark", "mobilenetv3_large", "MobileNetV3-L"),
]


def test_labels() -> np.ndarray:
    root = ROOT / "data" / "hard_case_classifier_v1"
    y = []
    for label, cls in enumerate(CLASSES):
        class_dir = root / "test" / cls
        if not class_dir.exists():
            continue
        n = sum(1 for p in sorted(class_dir.iterdir()) if p.suffix.lower() in IMG_EXTS)
        y.extend([label] * n)
    return np.array(y, dtype=np.int64)


def per_class(report: dict) -> dict:
    return {
        cls: {
            "precision": round(report[cls]["precision"], 4),
            "recall": round(report[cls]["recall"], 4),
            "f1": round(report[cls]["f1-score"], 4),
            "support": int(report[cls]["support"]),
        }
        for cls in CLASSES
        if cls in report
    }


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    y_true = test_labels()
    print(f"[INFO] test labels: {len(y_true)}")

    out: dict = {"classes": CLASSES}

    # ---- stage-1 backbone benchmarks -------------------------------------
    stage1 = []
    for run_dir, label, nice in STAGE1:
        d = ROOT / "runs" / "dl" / run_dir
        res = json.loads((d / "RESULT.json").read_text())
        head = joblib.load(d / f"{label}_637_mlp_head.joblib")
        X = np.hstack([
            np.load(d / f"test_{label}.npy"),
            np.load(d / "test_handcrafted_637.npy"),
        ]).astype(np.float32)
        assert X.shape[0] == len(y_true), f"{label}: {X.shape[0]} vs {len(y_true)}"
        y_pred = head.predict(X)
        cm = confusion_matrix(y_true, y_pred, labels=range(len(CLASSES)))
        test = res["splits"]["test"]
        val = res["splits"]["val"]
        lat = res.get("latency_cpu", {})
        cand_key = next((k for k in lat if k.startswith("candidate")), None)
        stage1.append({
            "id": label,
            "nice": nice,
            "backbone": res.get("model", ""),
            "feature_dim": res.get("feature_dim"),
            "train_seconds": res.get("train_seconds"),
            "val_accuracy": round(val["accuracy"], 4),
            "val_macro_f1": round(val["macro_f1"], 4),
            "test_accuracy": round(test["accuracy"], 4),
            "test_macro_f1": round(test["macro_f1"], 4),
            "per_class": per_class(test["classification_report"]),
            "latency_ms": round(lat[cand_key]["median_ms"], 1) if cand_key else None,
            "latency_p95_ms": round(lat[cand_key]["p95_ms"], 1) if cand_key else None,
            "cm": cm.tolist(),
        })
        acc_check = float((y_pred == y_true).mean())
        print(f"[OK] {nice}: cm acc {acc_check:.4f} vs reported {test['accuracy']:.4f}")
        assert abs(acc_check - test["accuracy"]) < 1e-3, f"{label} CM does not match RESULT.json"
    out["stage1"] = stage1

    # ---- deployed baseline on the same split ------------------------------
    # Latency: each benchmark session re-measured the deployed model (47.6-58.7 ms
    # spread from session noise), so report the median across all six sessions
    # rather than whichever session happens to be listed first.
    dep = json.loads((ROOT / "runs" / "dl" / "convnext_hardcase_tuned" / "eval_new_hardcase.json").read_text())
    dt, dv = dep["splits"]["test"], dep["splits"]["val"]
    dep_lats = []
    dep_p95s = []
    for run_dir, _, _ in STAGE1:
        lat = json.loads((ROOT / "runs" / "dl" / run_dir / "RESULT.json").read_text())["latency_cpu"]["current_convnext_637"]
        dep_lats.append(lat["median_ms"])
        dep_p95s.append(lat["p95_ms"])
    out["deployed"] = {
        "id": "convnext_tiny",
        "nice": "ConvNeXt-Tiny (deployed)",
        "val_accuracy": round(dv["accuracy"], 4),
        "val_macro_f1": round(dv["macro_f1"], 4),
        "test_accuracy": round(dt["accuracy"], 4),
        "test_macro_f1": round(dt["macro_f1"], 4),
        "per_class": per_class(dt["classification_report"]),
        "latency_ms": round(float(np.median(dep_lats)), 1),
        "latency_p95_ms": round(float(np.median(dep_p95s)), 1),
        "latency_sessions_ms": [round(v, 1) for v in dep_lats],
    }

    # ---- seven-architecture detector sweep --------------------------------
    sweep = json.loads((ROOT / "docs" / "01_final_report" / "sweep_seven_architecture_metrics.json").read_text())
    out["detector_sweep"] = sweep

    # ---- torch retrain comparison -----------------------------------------
    torch_cmp = json.loads((ROOT / "runs" / "dl" / "comparison_models_torch" / "comparison_results.json").read_text())
    for name in ("resnet50", "mobilenetv2"):
        hist_path = ROOT / "runs" / "dl" / "comparison_models_torch" / name / "training_history.csv"
        with hist_path.open() as fh:
            rows = list(csv.DictReader(fh))
        torch_cmp[name]["history"] = {
            k: [round(float(r[k]), 4) for r in rows]
            for k in ("accuracy", "loss", "val_accuracy", "val_loss")
        }
        for png, dest in (("confusion_matrix.png", f"cm_torch_{name}.png"),):
            shutil.copyfile(ROOT / "runs" / "dl" / "comparison_models_torch" / name / png, FIG_DIR / dest)
    out["torch_comparison"] = torch_cmp

    # ---- logs ---------------------------------------------------------------
    shutil.copyfile(ROOT / "runs" / "dl" / "_stage1_extra_logs" / "train.log", OUT_DIR / "stage1_train.log")
    shutil.copyfile(ROOT / "runs" / "torch_cmp.log", OUT_DIR / "torch_cmp.log")
    # rtdetr_train_cleaned.txt has "N: " line-number prefixes and U+FFFD mojibake
    # baked in; strip both so the log viewer shows clean text.
    raw = (ROOT / "runs" / "rtdetr_train_cleaned.txt").read_text(encoding="utf-8", errors="replace")
    cleaned = "\n".join(
        line.split(": ", 1)[1] if line.split(": ", 1)[0].isdigit() else line
        for line in raw.replace("�", "").splitlines()
    )
    (OUT_DIR / "rtdetr_train.log").write_text(cleaned, encoding="utf-8")

    dest = OUT_DIR / "new_models.json"
    dest.write_text(json.dumps(out, separators=(",", ":")))
    print(f"[DONE] {dest} ({dest.stat().st_size/1024:.0f} KB)")


if __name__ == "__main__":
    main()
