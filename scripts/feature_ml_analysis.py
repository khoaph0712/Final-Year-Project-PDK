"""Feature-based ML analysis for waste-object classification.

Pipeline (lecture-requested):
1) Extract object crops from YOLO labels (bbox -> crop)
2) Compute handcrafted features in both domains:
   - Spatial domain: intensity/texture/edge statistics
   - Frequency domain: FFT radial-energy distribution
3) Train classic ML baselines first (LogReg, SVM, RandomForest)
4) Export comparison tables, confusion matrices, and chart commentary
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from dataclasses import dataclass
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATA = ROOT / "merged_dataset_v3" / "data.yaml"
DEFAULT_OUT = ROOT / "runs" / "feature_ml_analysis"
DEFAULT_DOMAIN_OUT = ROOT / "frequency_analysis"


@dataclass
class Sample:
    feature: np.ndarray
    class_id: int
    class_name: str


def resolve_split_images(ds_root: Path, split_rel: str) -> Path:
    path = Path(split_rel)
    if path.is_absolute():
        return path
    return ds_root / path


def image_to_label_path(image_path: Path) -> Path:
    # YOLO layout: */images/*.jpg -> */labels/*.txt
    return Path(str(image_path).replace("\\images\\", "\\labels\\")).with_suffix(".txt")


def clamp_box(x1: int, y1: int, x2: int, y2: int, w: int, h: int) -> tuple[int, int, int, int]:
    x1 = max(0, min(x1, w - 1))
    y1 = max(0, min(y1, h - 1))
    x2 = max(x1 + 1, min(x2, w))
    y2 = max(y1 + 1, min(y2, h))
    return x1, y1, x2, y2


def extract_spatial_features(crop_bgr: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, (64, 64), interpolation=cv2.INTER_AREA)

    mean = float(np.mean(gray))
    std = float(np.std(gray))
    p10 = float(np.percentile(gray, 10))
    p50 = float(np.percentile(gray, 50))
    p90 = float(np.percentile(gray, 90))

    # Texture proxies from gradients.
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    grad_mag = np.sqrt(gx * gx + gy * gy)
    grad_mean = float(np.mean(grad_mag))
    grad_std = float(np.std(grad_mag))

    # Edge density.
    edges = cv2.Canny(gray, 80, 160)
    edge_density = float(np.mean(edges > 0))

    return np.array([mean, std, p10, p50, p90, grad_mean, grad_std, edge_density], dtype=np.float32)


def extract_frequency_features(crop_bgr: np.ndarray, bins: int = 8) -> np.ndarray:
    gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, (64, 64), interpolation=cv2.INTER_AREA).astype(np.float32)
    gray = gray - float(np.mean(gray))

    fft = np.fft.fftshift(np.fft.fft2(gray))
    mag = np.abs(fft)
    power = mag * mag

    h, w = power.shape
    cy, cx = h // 2, w // 2
    yy, xx = np.indices((h, w))
    radius = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
    rmax = float(radius.max())

    features: list[float] = []
    total = float(np.sum(power)) + 1e-9
    for i in range(bins):
        r0 = (i / bins) * rmax
        r1 = ((i + 1) / bins) * rmax
        mask = (radius >= r0) & (radius < r1)
        band_energy = float(np.sum(power[mask])) / total
        features.append(band_energy)

    # Helpful scalar summary for "high-frequency richness".
    high_freq_energy = float(np.sum(features[bins // 2 :]))
    return np.array(features + [high_freq_energy], dtype=np.float32)


def extract_combined_features(crop_bgr: np.ndarray) -> np.ndarray:
    spatial = extract_spatial_features(crop_bgr)
    frequency = extract_frequency_features(crop_bgr)
    return np.concatenate([spatial, frequency], axis=0)


def load_samples(
    image_dir: Path,
    class_names: list[str],
    max_objects: int | None,
    min_box_px: int = 10,
) -> list[Sample]:
    image_paths = [
        p
        for p in image_dir.rglob("*")
        if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    ]
    image_paths.sort()

    samples: list[Sample] = []
    for img_path in image_paths:
        label_path = image_to_label_path(img_path)
        if not label_path.exists():
            continue

        img = cv2.imread(str(img_path))
        if img is None:
            continue
        h, w = img.shape[:2]

        lines = label_path.read_text(encoding="utf-8").splitlines()
        for line in lines:
            parts = line.strip().split()
            if len(parts) != 5:
                continue
            cls_id = int(float(parts[0]))
            if cls_id < 0 or cls_id >= len(class_names):
                continue

            cx, cy, bw, bh = [float(x) for x in parts[1:]]
            x1 = int((cx - bw / 2.0) * w)
            y1 = int((cy - bh / 2.0) * h)
            x2 = int((cx + bw / 2.0) * w)
            y2 = int((cy + bh / 2.0) * h)
            x1, y1, x2, y2 = clamp_box(x1, y1, x2, y2, w, h)

            if (x2 - x1) < min_box_px or (y2 - y1) < min_box_px:
                continue

            crop = img[y1:y2, x1:x2]
            feat = extract_combined_features(crop)
            samples.append(Sample(feat, cls_id, class_names[cls_id]))

    if max_objects is None or len(samples) <= max_objects:
        return samples
    return stratified_subsample(samples, max_objects, len(class_names), seed=42)


def stratified_subsample(samples: list[Sample], max_objects: int, n_classes: int, seed: int = 42) -> list[Sample]:
    by_class: dict[int, list[Sample]] = {cid: [] for cid in range(n_classes)}
    for s in samples:
        by_class[s.class_id].append(s)

    rng = random.Random(seed)
    for bucket in by_class.values():
        rng.shuffle(bucket)

    base = max(1, max_objects // max(1, n_classes))
    selected: list[Sample] = []
    leftovers: list[Sample] = []

    for cid in range(n_classes):
        bucket = by_class[cid]
        selected.extend(bucket[:base])
        leftovers.extend(bucket[base:])

    if len(selected) < max_objects and leftovers:
        rng.shuffle(leftovers)
        selected.extend(leftovers[: (max_objects - len(selected))])

    if len(selected) > max_objects:
        rng.shuffle(selected)
        selected = selected[:max_objects]
    return selected


def to_matrix(samples: list[Sample]) -> tuple[np.ndarray, np.ndarray]:
    x = np.vstack([s.feature for s in samples]).astype(np.float32)
    y = np.array([s.class_id for s in samples], dtype=np.int64)
    return x, y


def object_difference_report(x_train: np.ndarray, y_train: np.ndarray, class_names: list[str]) -> list[dict]:
    feature_means = {}
    for cid in np.unique(y_train):
        feature_means[int(cid)] = np.mean(x_train[y_train == cid], axis=0)

    global_mean = np.mean(x_train, axis=0)
    report = []
    for cid, mu in feature_means.items():
        delta = np.abs(mu - global_mean)
        top_idx = np.argsort(-delta)[:3].tolist()
        report.append(
            {
                "class_id": cid,
                "class_name": class_names[cid],
                "most_distinct_feature_indices": top_idx,
                "distinctiveness_score": float(np.mean(delta[top_idx])),
            }
        )
    report.sort(key=lambda r: r["distinctiveness_score"], reverse=True)
    return report


def build_models() -> dict[str, Pipeline | RandomForestClassifier]:
    return {
        # Why: strong linear baseline on scaled feature vectors.
        "logreg": Pipeline(
            [("scaler", StandardScaler()), ("clf", LogisticRegression(max_iter=1500, n_jobs=None))]
        ),
        # Why: handles non-linear boundaries in handcrafted features.
        "svm_rbf": Pipeline([("scaler", StandardScaler()), ("clf", SVC(kernel="rbf", probability=False))]),
        # Why: tree baseline, robust to mixed feature scales, gives importance.
        "rf": RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1),
    }


def save_confusion(y_true: np.ndarray, y_pred: np.ndarray, class_names: list[str], out_path: Path, title: str) -> None:
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))))
    fig, ax = plt.subplots(figsize=(8, 7))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(ax=ax, cmap="Blues", xticks_rotation=35, colorbar=False)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def save_comparison_chart(results: list[dict], out_path: Path) -> None:
    models = [r["model"] for r in results]
    acc = [r["accuracy"] for r in results]
    f1m = [r["f1_macro"] for r in results]

    x = np.arange(len(models))
    w = 0.36

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(x - w / 2, acc, width=w, label="Accuracy")
    ax.bar(x + w / 2, f1m, width=w, label="F1-macro")
    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.set_ylim(0.0, 1.0)
    ax.set_title("ML comparison on extracted features")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def save_domain_importance_chart(trained_models: dict[str, object], out_path: Path) -> None:
    # Feature layout:
    # 0..7 spatial, 8..16 frequency (8 radial bins + 1 high-freq summary).
    importances = None
    source = None

    rf_model = trained_models.get("rf")
    if rf_model is not None and hasattr(rf_model, "feature_importances_"):
        importances = np.asarray(rf_model.feature_importances_, dtype=np.float64)
        source = "RandomForest feature_importances_"

    if importances is None:
        raise ValueError("Cannot build domain-importance chart: no model with feature importance is available.")

    spatial_sum = float(np.sum(importances[:8]))
    frequency_sum = float(np.sum(importances[8:]))
    denom = spatial_sum + frequency_sum + 1e-12
    spatial_pct = 100.0 * spatial_sum / denom
    frequency_pct = 100.0 * frequency_sum / denom

    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(["Spatial", "Frequency"], [spatial_pct, frequency_pct], color=["#4E79A7", "#F28E2B"])
    ax.set_title("Domain contribution by feature importance")
    ax.set_ylabel("Contribution (%)")
    ax.set_ylim(0.0, 100.0)
    ax.grid(True, axis="y", alpha=0.3)
    for bar in bars:
        y = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2.0, y + 1.0, f"{y:.1f}%", ha="center", va="bottom", fontsize=10)
    ax.text(0.5, -0.18, f"Source: {source}", transform=ax.transAxes, ha="center", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def export_domain_summaries(
    x_train: np.ndarray,
    y_train: np.ndarray,
    class_names: list[str],
    out_dir: Path,
    domain_importance: tuple[float, float] | None,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    spatial_names = [
        "mean_intensity",
        "std_intensity",
        "p10_intensity",
        "p50_intensity",
        "p90_intensity",
        "grad_mean",
        "grad_std",
        "edge_density",
    ]
    frequency_names = [
        "fft_bin_1",
        "fft_bin_2",
        "fft_bin_3",
        "fft_bin_4",
        "fft_bin_5",
        "fft_bin_6",
        "fft_bin_7",
        "fft_bin_8",
        "high_freq_energy",
    ]

    spatial_csv = out_dir / "spatial_summary.csv"
    with spatial_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["class_id", "class_name", "n_objects", *spatial_names])
        for cid in range(len(class_names)):
            mask = y_train == cid
            n = int(np.sum(mask))
            if n == 0:
                vals = [0.0] * len(spatial_names)
            else:
                vals = np.mean(x_train[mask, :8], axis=0).tolist()
            w.writerow([cid, class_names[cid], n, *[round(float(v), 6) for v in vals]])

    frequency_csv = out_dir / "frequency_summary.csv"
    with frequency_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["class_id", "class_name", "n_objects", *frequency_names])
        for cid in range(len(class_names)):
            mask = y_train == cid
            n = int(np.sum(mask))
            if n == 0:
                vals = [0.0] * len(frequency_names)
            else:
                vals = np.mean(x_train[mask, 8:], axis=0).tolist()
            w.writerow([cid, class_names[cid], n, *[round(float(v), 6) for v in vals]])

    comparison_csv = out_dir / "domain_comparison.csv"
    spatial_pct = None
    frequency_pct = None
    if domain_importance is not None:
        spatial_pct, frequency_pct = domain_importance
    with comparison_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "metric",
                "spatial_value",
                "frequency_value",
                "note",
            ]
        )
        w.writerow(
            [
                "feature_count",
                8,
                9,
                "Spatial has 8 handcrafted features, Frequency has 9 FFT-derived features",
            ]
        )
        w.writerow(
            [
                "model_importance_pct",
                "" if spatial_pct is None else round(spatial_pct, 4),
                "" if frequency_pct is None else round(frequency_pct, 4),
                "From RandomForest feature_importances_",
            ]
        )


def write_report(
    out_dir: Path,
    class_names: list[str],
    train_count: int,
    test_count: int,
    results: list[dict],
    object_diff: list[dict],
) -> None:
    lines: list[str] = []
    lines.append("# Feature + ML Analysis Report\n")
    lines.append("## Scope")
    lines.append("- Mobile is intentionally excluded in this stage.")
    lines.append("- Focus: feature extraction + classical ML model comparison.\n")

    lines.append("## Pipeline")
    lines.append("1. Extract object crops from YOLO labels.")
    lines.append("2. Extract spatial-domain and frequency-domain features.")
    lines.append("3. Train ML models first (LogReg, SVM-RBF, RandomForest).")
    lines.append("4. Compare with accuracy/F1/confusion matrix.\n")

    lines.append("## Data")
    lines.append(f"- Train objects: **{train_count}**")
    lines.append(f"- Test objects: **{test_count}**")
    lines.append(f"- Classes: {', '.join(class_names)}\n")

    lines.append("## Model choice rationale")
    lines.append("- **Logistic Regression:** simple linear baseline on standardized feature vectors.")
    lines.append("- **SVM (RBF):** captures non-linear class boundaries from handcrafted features.")
    lines.append("- **Random Forest:** robust tree-based baseline and interpretable feature importance.\n")

    lines.append("## Results")
    lines.append("| Model | Accuracy | F1-macro |")
    lines.append("|---|---:|---:|")
    for r in results:
        lines.append(f"| {r['model']} | {r['accuracy']:.4f} | {r['f1_macro']:.4f} |")
    lines.append("")

    lines.append("## Class difference comments (objects)")
    lines.append(
        "Top distinct feature indices are reported to explain which classes differ most in spatial/frequency signatures."
    )
    for item in object_diff:
        lines.append(
            f"- **{item['class_name']}**: distinctiveness `{item['distinctiveness_score']:.4f}`, "
            f"top feature idx `{item['most_distinct_feature_indices']}`"
        )
    lines.append("")

    lines.append("## Chart comments")
    lines.append(
        "- `chart_domain_importance.png`: compares spatial vs frequency contribution based on model feature "
        "importance (not raw magnitude, so it is scale-safe)."
    )
    lines.append(
        "- `chart_model_comparison.png`: compares Accuracy/F1 across ML models to justify chosen baseline."
    )
    lines.append("")

    report_path = out_dir / "REPORT.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--max-train-objects", type=int, default=8000)
    parser.add_argument("--max-test-objects", type=int, default=3000)
    parser.add_argument("--domain-out", type=Path, default=DEFAULT_DOMAIN_OUT)
    args = parser.parse_args()

    cfg = yaml.safe_load(args.data.read_text(encoding="utf-8"))
    class_names = list(cfg["names"])
    ds_root = Path(cfg["path"])

    train_img_dir = resolve_split_images(ds_root, cfg["train"])
    test_img_dir = resolve_split_images(ds_root, cfg.get("test", cfg["val"]))

    args.out.mkdir(parents=True, exist_ok=True)

    print("[1/4] Extracting train features...")
    train_samples = load_samples(train_img_dir, class_names, args.max_train_objects)
    if not train_samples:
        raise SystemExit(f"No train samples loaded from {train_img_dir}")
    x_train, y_train = to_matrix(train_samples)

    print("[2/4] Extracting test features...")
    test_samples = load_samples(test_img_dir, class_names, args.max_test_objects)
    if not test_samples:
        raise SystemExit(f"No test samples loaded from {test_img_dir}")
    x_test, y_test = to_matrix(test_samples)

    class_support = {
        "train": {class_names[i]: int(v) for i, v in enumerate(np.bincount(y_train, minlength=len(class_names)))},
        "test": {class_names[i]: int(v) for i, v in enumerate(np.bincount(y_test, minlength=len(class_names)))},
    }
    (args.out / "class_support.json").write_text(json.dumps(class_support, indent=2), encoding="utf-8")

    models = build_models()
    results: list[dict] = []
    detailed: dict[str, dict] = {}
    trained_models: dict[str, object] = {}

    print("[3/4] Training ML models...")
    for name, model in models.items():
        print(f"  - {name}")
        model.fit(x_train, y_train)
        pred = model.predict(x_test)
        trained_models[name] = model
        acc = float(accuracy_score(y_test, pred))
        f1m = float(f1_score(y_test, pred, average="macro"))
        results.append({"model": name, "accuracy": acc, "f1_macro": f1m})

        detailed[name] = classification_report(
            y_test,
            pred,
            labels=list(range(len(class_names))),
            target_names=class_names,
            output_dict=True,
            zero_division=0,
        )
        save_confusion(y_test, pred, class_names, args.out / f"confusion_{name}.png", f"Confusion matrix - {name}")

    results.sort(key=lambda r: (r["f1_macro"], r["accuracy"]), reverse=True)
    (args.out / "metrics_summary.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    (args.out / "classification_reports.json").write_text(json.dumps(detailed, indent=2), encoding="utf-8")

    object_diff = object_difference_report(x_train, y_train, class_names)
    (args.out / "object_difference.json").write_text(json.dumps(object_diff, indent=2), encoding="utf-8")

    print("[4/4] Saving charts + report...")
    save_comparison_chart(results, args.out / "chart_model_comparison.png")
    save_domain_importance_chart(trained_models, args.out / "chart_domain_importance.png")
    rf_importances = np.asarray(trained_models["rf"].feature_importances_, dtype=np.float64)
    spatial_sum = float(np.sum(rf_importances[:8]))
    frequency_sum = float(np.sum(rf_importances[8:]))
    denom = spatial_sum + frequency_sum + 1e-12
    domain_importance = (100.0 * spatial_sum / denom, 100.0 * frequency_sum / denom)
    export_domain_summaries(x_train, y_train, class_names, args.domain_out, domain_importance)
    write_report(args.out, class_names, len(train_samples), len(test_samples), results, object_diff)

    print(f"[OK] Done. See {args.out} and {args.domain_out}")


if __name__ == "__main__":
    main()
