"""Feature-based ML analysis for waste-object classification.

Lecture-style workflow (do not reorder when presenting results):
1) **Spatial domain + frequency domain** — build descriptors for each object crop
   (spatial statistics + FFT radial energy).
2) **Analyse and comment** — explain how object *classes* differ using those domains
   (see REPORT / object_difference.json and ml/frequency_analysis/*.csv).
3) **Extract features first, then ML** — classical models are trained only on the
   stacked feature vectors from (1), not on raw pixels inside this script.
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
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
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
from sklearn.svm import LinearSVC

try:
    from xgboost import XGBClassifier
except ImportError:  # Keep the script usable before optional dependency install.
    XGBClassifier = None


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATA = ROOT / "merged_dataset_v3" / "data.yaml"
DEFAULT_OUT = ROOT / "runs" / "ml" / "feature_ml_analysis"
DEFAULT_DOMAIN_OUT = ROOT / "ml" / "frequency_analysis"

# Combined feature layout: spatial (0..7) then frequency (8..16) — keep in sync with extract_*().
FEATURE_SPATIAL_NAMES = [
    "mean_intensity",
    "std_intensity",
    "p10_intensity",
    "p50_intensity",
    "p90_intensity",
    "grad_mean",
    "grad_std",
    "edge_density",
]
FEATURE_FREQ_NAMES = [
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
N_SPATIAL = len(FEATURE_SPATIAL_NAMES)
N_FREQ = len(FEATURE_FREQ_NAMES)
FEATURE_COLOR_NAMES = (
    [f"hsv_h_hist_{i + 1}" for i in range(16)]
    + [f"hsv_s_hist_{i + 1}" for i in range(8)]
    + [f"hsv_v_hist_{i + 1}" for i in range(8)]
    + [
        "b_mean",
        "g_mean",
        "r_mean",
        "b_std",
        "g_std",
        "r_std",
        "h_mean",
        "s_mean",
        "v_mean",
        "h_std",
        "s_std",
        "v_std",
    ]
)
FEATURE_HOG_NAMES = [f"hog_{i + 1}" for i in range(576)]
N_COLOR = len(FEATURE_COLOR_NAMES)
N_HOG = len(FEATURE_HOG_NAMES)
FEATURE_GROUPS = [
    ("Spatial", FEATURE_SPATIAL_NAMES),
    ("Frequency", FEATURE_FREQ_NAMES),
    ("Color", FEATURE_COLOR_NAMES),
    ("HOG", FEATURE_HOG_NAMES),
]
ALL_FEATURE_NAMES = [name for _, names in FEATURE_GROUPS for name in names]


@dataclass
class Sample:
    feature: np.ndarray
    class_id: int
    class_name: str


def resolve_split_images(ds_root: Path, split_rel: str) -> Path:
    path = Path(split_rel)
    if path.is_absolute():
        return path
    direct = ds_root / path
    if direct.exists():
        return direct
    # Roboflow raw exports may use "../test/images" in a root-level data.yaml.
    if split_rel.startswith("../"):
        fallback = ds_root / split_rel.replace("../", "", 1)
        if fallback.exists():
            return fallback
    return direct


def image_to_label_path(image_path: Path) -> Path:
    # YOLO layout: */images/*.jpg -> */labels/*.txt
    return Path(str(image_path).replace("\\images\\", "\\labels\\")).with_suffix(".txt")


def filter_class_schedule(
    class_names: list[str], exclude_class_names: list[str]
) -> tuple[list[str], dict[int, int]]:
    """Drop excluded classes and map remaining YOLO class ids to 0..K-1."""
    ex = {x.strip().lower() for x in exclude_class_names if x.strip()}
    kept: list[str] = []
    old_to_new: dict[int, int] = {}
    for old_id, name in enumerate(class_names):
        if name.lower() in ex:
            continue
        old_to_new[old_id] = len(kept)
        kept.append(name)
    return kept, old_to_new


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


def extract_color_features(crop_bgr: np.ndarray) -> np.ndarray:
    resized = cv2.resize(crop_bgr, (64, 64), interpolation=cv2.INTER_AREA)
    hsv = cv2.cvtColor(resized, cv2.COLOR_BGR2HSV)

    h_hist = cv2.calcHist([hsv], [0], None, [16], [0, 180]).flatten()
    s_hist = cv2.calcHist([hsv], [1], None, [8], [0, 256]).flatten()
    v_hist = cv2.calcHist([hsv], [2], None, [8], [0, 256]).flatten()
    hist = np.concatenate([h_hist, s_hist, v_hist]).astype(np.float32)
    hist = hist / (float(np.sum(hist)) + 1e-9)

    bgr_mean, bgr_std = cv2.meanStdDev(resized)
    hsv_mean, hsv_std = cv2.meanStdDev(hsv)
    stats = np.concatenate(
        [
            bgr_mean.flatten(),
            bgr_std.flatten(),
            hsv_mean.flatten(),
            hsv_std.flatten(),
        ]
    ).astype(np.float32)
    return np.concatenate([hist, stats], axis=0)


def extract_hog_features(crop_bgr: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, (64, 64), interpolation=cv2.INTER_AREA)
    hog = cv2.HOGDescriptor(
        _winSize=(64, 64),
        _blockSize=(16, 16),
        _blockStride=(16, 16),
        _cellSize=(8, 8),
        _nbins=9,
    )
    return hog.compute(gray).flatten().astype(np.float32)


def extract_combined_features(crop_bgr: np.ndarray) -> np.ndarray:
    spatial = extract_spatial_features(crop_bgr)
    frequency = extract_frequency_features(crop_bgr)
    color = extract_color_features(crop_bgr)
    hog = extract_hog_features(crop_bgr)
    return np.concatenate([spatial, frequency, color, hog], axis=0)


def feature_slice(start_group: int, end_group: int | None = None) -> slice:
    offsets = np.cumsum([0] + [len(names) for _, names in FEATURE_GROUPS])
    if end_group is None:
        end_group = start_group + 1
    return slice(int(offsets[start_group]), int(offsets[end_group]))


def load_samples(
    image_dir: Path,
    kept_class_names: list[str],
    old_to_new: dict[int, int],
    full_nc: int,
    max_objects: int | None,
    max_per_class: int | None = None,
    min_box_px: int = 10,
    seed: int = 42,
) -> list[Sample]:
    image_paths = [
        p
        for p in image_dir.rglob("*")
        if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    ]
    rng = random.Random(seed)
    rng.shuffle(image_paths)

    n_kept = len(kept_class_names)
    counts = [0] * n_kept if max_per_class is not None else None

    samples: list[Sample] = []
    for img_path in image_paths:
        if max_per_class is not None and counts is not None and all(c >= max_per_class for c in counts):
            break

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
            if cls_id < 0 or cls_id >= full_nc:
                continue
            if cls_id not in old_to_new:
                continue
            new_id = old_to_new[cls_id]
            if max_per_class is not None and counts is not None and counts[new_id] >= max_per_class:
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
            samples.append(Sample(feat, new_id, kept_class_names[new_id]))
            if counts is not None:
                counts[new_id] += 1

    if max_objects is None or len(samples) <= max_objects:
        return samples
    return stratified_subsample(samples, max_objects, n_kept, seed=42)


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


def compact_supported_classes(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    class_names: list[str],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[str], dict]:
    """Keep only classes with both train and test samples, then remap to 0..K-1.

    Some external datasets, especially official TACO with rare 60-class labels,
    have classes missing from one split. XGBoost requires contiguous labels, and
    evaluation is not meaningful for a class with no train or no test support.
    """
    train_counts = np.bincount(y_train, minlength=len(class_names))
    test_counts = np.bincount(y_test, minlength=len(class_names))
    keep_old = [
        cid
        for cid in range(len(class_names))
        if int(train_counts[cid]) > 0 and int(test_counts[cid]) > 0
    ]
    dropped = [
        {
            "class_id": cid,
            "class_name": class_names[cid],
            "train_count": int(train_counts[cid]),
            "test_count": int(test_counts[cid]),
        }
        for cid in range(len(class_names))
        if cid not in keep_old
    ]
    if len(keep_old) == len(class_names):
        return x_train, y_train, x_test, y_test, class_names, {"dropped_unsupported_classes": []}
    if len(keep_old) < 2:
        raise SystemExit("Fewer than two classes have both train and test support; ML comparison is not meaningful.")

    old_to_new = {old: new for new, old in enumerate(keep_old)}
    train_mask = np.isin(y_train, keep_old)
    test_mask = np.isin(y_test, keep_old)
    y_train_new = np.array([old_to_new[int(v)] for v in y_train[train_mask]], dtype=np.int64)
    y_test_new = np.array([old_to_new[int(v)] for v in y_test[test_mask]], dtype=np.int64)
    kept_names = [class_names[old] for old in keep_old]
    meta = {
        "dropped_unsupported_classes": dropped,
        "note": "Classes without both train and test support were removed for this ML run and remaining labels were remapped to 0..K-1.",
    }
    return x_train[train_mask], y_train_new, x_test[test_mask], y_test_new, kept_names, meta


def object_difference_report(x_train: np.ndarray, y_train: np.ndarray, class_names: list[str]) -> list[dict]:
    """Per-class deviation from global mean, with named spatial/frequency commentary."""
    assert x_train.shape[1] == len(ALL_FEATURE_NAMES)
    global_mean = np.mean(x_train, axis=0)

    feature_means: dict[int, np.ndarray] = {}
    for cid in np.unique(y_train):
        feature_means[int(cid)] = np.mean(x_train[y_train == cid], axis=0)

    report: list[dict] = []
    for cid, mu in feature_means.items():
        delta = np.abs(mu - global_mean)
        d_sp = delta[feature_slice(0)]
        d_fq = delta[feature_slice(1)]
        top_sp = np.argsort(-d_sp)[:3].tolist()
        top_fq = np.argsort(-d_fq)[:3].tolist()
        spatial_score = float(np.mean(d_sp[top_sp]))
        freq_score = float(np.mean(d_fq[top_fq]))
        top_any = np.argsort(-delta)[:3].tolist()
        overall = float(np.mean(delta[top_any]))

        if spatial_score > freq_score * 1.12:
            dom_phrase = "more in the **spatial** domain than in frequency"
        elif freq_score > spatial_score * 1.12:
            dom_phrase = "more in the **frequency** domain than in spatial texture/intensity"
        else:
            dom_phrase = "in **both spatial and frequency** domains about equally"

        sp_names = ", ".join(FEATURE_SPATIAL_NAMES[i] for i in top_sp)
        fq_names = ", ".join(FEATURE_FREQ_NAMES[i] for i in top_fq)
        lecture_comment = (
            f"Compared to the dataset average, **{class_names[cid]}** differs {dom_phrase}: "
            f"strongest spatial shifts involve `{sp_names}`; "
            f"strongest frequency shifts involve `{fq_names}`."
        )

        report.append(
            {
                "class_id": cid,
                "class_name": class_names[cid],
                "most_distinct_feature_indices": top_any,
                "most_distinct_feature_names": [ALL_FEATURE_NAMES[i] for i in top_any],
                "distinctiveness_score": overall,
                "spatial_top_indices": top_sp,
                "spatial_top_names": [FEATURE_SPATIAL_NAMES[i] for i in top_sp],
                "spatial_distinctiveness": spatial_score,
                "frequency_top_indices": top_fq,
                "frequency_top_names": [FEATURE_FREQ_NAMES[i] for i in top_fq],
                "frequency_distinctiveness": freq_score,
                "lecture_comment": lecture_comment,
            }
        )
    report.sort(key=lambda r: r["distinctiveness_score"], reverse=True)
    return report


def build_models(n_classes: int) -> dict[str, object]:
    models: dict[str, object] = {
        # Why: simple, readable tree baseline requested by lecturer.
        "decision_tree": DecisionTreeClassifier(
            criterion="gini",
            max_depth=24,
            min_samples_leaf=3,
            random_state=42,
        ),
        # Why: strong linear baseline on scaled feature vectors.
        "logreg": Pipeline(
            [("scaler", StandardScaler()), ("clf", LogisticRegression(max_iter=1000, n_jobs=None))]
        ),
        # Why: scalable classical SVM baseline for the larger color+HOG vector.
        "linear_svm": Pipeline([("scaler", StandardScaler()), ("clf", LinearSVC(C=1.0, max_iter=5000, random_state=42))]),
        # Why: tree baseline, robust to mixed feature scales, gives importance.
        "rf": RandomForestClassifier(n_estimators=250, random_state=42, n_jobs=-1),
        # Why: often stronger than RF on high-dimensional handcrafted descriptors.
        "extra_trees": ExtraTreesClassifier(n_estimators=350, random_state=42, n_jobs=-1),
    }
    if XGBClassifier is not None:
        xgb_common = {
            "n_estimators": 300,
            "max_depth": 5,
            "learning_rate": 0.08,
            "subsample": 0.9,
            "colsample_bytree": 0.85,
            "tree_method": "hist",
            "random_state": 42,
            "n_jobs": -1,
        }
        if n_classes == 2:
            models["xgboost"] = XGBClassifier(
                **xgb_common,
                objective="binary:logistic",
                eval_metric="logloss",
            )
        else:
            models["xgboost"] = XGBClassifier(
                **xgb_common,
                objective="multi:softprob",
                eval_metric="mlogloss",
            )
    else:
        print("[WARN] xgboost is not installed; skipping XGBoost. Run: pip install xgboost")
    return models


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
    importances = None
    source = None

    rf_model = trained_models.get("rf")
    if rf_model is not None and hasattr(rf_model, "feature_importances_"):
        importances = np.asarray(rf_model.feature_importances_, dtype=np.float64)
        source = "RandomForest feature_importances_"

    if importances is None:
        raise ValueError("Cannot build domain-importance chart: no model with feature importance is available.")

    group_sums = []
    offset = 0
    for group_name, names in FEATURE_GROUPS:
        next_offset = offset + len(names)
        group_sums.append((group_name, float(np.sum(importances[offset:next_offset]))))
        offset = next_offset
    denom = sum(v for _, v in group_sums) + 1e-12
    group_pcts = [(name, 100.0 * value / denom) for name, value in group_sums]

    fig, ax = plt.subplots(figsize=(6, 4))
    colors = ["#4E79A7", "#F28E2B", "#59A14F", "#E15759"]
    bars = ax.bar([name for name, _ in group_pcts], [pct for _, pct in group_pcts], color=colors[: len(group_pcts)])
    ax.set_title("Feature-group contribution by importance")
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
    domain_importance: dict[str, float] | None,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    spatial_csv = out_dir / "spatial_summary.csv"
    with spatial_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["class_id", "class_name", "n_objects", *FEATURE_SPATIAL_NAMES])
        for cid in range(len(class_names)):
            mask = y_train == cid
            n = int(np.sum(mask))
            if n == 0:
                vals = [0.0] * N_SPATIAL
            else:
                vals = np.mean(x_train[mask, feature_slice(0)], axis=0).tolist()
            w.writerow([cid, class_names[cid], n, *[round(float(v), 6) for v in vals]])

    frequency_csv = out_dir / "frequency_summary.csv"
    with frequency_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["class_id", "class_name", "n_objects", *FEATURE_FREQ_NAMES])
        for cid in range(len(class_names)):
            mask = y_train == cid
            n = int(np.sum(mask))
            if n == 0:
                vals = [0.0] * N_FREQ
            else:
                vals = np.mean(x_train[mask, feature_slice(1)], axis=0).tolist()
            w.writerow([cid, class_names[cid], n, *[round(float(v), 6) for v in vals]])

    color_csv = out_dir / "color_summary.csv"
    with color_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["class_id", "class_name", "n_objects", *FEATURE_COLOR_NAMES])
        for cid in range(len(class_names)):
            mask = y_train == cid
            n = int(np.sum(mask))
            if n == 0:
                vals = [0.0] * N_COLOR
            else:
                vals = np.mean(x_train[mask, feature_slice(2)], axis=0).tolist()
            w.writerow([cid, class_names[cid], n, *[round(float(v), 6) for v in vals]])

    comparison_csv = out_dir / "domain_comparison.csv"
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
                len(ALL_FEATURE_NAMES),
                "",
                (
                    f"Feature vector has {N_SPATIAL} spatial, {N_FREQ} frequency, "
                    f"{N_COLOR} color, and {N_HOG} HOG features"
                ),
            ]
        )
        if domain_importance:
            for group_name, pct in domain_importance.items():
                w.writerow(["model_importance_pct", group_name, round(pct, 4), "From RandomForest feature_importances_"])


def write_report(
    out_dir: Path,
    class_names: list[str],
    train_count: int,
    test_count: int,
    results: list[dict],
    object_diff: list[dict],
    *,
    data_yaml: Path,
    dataset_root: Path,
    yaml_all_classes: list[str],
    excluded_classes: list[str],
) -> None:
    lines: list[str] = []
    lines.append("# Feature + ML Analysis Report\n")

    lines.append("## Dataset and classes (examiner checklist)")
    lines.append(f"- **Dataset:** YOLO-format merged dataset at `{dataset_root}` (config: `{data_yaml}`).")
    lines.append(
        f"- **Classes defined in `data.yaml`:** **{len(yaml_all_classes)}** — "
        f"{', '.join(yaml_all_classes)}."
    )
    if excluded_classes:
        lines.append(
            f"- **Classes used in this ML run:** **{len(class_names)}** — "
            f"{', '.join(class_names)} (excluded from training/eval here: **{', '.join(excluded_classes)}**)."
        )
    else:
        lines.append(f"- **Classes used in this ML run:** **{len(class_names)}** — {', '.join(class_names)}.")
    lines.append(
        "- **Class balance:** The **raw** dataset splits can be **imbalanced** (different sources merged into "
        "`merged_dataset_v3`). When per-class caps are enabled, this script samples **up to N object crops per class** "
        "on train/test so counts are **intentionally balanced at the crop level** where enough boxes exist; "
        "see `class_support.json` for exact train/test counts. Rare classes may still fall **below** the cap."
    )
    lines.append("")

    lines.append("## Handcrafted features used for ML (not raw pixels)")
    lines.append(
        f"Each object crop is resized to 64x64 internally, then summarized into **{len(ALL_FEATURE_NAMES)}** floats."
    )
    lines.append(
        f"- **Why {len(ALL_FEATURE_NAMES)} features?** The vector is the fixed concatenation of "
        f"`{N_SPATIAL}` spatial + `{N_FREQ}` frequency/FFT + `{N_COLOR}` color + `{N_HOG}` HOG features. "
        "This gives a lecturer-explainable representation of brightness/edges, texture frequencies, color distribution, "
        "and local shape/orientation instead of giving raw pixels directly to classical ML."
    )
    lines.append("### Spatial domain (8 features)")
    for i, name in enumerate(FEATURE_SPATIAL_NAMES):
        lines.append(f"{i + 1}. `{name}`")
    lines.append("### Frequency domain (9 features; FFT radial bins + `high_freq_energy`)")
    for i, name in enumerate(FEATURE_FREQ_NAMES):
        lines.append(f"{i + 9}. `{name}`")
    lines.append(f"### Color domain ({N_COLOR} features)")
    lines.append("- HSV histograms plus BGR/HSV mean and standard deviation.")
    lines.append(f"### HOG texture/shape domain ({N_HOG} features)")
    lines.append("- Histogram of Oriented Gradients descriptor from each 64x64 crop.")
    lines.append("")

    lines.append("## Models trained on extracted features")
    lines.append("| Model | Role |")
    lines.append("|---|---|")
    lines.append("| `decision_tree` | **DecisionTreeClassifier** baseline; easiest tree model to explain. |")
    lines.append("| `logreg` | Logistic Regression on **StandardScaler**-normalized features (linear baseline). |")
    lines.append("| `linear_svm` | **Linear SVM** on scaled features, suitable for the larger color+HOG vector. |")
    lines.append("| `extra_trees` | **ExtraTreesClassifier** (350 trees) - stronger high-dimensional classical baseline. |")
    lines.append("| `rf` | **RandomForestClassifier** (250 trees) - tree baseline + feature importance for charts. |")
    if any(r["model"] == "xgboost" for r in results):
        lines.append("| `xgboost` | **XGBoost** gradient-boosted tree baseline requested in lecturer notes. |")
    lines.append("")

    lines.append("## Figures to include in the thesis / lecturer report")
    lines.append("**Classical ML (this folder)**")
    lines.append("- `chart_model_comparison.png` — Accuracy & F1-macro across ML models (no epoch-wise loss; ML is not trained by gradient descent here).")
    lines.append("- `confusion_decision_tree.png`, `confusion_linear_svm.png`, `confusion_rf.png`, `confusion_xgboost.png` - requested ML confusion matrices where available.")
    lines.append("- `classification_reports.json` - precision, recall, F1-score, and support for every class/model.")
    lines.append("- `chart_domain_importance.png` — spatial/frequency/color/HOG contribution (RF feature importance).")
    lines.append("- `ml/frequency_analysis/` — spatial/frequency summary CSVs + optional spectrum plots.")
    lines.append("**Deep learning (separate runs; loss / training curves)**")
    lines.append("- `runs/dl/trash_yolov8n_v3/results.png` — Ultralytics training curves (loss, mAP, precision, recall).")
    lines.append("- `runs/dl/trash_yolov8n_v3/results.csv` — numeric log for custom plots.")
    lines.append("- Run `python scripts/plot_training.py` → writes `training_curves.png` next to the chosen run’s `quality_check/`.")
    lines.append("- `runs/dl/dl_baseline/training_loss.png` — tiny CNN baseline loss vs epoch on crops.")
    lines.append("")

    lines.append("## Scope")
    lines.append("- Mobile is intentionally excluded in this stage.")
    lines.append("- Focus: feature extraction + classical ML model comparison.")
    lines.append(f"- Classes in this run: {', '.join(class_names)}\n")

    lines.append("## Lecture workflow checklist")
    lines.append(
        "1. **Spatial + frequency domains** — each crop gets handcrafted **spatial** statistics "
        "(intensity, gradients, edges) and **frequency** descriptors (2D FFT radial energy bins + high-frequency summary)."
    )
    lines.append(
        "2. **Comment how objects differ** — before judging ML scores, read the per-class notes below and "
        "`ml/frequency_analysis/spatial_summary.csv` + `frequency_summary.csv` (class-wise means)."
    )
    lines.append(
        "3. **Extract features, then ML** — Decision Tree / SVM / RandomForest / XGBoost are trained **only** on the stacked "
        f"{len(ALL_FEATURE_NAMES)}-D feature vectors from step 1 (not raw pixels inside this script).\n"
    )

    lines.append("## Pipeline (implementation order)")
    lines.append("1. Crop objects from YOLO boxes; build the fixed-length feature vector per crop.")
    lines.append("2. Export domain CSVs + object-difference commentary (spatial/frequency/color/HOG).")
    lines.append("3. Fit classical ML on `X_train`; evaluate on `X_test`.\n")

    lines.append("## Data")
    lines.append(f"- Train object crops: **{train_count}**")
    lines.append(f"- Test object crops: **{test_count}**")
    lines.append(
        "- Counts come from a **per-class** cap when enabled (each class can reach the cap independently; "
        "this is not a single 4000-total budget split across classes)."
    )
    lines.append(f"- Classes: {', '.join(class_names)}\n")

    lines.append("## Comments: how object classes differ (feature domains)")
    lines.append(
        "Each bullet compares that class’s **mean feature vector** to the **global mean** over all training crops: "
        "which domain (spatial / frequency / both) shows the largest shift, and which named descriptors move most."
    )
    for item in object_diff:
        lines.append(f"- {item['lecture_comment']}")
        lines.append(
            f"  - *Scores:* overall `{item['distinctiveness_score']:.4f}`, spatial `{item['spatial_distinctiveness']:.4f}`, "
            f"frequency `{item['frequency_distinctiveness']:.4f}`."
        )
    lines.append("")

    lines.append("## ML results (features → models)")
    lines.append("| Model | Accuracy | F1-macro |")
    lines.append("|---|---:|---:|")
    for r in results:
        lines.append(f"| {r['model']} | {r['accuracy']:.4f} | {r['f1_macro']:.4f} |")
    lines.append("")

    lines.append("## Model choice rationale")
    lines.append("- **Decision Tree:** easiest baseline to explain; shows whether simple feature thresholds can separate classes.")
    lines.append("- **Logistic Regression:** simple linear baseline on standardized feature vectors.")
    lines.append("- **Linear SVM:** scalable margin-based baseline for high-dimensional handcrafted descriptors.")
    lines.append("- **Random Forest:** robust tree-based baseline and interpretable feature importance.")
    lines.append("- **ExtraTrees:** tree ensemble baseline that often improves over RF on noisy, high-dimensional features.\n")
    if any(r["model"] == "xgboost" for r in results):
        lines.append("- **XGBoost:** boosted tree ensemble that tests whether sequential error correction improves the handcrafted-feature baseline.\n")

    lines.append("## Chart comments")
    lines.append(
        "- `chart_domain_importance.png`: compares spatial, frequency, color, and HOG contribution based on model feature "
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
    parser.add_argument(
        "--exclude-classes",
        type=str,
        default="other",
        help="Comma-separated names to drop from data.yaml (e.g. other). Empty string = keep all.",
    )
    parser.add_argument(
        "--max-per-class-train",
        type=int,
        default=4000,
        help="Max object crops PER CLASS on train (each class independently; NOT a 4k total cap). 0 = off, use --max-train-objects.",
    )
    parser.add_argument(
        "--max-per-class-test",
        type=int,
        default=4000,
        help="Max object crops PER CLASS on test (each class independently, not total). 0 = off, use --max-test-objects.",
    )
    parser.add_argument("--max-train-objects", type=int, default=8000)
    parser.add_argument("--max-test-objects", type=int, default=3000)
    parser.add_argument("--domain-out", type=Path, default=DEFAULT_DOMAIN_OUT)
    args = parser.parse_args()

    cfg = yaml.safe_load(args.data.read_text(encoding="utf-8"))
    class_names = list(cfg["names"])
    full_nc = len(class_names)
    ds_root = Path(cfg.get("path", args.data.parent))

    exclude_list = [s.strip() for s in args.exclude_classes.split(",") if s.strip()]
    kept_names, old_to_new = filter_class_schedule(class_names, exclude_list)
    if not kept_names:
        raise SystemExit("All classes excluded; fix --exclude-classes.")

    train_img_dir = resolve_split_images(ds_root, cfg["train"])
    test_img_dir = resolve_split_images(ds_root, cfg.get("test", cfg["val"]))

    args.out.mkdir(parents=True, exist_ok=True)

    max_per_train = args.max_per_class_train if args.max_per_class_train > 0 else None
    max_per_test = args.max_per_class_test if args.max_per_class_test > 0 else None
    train_max_obj = None if max_per_train is not None else args.max_train_objects
    test_max_obj = None if max_per_test is not None else args.max_test_objects

    print(
        f"Using {len(kept_names)} classes {kept_names}"
        + (f" (excluded: {exclude_list})" if exclude_list else "")
    )
    print("Sampling caps are per class (e.g. 4000 plastic + 4000 glass + ...), not a single global 4k limit.")
    if max_per_train:
        print(f"Train: up to {max_per_train} object crops per class.")
    if max_per_test:
        print(f"Test: up to {max_per_test} object crops per class.")

    print("[1/5] Extracting train features (spatial + frequency + color + HOG domains)...")
    train_samples = load_samples(
        train_img_dir,
        kept_names,
        old_to_new,
        full_nc,
        train_max_obj,
        max_per_class=max_per_train,
    )
    if not train_samples:
        raise SystemExit(f"No train samples loaded from {train_img_dir}")
    x_train, y_train = to_matrix(train_samples)

    print("[2/5] Analysis: how object classes differ (train features vs global mean)...")
    object_diff = object_difference_report(x_train, y_train, kept_names)
    (args.out / "object_difference.json").write_text(json.dumps(object_diff, indent=2), encoding="utf-8")

    print("[3/5] Extracting test features...")
    test_samples = load_samples(
        test_img_dir,
        kept_names,
        old_to_new,
        full_nc,
        test_max_obj,
        max_per_class=max_per_test,
    )
    if not test_samples:
        raise SystemExit(f"No test samples loaded from {test_img_dir}")
    x_test, y_test = to_matrix(test_samples)

    x_train, y_train, x_test, y_test, kept_names, support_meta = compact_supported_classes(
        x_train, y_train, x_test, y_test, kept_names
    )

    k = len(kept_names)
    class_support = {
        "train": {kept_names[i]: int(v) for i, v in enumerate(np.bincount(y_train, minlength=k))},
        "test": {kept_names[i]: int(v) for i, v in enumerate(np.bincount(y_test, minlength=k))},
    }
    class_support.update(support_meta)
    (args.out / "class_support.json").write_text(json.dumps(class_support, indent=2), encoding="utf-8")

    models = build_models(k)
    results: list[dict] = []
    detailed: dict[str, dict] = {}
    trained_models: dict[str, object] = {}

    print("[4/5] Training ML models on extracted features only...")
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
            labels=list(range(k)),
            target_names=kept_names,
            output_dict=True,
            zero_division=0,
        )
        save_confusion(y_test, pred, kept_names, args.out / f"confusion_{name}.png", f"Confusion matrix - {name}")

    results.sort(key=lambda r: (r["f1_macro"], r["accuracy"]), reverse=True)
    (args.out / "metrics_summary.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    (args.out / "classification_reports.json").write_text(json.dumps(detailed, indent=2), encoding="utf-8")

    print("[5/5] Saving charts, domain CSVs, and report...")
    save_comparison_chart(results, args.out / "chart_model_comparison.png")
    save_domain_importance_chart(trained_models, args.out / "chart_domain_importance.png")
    rf_importances = np.asarray(trained_models["rf"].feature_importances_, dtype=np.float64)
    group_sums: dict[str, float] = {}
    offset = 0
    for group_name, names in FEATURE_GROUPS:
        next_offset = offset + len(names)
        group_sums[group_name] = float(np.sum(rf_importances[offset:next_offset]))
        offset = next_offset
    denom = sum(group_sums.values()) + 1e-12
    domain_importance = {group_name: 100.0 * value / denom for group_name, value in group_sums.items()}
    export_domain_summaries(x_train, y_train, kept_names, args.domain_out, domain_importance)
    write_report(
        args.out,
        kept_names,
        len(train_samples),
        len(test_samples),
        results,
        object_diff,
        data_yaml=args.data.resolve(),
        dataset_root=ds_root.resolve(),
        yaml_all_classes=list(cfg["names"]),
        excluded_classes=exclude_list,
    )

    print(f"[OK] Done. See {args.out} and {args.domain_out}")


if __name__ == "__main__":
    main()

