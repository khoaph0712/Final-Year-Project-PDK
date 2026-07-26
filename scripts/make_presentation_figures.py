#!/usr/bin/env python
"""Build the confusion-matrix and heatmap figures used by the FYP presentation site.

Everything here is replotted from JSON/CSV already committed under runs/, so the
figures can be restyled without re-running any model. Outputs land directly in
web/assets/figures/.

Sources
  runs/audits/classifier_domain_gap.json ......... deployed ConvNeXt confusion matrices
  runs/audits/CLASSIFIER_CLEAN_TEST_EVAL.md ..... (context only, not parsed)
  runs/audits/convnext_clean_eval.json .......... per-class precision/recall/F1
  runs/audits/detector_conf_sweep_field_6class.json  confidence-threshold sweep
  runs/dl/cross_dataset_validation/cross_dataset_results.json  domain-transfer matrix
  runs/comparisons/phase2_model_comparison.csv .. model x dataset accuracy
"""

import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

ROOT = Path(__file__).resolve().parent.parent
AUDITS = ROOT / "runs" / "audits"
WEB = ROOT / "web" / "assets" / "figures"

# Matches the site accent ramp (--accent #047857 / --accent-bright #34d399).
EMERALD = LinearSegmentedColormap.from_list(
    "emerald", ["#ffffff", "#e7f6ef", "#a7ddc5", "#34d399", "#0e9f6e", "#047857", "#04372a"]
)

CLASSES = ["Plastic", "Glass", "Metal", "Paper", "Cardboard", "Organic", "Background"]

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 9,
    "axes.titlesize": 11,
    "axes.titleweight": "bold",
    "axes.labelsize": 9.5,
    "figure.facecolor": "white",
    "savefig.facecolor": "white",
})


def save(fig, name):
    out = WEB / name
    fig.savefig(out, dpi=170, bbox_inches="tight", pad_inches=0.22)
    plt.close(fig)
    print(f"[fig] {name}  ({out.stat().st_size / 1024:.0f} KB)")


def annotate(ax, values, labels, fontsize=8):
    """Write one label per cell, picking ink colour from the cell's own luminance
    rather than a fixed cutoff, so mid-tone greens stay readable either way."""
    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            r, g, b, _ = EMERALD(float(np.clip(values[i, j], 0, 1)))
            lum = 0.2126 * r + 0.7152 * g + 0.0722 * b
            ax.text(j, i, labels[i][j], ha="center", va="center", fontsize=fontsize,
                    color="#1b1b18" if lum > 0.45 else "#ffffff")


def heat_axes(ax, xticks, yticks, xlabel="", ylabel=""):
    ax.set_xticks(range(len(xticks)), xticks, rotation=30, ha="right")
    ax.set_yticks(range(len(yticks)), yticks)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_xticks(np.arange(-0.5, len(xticks), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(yticks), 1), minor=True)
    ax.grid(which="minor", color="#ffffff", linewidth=1.4)
    ax.tick_params(which="minor", length=0)
    for side in ax.spines.values():
        side.set_visible(False)


def confusion_panel(ax, cm, title):
    """Row-normalized confusion matrix annotated with percentage and raw count."""
    cm = np.asarray(cm, dtype=float)
    norm = cm / cm.sum(axis=1, keepdims=True)
    ax.imshow(norm, cmap=EMERALD, vmin=0, vmax=1)
    labels = [[f"{norm[i, j] * 100:.0f}%\n{int(cm[i, j])}" if cm[i, j] else "-"
               for j in range(cm.shape[1])] for i in range(cm.shape[0])]
    annotate(ax, norm, labels)
    heat_axes(ax, CLASSES, CLASSES, "Predicted class", "True class")
    acc = np.trace(cm) / cm.sum()
    ax.set_title(f"{title}\naccuracy {acc * 100:.1f}%  ({int(cm.sum()):,} crops)")


# --------------------------------------------------------------- fig: classifier CMs

def classifier_confusions():
    gap = json.loads((AUDITS / "classifier_domain_gap.json").read_text(encoding="utf-8"))
    studio = np.array(gap["domains"]["studio"]["confusion_matrix"], dtype=float)
    field = np.array(gap["domains"]["field"]["confusion_matrix"], dtype=float)

    # Overall clean-test confusion: the two domains partition the same test split.
    fig, ax = plt.subplots(figsize=(7.4, 6.2))
    confusion_panel(ax, studio + field, "Deployed classifier (ConvNeXt-Tiny + 637 features)")
    save(fig, "cm_convnext_clean_test.png")

    # Domain attribution per the verified lineage audit (paper section 4.5):
    # studio bucket = Kaggle Garbage Classification crops, field = Roboflow exports.
    studio_acc = gap["domains"]["studio"]["accuracy"]
    field_acc = gap["domains"]["field"]["accuracy"]
    fig, axes = plt.subplots(1, 2, figsize=(13.6, 6.0))
    confusion_panel(axes[0], studio,
                    f"Studio crops - Kaggle Garbage Classification (n={int(studio.sum())}, {studio_acc:.1%})")
    confusion_panel(axes[1], field,
                    f"Field crops - Roboflow detection exports (n={int(field.sum())}, {field_acc:.1%})")
    fig.suptitle("Same deployed classifier, same test split, split by acquisition domain",
                 fontsize=11.5, fontweight="bold", y=1.02)
    fig.tight_layout()
    save(fig, "cm_convnext_domains.png")


# --------------------------------------------------------- fig: per-class metric heatmap

def per_class_heatmap():
    ev = json.loads((AUDITS / "convnext_clean_eval.json").read_text(encoding="utf-8"))
    report = ev["splits"]["test"]["classification_report"]
    keys = ["plastic", "glass", "metal", "paper", "cardboard", "organic", "Background"]
    metrics = ["precision", "recall", "f1-score"]
    values = np.array([[report[k][m] for m in metrics] for k in keys])
    support = [int(report[k]["support"]) for k in keys]

    fig, ax = plt.subplots(figsize=(6.4, 5.6))
    ax.imshow(values, cmap=EMERALD, vmin=0.75, vmax=1.0)
    labels = [[f"{v:.3f}" for v in row] for row in values]
    annotate(ax, (values - 0.75) / 0.25, labels)
    rows = [f"{c}  (n={n})" for c, n in zip(CLASSES, support)]
    heat_axes(ax, ["Precision", "Recall", "F1"], rows)
    ax.set_title("Per-class scores, deployed classifier\n"
                 f"clean hard-case test split, {ev['splits']['test']['samples']:,} crops, "
                 f"accuracy {ev['splits']['test']['accuracy'] * 100:.2f}%")
    save(fig, "heat_classifier_per_class.png")


# ------------------------------------------------------------ fig: domain-transfer matrix

def domain_transfer_heatmap():
    src = ROOT / "runs" / "dl" / "cross_dataset_validation" / "cross_dataset_results.json"
    rows = json.loads(src.read_text(encoding="utf-8"))
    label = {"studio": "Studio\n(TrashNet / Kaggle)", "field": "Field\n(Roboflow / TACO)"}
    order = ["studio", "field"]

    grid = np.zeros((2, 2))
    for r in rows:
        tr, te = order.index(r["train_domain"]), order.index(r["test_domain"])
        grid[tr, tr] = r["in_domain_accuracy"]
        grid[tr, te] = r["cross_domain_accuracy"]

    fig, ax = plt.subplots(figsize=(6.0, 4.8))
    ax.imshow(grid, cmap=EMERALD, vmin=0.3, vmax=0.85)
    labels = [[f"{grid[i, j] * 100:.1f}%\n{'in-domain' if i == j else 'cross-domain'}"
               for j in range(2)] for i in range(2)]
    annotate(ax, (grid - 0.3) / 0.55, labels)
    heat_axes(ax, [label[o] for o in order], [label[o] for o in order],
              "Tested on", "Trained on")
    gaps = {r["train_domain"]: r["generalization_gap_pp"] for r in rows}
    ax.set_title("Domain transfer: accuracy when the test domain is unseen\n"
                 f"drop of {gaps['studio']:.1f} pp studio→field, "
                 f"{gaps['field']:.1f} pp field→studio")
    save(fig, "heat_domain_transfer.png")


# ------------------------------------------------------- fig: model x dataset accuracy

def model_dataset_heatmap():
    src = ROOT / "runs" / "comparisons" / "phase2_model_comparison.csv"
    rows = list(csv.DictReader(src.open(encoding="utf-8")))

    datasets = ["merged_6class", "trashnet", "gini_binary", "taco_official_partial"]
    ds_label = ["Merged\n(6 class)", "TrashNet\n(5 class)", "GINI\n(binary)", "TACO\n(39 class)"]
    models = ["xgboost", "extra_trees", "rf", "linear_svm", "logreg", "decision_tree", "cnn", "ann"]
    m_label = ["XGBoost", "ExtraTrees", "Random Forest", "Linear SVM", "LogReg",
               "Decision Tree", "CNN (scratch)", "ANN (scratch)"]

    lookup = {(r["dataset"], r["model"]): float(r["accuracy"]) for r in rows}
    grid = np.array([[lookup.get((d, m), np.nan) for d in datasets] for m in models])

    fig, ax = plt.subplots(figsize=(6.6, 6.2))
    ax.imshow(grid, cmap=EMERALD, vmin=0, vmax=1)
    labels = [[f"{v * 100:.1f}%" if not np.isnan(v) else "n/a" for v in row] for row in grid]
    annotate(ax, np.nan_to_num(grid), labels)
    heat_axes(ax, ds_label, m_label, "Benchmark dataset", "Model")
    ax.set_title("Accuracy of every baseline on every benchmark\n"
                 "one grid: class count and domain matter more than model family")
    save(fig, "heat_model_dataset.png")


# ------------------------------------------------------------ fig: confidence sweep

def conf_sweep_heatmap():
    sweep = json.loads((AUDITS / "detector_conf_sweep_field_6class.json").read_text(encoding="utf-8"))
    thresholds = ["0.10", "0.07", "0.05", "0.04"]
    series = [
        ("Field recall", "taco_field_test_recall"),
        ("Field small-box recall", "taco_field_test_small_box_recall"),
        ("Studio recall", "studio_clean_val_recall"),
        ("Studio small-box recall", "studio_clean_val_small_box_recall"),
    ]
    grid = np.array([[sweep[key].get(t, np.nan) for t in thresholds] for _, key in series])

    fig, ax = plt.subplots(figsize=(6.6, 4.2))
    ax.imshow(grid, cmap=EMERALD, vmin=0.25, vmax=0.7)
    labels = [[f"{v * 100:.1f}%" if not np.isnan(v) else "not run" for v in row] for row in grid]
    annotate(ax, np.nan_to_num((grid - 0.25) / 0.45), labels)
    heat_axes(ax, thresholds, [name for name, _ in series],
              "Detector confidence threshold", "")
    ax.set_title("Lowering the confidence threshold lifts recall in both domains\n"
                 "no studio/field trade-off along this lever")
    save(fig, "heat_detector_conf_sweep.png")


def main():
    WEB.mkdir(parents=True, exist_ok=True)
    classifier_confusions()
    per_class_heatmap()
    domain_transfer_heatmap()
    model_dataset_heatmap()
    conf_sweep_heatmap()


if __name__ == "__main__":
    main()
