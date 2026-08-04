"""Generate the presentation figures the written report has but the site was missing.

Two of the report's results figures had no counterpart on the presentation site:

  * Figure 16 - leakage flagged in each evaluation split at Hamming thresholds 0-4,
    the sensitivity check behind the 43% headline. Recomputed here from the audit
    manifest (runs/audits/model_risk_audit.json), which stores the Hamming distance
    of every flagged pair, so the sweep needs no re-hashing. The headline split
    reproduces the report exactly: 39.1% at distance 0 rising to 43.0% (2,406 of
    5,600 images) at distance 4.
  * Figure 34 - the end-to-end decision-layer ablation on the identical 1,042-crop
    set, read straight from runs/audits/pipeline_bin_decision_eval_*.json.

One existing figure is rebuilt rather than copied. cmp_ml_vs_dl.png was titled "ML
baselines vs DL baseline" but plotted no deep model at all, and omitted XGBoost - the
strongest classical model - entirely. It is regenerated here as the report's Figure 24
actually describes it: the classical branch against the deep branch on one axis, every
bar labelled with the split it was measured on.

Three further figures already existed as run artefacts and are copied rather than
regenerated, so the site shows the same pixels the report does:

  * cmp_classical_models.png      <- the six-class benchmark of report Table 12
  * cm_xgboost.png               <- report Figure 21 (XGBoost is the strongest
                                    classical model; the site previously showed
                                    ExtraTrees, which its own ranking chart
                                    contradicted)
  * ml_domain_importance_report.png <- report Figure 22, the feature-domain split
                                    on the report's Table 9 decomposition
                                    (spatial / frequency / colour / HOG)

Run with the project venv:
    C:/FYP/.venv311/Scripts/python.exe scripts/make_report_alignment_figures.py
"""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
REPO = Path("C:/FYP")  # run artefacts live outside the worktree
WEB = ROOT / "web" / "assets" / "figures"
AUDITS = REPO / "runs" / "audits"
LECTURER_RUN = REPO / "runs" / "ml" / "feature_ml_lecturer_6class_4k"

# Emerald accent, matching the site's own palette.
GREEN = "#047857"
GREEN_LIGHT = "#34d399"
INK = "#0f172a"
GREY = "#94a3b8"

plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "axes.edgecolor": "#cbd5e1",
        "axes.labelcolor": INK,
        "text.color": INK,
        "xtick.color": INK,
        "ytick.color": INK,
        "axes.grid": True,
        "grid.color": "#e2e8f0",
        "grid.linewidth": 0.7,
    }
)


def save(fig, name: str, dpi: int = 165) -> None:
    WEB.mkdir(parents=True, exist_ok=True)
    out = WEB / name
    fig.savefig(out, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  wrote {out.name} ({out.stat().st_size / 1024:.0f} KB)")


# --------------------------------------------------------------- Figure 16


# Split sizes as built, matching report Table 8. The classifier split is 7 classes
# x 800 crops; the detector splits are the pre-quarantine image counts.
SWEEP_SPLITS = [
    ("merged", "test", 5600, "merged_dataset_v5 test\n(classifier headline split)", GREEN),
    ("hardcase", "val", 3450, "yolo26_hardcase_dataset_v1 val", GREEN_LIGHT),
    ("hardcase", "test", 1275, "yolo26_hardcase_dataset_v1 test", "#0891b2"),
    ("hardcase_classifier", "test", 2696, "hard_case_classifier_v1 test", "#7c3aed"),
]


def leakage_sweep(audit: dict, dataset: str, split: str, n: int) -> list[float]:
    """Percent of eval images whose nearest training neighbour is within t bits."""
    leak = audit["datasets"][dataset]["leakage"]
    marker = os.sep + split + os.sep
    exact: set[str] = set()
    for pair in leak.get("exact", []):
        for side in ("a", "b"):
            if marker in pair[side]:
                exact.add(pair[side])
    nearest: dict[str, int] = {}
    for pair in leak.get("near", []):
        sides = pair["splits"].split("<->")
        if split not in sides or "train" not in sides:
            continue
        for side in ("a", "b"):
            if marker in pair[side]:
                nearest[pair[side]] = min(nearest.get(pair[side], 99), pair["hamming"])
    out = []
    for t in range(5):
        flagged = exact | {k for k, v in nearest.items() if v <= t}
        out.append(100.0 * len(flagged) / n)
    return out


def fig_leakage_threshold() -> None:
    audit = json.loads((AUDITS / "model_risk_audit.json").read_text(encoding="utf-8"))
    fig, ax = plt.subplots(figsize=(8.4, 4.6))
    for dataset, split, n, label, colour in SWEEP_SPLITS:
        pct = leakage_sweep(audit, dataset, split, n)
        ax.plot(range(5), pct, marker="o", lw=2.2, color=colour, label=label)
        ax.annotate(
            f"{pct[0]:.1f}%",
            (0, pct[0]),
            textcoords="offset points",
            xytext=(4, -13),
            fontsize=9,
            color=colour,
        )
        ax.annotate(
            f"{pct[4]:.1f}%",
            (4, pct[4]),
            textcoords="offset points",
            xytext=(7, -3),
            fontsize=9,
            fontweight="bold",
            color=colour,
        )
    ax.axvline(4, color=GREY, ls="--", lw=1.1)
    ax.set_xlim(-0.3, 4.75)
    ax.set_xticks(range(5))
    ax.set_xlabel("Hamming distance threshold (bits out of 64)   —   dashed line: the audit's 4-bit threshold")
    ax.set_ylabel("Share of evaluation split flagged (%)")
    ax.set_title("Leakage flagged per evaluation split, thresholds 0 to 4", fontsize=12)
    ax.legend(
        fontsize=8.5,
        loc="upper left",
        bbox_to_anchor=(1.01, 1.0),
        borderaxespad=0.0,
        framealpha=0.95,
    )
    fig.tight_layout()
    save(fig, "eda_leakage_threshold.png")


# --------------------------------------------------------------- Figure 34

# Report Figure 34: the gated baseline that ran in production until 2026-07-18, the
# deployed ungated configuration, and the two configurations measured and rejected.
ABLATION_ORDER = [
    ("baseline", "baseline\n(gated, ex-production)"),
    ("no_gate", "no_gate\n(DEPLOYED)"),
    ("margin20", "margin20\n(rejected)"),
    ("no_prior_damping", "no_prior_damping\n(rejected)"),
]
ABLATION_METRICS = [
    ("materialAccuracy", "Material accuracy", GREEN),
    ("binAccuracy", "Bin-routing accuracy", GREEN_LIGHT),
    ("wasteStateAccuracy", "Waste-state accuracy", "#0891b2"),
    ("secondaryMaterialRecall", "Secondary-material recall", "#7c3aed"),
]


def fig_decision_ablation() -> None:
    runs = []
    for tag, label in ABLATION_ORDER:
        path = AUDITS / f"pipeline_bin_decision_eval_{tag}.json"
        runs.append((label, json.loads(path.read_text(encoding="utf-8"))))
    n = runs[0][1]["n"]

    fig, ax = plt.subplots(figsize=(9.2, 4.8))
    width = 0.2
    for i, (key, label, colour) in enumerate(ABLATION_METRICS):
        xs = [j + (i - 1.5) * width for j in range(len(runs))]
        ys = [r[key] * 100 for _, r in runs]
        bars = ax.bar(xs, ys, width * 0.92, label=label, color=colour)
        for bar, y in zip(bars, ys):
            ax.annotate(
                f"{y:.1f}",
                (bar.get_x() + bar.get_width() / 2, y),
                textcoords="offset points",
                xytext=(0, 3),
                ha="center",
                fontsize=7.8,
            )
    # The bin metric is close to degenerate on this set: five of six materials route
    # to Recycling, so a constant "Recycling" predictor scores 99.6%.
    ax.axhline(99.6, color="#b45309", ls=":", lw=1.4)
    ax.text(
        len(runs) - 0.55,
        99.9,
        "constant-\"Recycling\" baseline 99.6%",
        ha="right",
        fontsize=8.5,
        color="#b45309",
    )
    ax.set_xticks(range(len(runs)))
    ax.set_xticklabels([label for label, _ in runs], fontsize=9)
    ax.set_ylim(40, 108)
    ax.set_ylabel("Score (%)")
    ax.set_title(
        f"Decision-layer ablations on the identical {n:,}-crop evaluation set", fontsize=12
    )
    ax.legend(
        fontsize=8.5,
        ncol=4,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.13),
        frameon=False,
    )
    fig.tight_layout()
    save(fig, "cmp_decision_ablation.png")


# --------------------------------------------------------------- Figure 24


def fig_ml_vs_dl() -> None:
    classical = json.loads((LECTURER_RUN / "metrics_summary.json").read_text(encoding="utf-8"))
    nice = {
        "xgboost": "XGBoost",
        "extra_trees": "ExtraTrees",
        "rf": "Random Forest",
        "linear_svm": "Linear SVM",
        "logreg": "Logistic Reg.",
        "decision_tree": "Decision Tree",
    }
    tiny = json.loads((REPO / "runs" / "dl" / "dl_baseline" / "metrics.json").read_text(encoding="utf-8"))
    torch_cmp = json.loads(
        (ROOT / "web" / "assets" / "data" / "new_models.json").read_text(encoding="utf-8")
    )["torch_comparison"]

    # (label, accuracy %, family) - family drives the colour and the legend grouping.
    rows = [(nice[m["model"]], m["accuracy"] * 100, "classical") for m in classical]
    rows.append(("Tiny CNN\n(from scratch)", tiny["accuracy"] * 100, "scratch"))
    rows += [
        ("MobileNetV2", torch_cmp["mobilenetv2"]["accuracy"] * 100, "deep"),
        ("ResNet50", torch_cmp["resnet50"]["accuracy"] * 100, "deep"),
        ("EfficientNetB0", 94.29, "deep"),
        ("ConvNeXt-Tiny\n+ 637 (DEPLOYED)", 93.77, "deployed"),
    ]
    palette = {
        "classical": GREEN_LIGHT,
        "scratch": "#f59e0b",
        "deep": "#0891b2",
        "deployed": GREEN,
    }
    legend = {
        "classical": "Classical, 637 handcrafted features (six-class benchmark)",
        "scratch": "Deep, no pretraining (balanced crop benchmark)",
        "deep": "Deep, ImageNet-pretrained (balanced crop benchmark, pre-audit)",
        "deployed": "Deployed hybrid (leakage-quarantined clean test)",
    }

    fig, ax = plt.subplots(figsize=(10.4, 5.0))
    xs = range(len(rows))
    bars = ax.bar(
        xs, [r[1] for r in rows], 0.68, color=[palette[r[2]] for r in rows]
    )
    for bar, (_, acc, _) in zip(bars, rows):
        ax.annotate(
            f"{acc:.1f}",
            (bar.get_x() + bar.get_width() / 2, acc),
            textcoords="offset points",
            xytext=(0, 4),
            ha="center",
            fontsize=9,
            fontweight="bold",
        )
    ax.set_xticks(list(xs))
    ax.set_xticklabels([r[0] for r in rows], fontsize=8.5, rotation=18, ha="right")
    ax.set_ylim(0, 108)
    ax.set_ylabel("Accuracy (%)")
    ax.set_title(
        "Classical branch against deep branch: every classifier trained in the project",
        fontsize=12,
    )
    handles = [
        plt.Rectangle((0, 0), 1, 1, color=palette[k]) for k in ("classical", "scratch", "deep", "deployed")
    ]
    ax.legend(
        handles,
        [legend[k] for k in ("classical", "scratch", "deep", "deployed")],
        fontsize=8,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.28),
        ncol=2,
        frameon=False,
    )
    fig.tight_layout()
    save(fig, "cmp_ml_vs_dl.png")


# --------------------------------------------------------------- copies

COPY_JOBS = [
    # The site's own ranking chart already put XGBoost first, but it was drawn from a
    # different run than report Table 12. Use the Table 12 run for both.
    (LECTURER_RUN / "chart_model_comparison.png", "cmp_classical_models.png"),
    (LECTURER_RUN / "confusion_xgboost.png", "cm_xgboost.png"),
    (LECTURER_RUN / "chart_domain_importance.png", "ml_domain_importance_report.png"),
]


def copy_run_figures() -> None:
    for src, dest in COPY_JOBS:
        if not src.exists():
            raise SystemExit(f"missing run artefact: {src}")
        shutil.copyfile(src, WEB / dest)
        print(f"  copied {dest} ({(WEB / dest).stat().st_size / 1024:.0f} KB)")


def main() -> None:
    print("Figure 16 - leakage threshold sweep")
    fig_leakage_threshold()
    print("Figure 24 - classical branch vs deep branch")
    fig_ml_vs_dl()
    print("Figure 34 - decision-layer ablation")
    fig_decision_ablation()
    print("Run-artefact copies")
    copy_run_figures()


if __name__ == "__main__":
    main()
