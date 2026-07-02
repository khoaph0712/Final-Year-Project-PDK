"""Controlled PCA sweep on cached 637-D handcrafted features.

The older PCA artifact trains a small MLP and can move with random
initialization. This script keeps the same cached 637-feature vectors, runs
repeatable classical models, and reports the measured trade-off for reducing
637 features to PCA spaces such as 128 components.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier

try:
    from xgboost import XGBClassifier
except ImportError:
    XGBClassifier = None


ROOT = Path(__file__).resolve().parent.parent
ARCHIVE_DIR = ROOT / "scripts" / "archive"
sys.path.append(str(ARCHIVE_DIR))
from ml_balanced_training import load_crops_and_balance  # noqa: E402


DEFAULT_CACHE_DIR = ROOT / "runs" / "dl" / "convnext_ensemble"
DEFAULT_OUT_DIR = ROOT / "runs" / "ml" / "pca_feature_model_sweep"
DEFAULT_DATA_YAML = ROOT / "data" / "merged_dataset_v5" / "data.yaml"
TARGET_CLASSES = ["plastic", "glass", "metal", "paper", "cardboard", "organic", "Background"]


def load_features_and_labels(cache_dir: Path, data_yaml: Path, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    x_train = np.load(cache_dir / "train_handcrafted_637.npy")
    x_test = np.load(cache_dir / "test_handcrafted_637.npy")

    _, y_train_list = load_crops_and_balance(
        data_yaml,
        TARGET_CLASSES,
        max_per_class=1000,
        is_train=True,
        seed=seed,
    )
    _, y_test_list = load_crops_and_balance(
        data_yaml,
        TARGET_CLASSES,
        max_per_class=300,
        is_train=False,
        seed=seed,
    )
    y_train = np.asarray(y_train_list, dtype=np.int64)
    y_test = np.asarray(y_test_list, dtype=np.int64)

    if len(x_train) != len(y_train) or len(x_test) != len(y_test):
        raise ValueError(
            "Feature/label length mismatch: "
            f"x_train={x_train.shape}, y_train={y_train.shape}, "
            f"x_test={x_test.shape}, y_test={y_test.shape}"
        )
    return x_train, y_train, x_test, y_test


def build_models(seed: int) -> dict[str, object]:
    models: dict[str, object] = {
        "decision_tree": DecisionTreeClassifier(max_depth=24, random_state=seed),
        "logreg": Pipeline(
            [
                ("scale", StandardScaler()),
                (
                    "clf",
                    LogisticRegression(
                        max_iter=3000,
                        class_weight="balanced",
                        random_state=seed,
                        n_jobs=-1,
                    ),
                ),
            ]
        ),
        "linear_svm": Pipeline(
            [
                ("scale", StandardScaler()),
                ("clf", LinearSVC(C=1.0, class_weight="balanced", max_iter=8000, random_state=seed)),
            ]
        ),
        "rf": RandomForestClassifier(
            n_estimators=220,
            max_depth=None,
            class_weight="balanced_subsample",
            random_state=seed,
            n_jobs=-1,
        ),
        "extra_trees": ExtraTreesClassifier(
            n_estimators=260,
            max_depth=None,
            class_weight="balanced",
            random_state=seed,
            n_jobs=-1,
        ),
    }
    if XGBClassifier is not None:
        models["xgboost"] = XGBClassifier(
            n_estimators=180,
            max_depth=5,
            learning_rate=0.08,
            subsample=0.85,
            colsample_bytree=0.85,
            objective="multi:softprob",
            eval_metric="mlogloss",
            random_state=seed,
            n_jobs=-1,
            tree_method="hist",
        )
    return models


def fit_transform_dimension(
    x_train: np.ndarray,
    x_test: np.ndarray,
    components: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, float]:
    if components >= x_train.shape[1]:
        return x_train, x_test, 100.0

    pca = PCA(n_components=components, random_state=seed)
    x_train_pca = pca.fit_transform(x_train)
    x_test_pca = pca.transform(x_test)
    explained_variance = float(np.sum(pca.explained_variance_ratio_) * 100)
    return x_train_pca, x_test_pca, explained_variance


def evaluate_models(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    components: int,
    explained_variance: float,
    seed: int,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for model_name, model in build_models(seed).items():
        start = time.time()
        model.fit(x_train, y_train)
        train_time = time.time() - start

        pred_start = time.time()
        y_pred = model.predict(x_test)
        latency_ms = (time.time() - pred_start) / len(y_test) * 1000

        rows.append(
            {
                "components": components,
                "explained_variance": explained_variance,
                "model": model_name,
                "accuracy": float(accuracy_score(y_test, y_pred)),
                "f1_macro": float(f1_score(y_test, y_pred, average="macro")),
                "f1_weighted": float(f1_score(y_test, y_pred, average="weighted")),
                "train_time_s": train_time,
                "latency_ms": latency_ms,
            }
        )
    return rows


def summarize_best(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    full_by_model = {r["model"]: r for r in rows if r["components"] == 637}
    summary: list[dict[str, object]] = []
    for row in rows:
        full = full_by_model.get(row["model"])
        if full is None:
            continue
        row = dict(row)
        row["accuracy_drop_pp"] = (full["accuracy"] - row["accuracy"]) * 100
        row["f1_macro_drop_pp"] = (full["f1_macro"] - row["f1_macro"]) * 100
        summary.append(row)
    return summary


def best_per_component(summary: list[dict[str, object]], components: list[int]) -> dict[int, dict[str, object]]:
    """Highest-accuracy model at each component count (the honest ceiling per dim)."""
    best: dict[int, dict[str, object]] = {}
    for comp in components:
        rows = [r for r in summary if r["components"] == comp]
        if rows:
            best[comp] = max(rows, key=lambda r: r["accuracy"])
    return best


def find_knee(best: dict[int, dict[str, object]], full_dim: int, tol_pp: float) -> dict[str, object] | None:
    """Smallest component count whose best model stays within tol_pp of the full-637 best."""
    if full_dim not in best:
        return None
    ceiling = best[full_dim]["accuracy"]
    for comp in sorted(c for c in best if c < full_dim):
        if (ceiling - best[comp]["accuracy"]) * 100 <= tol_pp:
            return best[comp]
    return None


def write_report(out_dir: Path, rows: list[dict[str, object]], components: list[int]) -> None:
    summary = summarize_best(rows)
    report = out_dir / "PCA_Model_Sweep_Report.md"

    full_dim = max(components)
    best = best_per_component(summary, components)
    ceiling_row = best.get(full_dim)
    knee_1pp = find_knee(best, full_dim, tol_pp=1.0)
    knee_2pp = find_knee(best, full_dim, tol_pp=2.0)

    lines: list[str] = [
        "# PCA 637-Feature Model Sweep",
        "",
        "This run compares the original 637 handcrafted features against PCA-reduced feature spaces using repeatable classical models. The cached features come from `runs/dl/convnext_ensemble`, with labels resolved from `data/merged_dataset_v5`.",
        "",
        f"- Train samples: 7,000",
        f"- Test samples: 2,100",
        f"- Classes: {', '.join(TARGET_CLASSES)}",
        f"- PCA dimensions tested: {', '.join(str(c) for c in sorted(components))}",
        "",
    ]

    if ceiling_row:
        lines.extend(
            [
                "## Headline",
                "",
                f"- **Full {full_dim}-D ceiling:** `{ceiling_row['model']}` at "
                f"{ceiling_row['accuracy'] * 100:.2f}% accuracy (macro-F1 {ceiling_row['f1_macro']:.4f}).",
            ]
        )
        if knee_1pp:
            lines.append(
                f"- **Best reduction within 1pp:** {knee_1pp['components']} components -> "
                f"`{knee_1pp['model']}` {knee_1pp['accuracy'] * 100:.2f}% "
                f"({knee_1pp['accuracy_drop_pp']:.2f}pp drop, "
                f"{knee_1pp['explained_variance']:.2f}% variance retained)."
            )
        if knee_2pp and (not knee_1pp or knee_2pp["components"] < knee_1pp["components"]):
            lines.append(
                f"- **Best reduction within 2pp:** {knee_2pp['components']} components -> "
                f"`{knee_2pp['model']}` {knee_2pp['accuracy'] * 100:.2f}% "
                f"({knee_2pp['accuracy_drop_pp']:.2f}pp drop, "
                f"{knee_2pp['explained_variance']:.2f}% variance retained)."
            )
        lines.append("")

    lines.extend(
        [
            "## Best model at each dimension",
            "",
            "| Components | Explained variance | Best model | Accuracy | Drop vs full |",
            "|---:|---:|---|---:|---:|",
        ]
    )
    for comp in sorted(best):
        r = best[comp]
        lines.append(
            f"| {comp} | {r['explained_variance']:.2f}% | {r['model']} | "
            f"{r['accuracy'] * 100:.2f}% | {r['accuracy_drop_pp']:.2f} pp |"
        )

    lines.extend(
        [
            "",
            "## Full Results",
            "",
            "| Components | Explained variance | Model | Accuracy | F1-macro | F1-weighted | Accuracy drop vs full | Latency |",
            "|---:|---:|---|---:|---:|---:|---:|---:|",
        ]
    )
    for row in sorted(summary, key=lambda r: (r["components"], r["model"])):
        lines.append(
            f"| {row['components']} | {row['explained_variance']:.2f}% | {row['model']} | "
            f"{row['accuracy'] * 100:.2f}% | {row['f1_macro']:.4f} | {row['f1_weighted']:.4f} | "
            f"{row['accuracy_drop_pp']:.2f} pp | {row['latency_ms']:.4f} ms |"
        )

    lines.extend(
        [
            "",
            "## Reporting Guidance",
            "",
            "- Cite the **best model per dimension** table, not a single cherry-picked model, when discussing the PCA trade-off.",
            "- The knee figures above are the honest cost of reduction: smallest dimension whose *best* classifier stays within 1pp / 2pp of the full-feature ceiling.",
            "- Keep the older `runs/dl/pca_experiments/PCA_Dimensionality_Report.md` as ANN-only evidence.",
            "",
        ]
    )

    report.write_text("\n".join(lines), encoding="utf-8")
    (out_dir / "pca_model_sweep_results.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


def write_chart(out_dir: Path, rows: list[dict[str, object]]) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[WARN] matplotlib not installed; skipping chart (report + JSON still written).")
        return
    summary = summarize_best(rows)
    models = sorted({r["model"] for r in summary})
    comps = sorted({r["components"] for r in summary})

    fig, ax = plt.subplots(figsize=(10, 5.5))
    for model in models:
        model_rows = [r for r in summary if r["model"] == model]
        model_rows.sort(key=lambda r: r["components"])
        ax.plot(
            [r["components"] for r in model_rows],
            [r["accuracy"] * 100 for r in model_rows],
            marker="o",
            linewidth=1.8,
            label=model,
        )

    ax.set_title("PCA Components vs Accuracy by Model")
    ax.set_xlabel("PCA components")
    ax.set_ylabel("Accuracy (%)")
    ax.set_xticks(comps)
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / "pca_model_sweep_accuracy.png", dpi=180)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA_YAML)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--components", type=int, nargs="+", default=[16, 32, 64, 96, 128, 192, 256, 384, 637])
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    x_train, y_train, x_test, y_test = load_features_and_labels(args.cache_dir, args.data, args.seed)
    rows: list[dict[str, object]] = []

    for components in args.components:
        print(f"[PCA] Preparing {components} components")
        x_train_dim, x_test_dim, explained = fit_transform_dimension(x_train, x_test, components, args.seed)
        print(f"[PCA] Explained variance: {explained:.2f}%")
        rows.extend(
            evaluate_models(
                x_train_dim,
                y_train,
                x_test_dim,
                y_test,
                components,
                explained,
                args.seed,
            )
        )

    write_report(args.out, rows, args.components)
    write_chart(args.out, rows)
    print(f"[OK] PCA model sweep written to {args.out}")


if __name__ == "__main__":
    main()
