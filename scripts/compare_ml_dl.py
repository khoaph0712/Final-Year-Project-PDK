"""Create a unified ML-vs-DL comparison report and chart."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_ML = ROOT / "runs" / "feature_ml_analysis" / "metrics_summary.json"
DEFAULT_DL = ROOT / "runs" / "dl_baseline" / "metrics.json"
DEFAULT_OUT = ROOT / "runs" / "model_comparison"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ml-metrics", type=Path, default=DEFAULT_ML)
    parser.add_argument("--dl-metrics", type=Path, default=DEFAULT_DL)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    ml_metrics = json.loads(args.ml_metrics.read_text(encoding="utf-8"))
    dl_metrics = json.loads(args.dl_metrics.read_text(encoding="utf-8"))
    args.out.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    for m in ml_metrics:
        rows.append({"model": m["model"], "accuracy": float(m["accuracy"]), "f1_macro": float(m["f1_macro"]), "family": "ML"})
    rows.append(
        {
            "model": dl_metrics.get("model", "tiny_cnn"),
            "accuracy": float(dl_metrics["accuracy"]),
            "f1_macro": float(dl_metrics["f1_macro"]),
            "family": "DL",
        }
    )
    rows.sort(key=lambda r: (r["f1_macro"], r["accuracy"]), reverse=True)
    (args.out / "comparison_metrics.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")

    labels = [r["model"] for r in rows]
    acc = [r["accuracy"] for r in rows]
    f1m = [r["f1_macro"] for r in rows]

    x = np.arange(len(labels))
    w = 0.36
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(x - w / 2, acc, width=w, label="Accuracy")
    ax.bar(x + w / 2, f1m, width=w, label="F1-macro")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0.0, 1.0)
    ax.set_title("Model comparison: ML baselines vs DL baseline")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(args.out / "chart_ml_vs_dl.png", dpi=140)
    plt.close(fig)

    lines: list[str] = []
    lines.append("# ML vs DL Comparison Report\n")
    lines.append("## Scope")
    lines.append("- ML-first analysis is kept as primary direction.")
    lines.append("- A lightweight DL model (tiny CNN) is added as a comparison baseline.\n")

    lines.append("## Why these models")
    lines.append("- **LogReg:** linear baseline on handcrafted features.")
    lines.append("- **SVM-RBF:** non-linear boundary baseline.")
    lines.append("- **RandomForest:** tree baseline with feature importance.")
    lines.append("- **Tiny CNN:** compact end-to-end DL comparator.\n")

    lines.append("## Results")
    lines.append("| Model | Family | Accuracy | F1-macro |")
    lines.append("|---|---|---:|---:|")
    for r in rows:
        lines.append(f"| {r['model']} | {r['family']} | {r['accuracy']:.4f} | {r['f1_macro']:.4f} |")
    lines.append("")

    best = rows[0]
    lines.append("## Conclusion")
    lines.append(
        f"- Best by F1-macro in this run: **{best['model']}** ({best['family']}) "
        f"with Accuracy `{best['accuracy']:.4f}` and F1-macro `{best['f1_macro']:.4f}`."
    )
    lines.append("- Keep ML-first pipeline for explainability and lecture requirements; DL is used for benchmark comparison.\n")

    lines.append("## Artifacts")
    lines.append("- `chart_ml_vs_dl.png`")
    lines.append("- `comparison_metrics.json`")
    lines.append("- `runs/feature_ml_analysis/*`")
    lines.append("- `runs/dl_baseline/*`")

    (args.out / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"[OK] Done. See {args.out}")


if __name__ == "__main__":
    main()
