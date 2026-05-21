"""Summarize external waste dataset candidates for the report.

This does not download data. It keeps the dataset-selection stage reproducible:
the report table is generated from a JSON registry so every candidate has a
source URL, format, risk, and project decision before any merge happens.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_REGISTRY = ROOT / "docs" / "02_dataset_training" / "external_dataset_registry.json"
DEFAULT_OUT = ROOT / "runs" / "external_dataset_registry" / "DATASET_CANDIDATE_SUMMARY.md"


def fmt(value: object) -> str:
    if value is None:
        return "TBD"
    return str(value)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    rows = json.loads(args.registry.read_text(encoding="utf-8"))
    args.out.parent.mkdir(parents=True, exist_ok=True)

    lines = [
        "# External Dataset Candidate Summary",
        "",
        "Generated from `docs/02_dataset_training/external_dataset_registry.json`.",
        "",
        "| Dataset | Format | Classes | Images | TACO Similarity | Decision |",
        "|---|---|---:|---:|---|---|",
    ]
    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    f"[{row['name']}]({row['source_url']})",
                    fmt(row["annotation_format"]),
                    fmt(row["classes"]),
                    fmt(row["images"]),
                    fmt(row["similar_to_taco"]),
                    fmt(row["action"]),
                ]
            )
            + " |"
        )

    lines.extend(["", "## Risks"])
    for row in rows:
        lines.append(f"- **{row['name']}**: {row['risk']}")

    args.out.write_text("\n".join(lines), encoding="utf-8")
    print(f"[OK] Wrote {args.out}")


if __name__ == "__main__":
    main()
