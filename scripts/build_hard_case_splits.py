"""Build source-aware split recommendations for full hard-case datasets."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_IN = ROOT / "external_datasets" / "hard_case_full" / "combined_manifest.csv"
DEFAULT_OUT = ROOT / "external_datasets" / "hard_case_full" / "source_aware_split_manifest.csv"
MATERIALS = {"Plastic", "Glass", "Metal", "Paper", "Cardboard", "Organic"}


def stable_bucket(value: str) -> int:
    digest = hashlib.md5(value.encode("utf-8")).hexdigest()
    return int(digest[:8], 16) % 100


def taco_batch(source_file: str) -> str:
    if "/" in source_file:
        return source_file.split("/", 1)[0]
    if "\\" in source_file:
        return source_file.split("\\", 1)[0]
    return "unknown_batch"


def split_realwaste(row: dict[str, str]) -> tuple[str, str, bool, bool, bool]:
    target = row.get("target_class", "")
    if target not in MATERIALS:
        return "review_pool", "RealWaste label outside current taxonomy", False, False, True

    key = f"realwaste:{target}:{row.get('source_file', '')}"
    bucket = stable_bucket(key)
    if bucket < 70:
        split = "classifier_train_candidate"
    elif bucket < 85:
        split = "classifier_hard_val"
    else:
        split = "classifier_hard_test"
    return split, "RealWaste material classification source", True, False, False


def split_taco(row: dict[str, str]) -> tuple[str, str, bool, bool, bool]:
    status = row.get("download_status", "")
    if status not in {"exists", "downloaded", "downloaded_fallback_640"}:
        return "taco_missing_image", "TACO image unavailable", False, False, True

    group = taco_batch(row.get("source_file", ""))
    if group in {"batch_12", "batch_13"}:
        split = "yolo_hard_val"
    elif group in {"batch_14", "batch_15"}:
        split = "yolo_hard_test"
    else:
        split = "yolo_train_candidate"

    target = row.get("target_class", "")
    needs_audit = "review" in target.split("|") or status == "downloaded_fallback_640"
    reason = "TACO batch-aware localization source"
    if status == "downloaded_fallback_640":
        reason += "; fallback image needs annotation scale check"
    return split, reason, False, True, needs_audit


def split_outerview(row: dict[str, str]) -> tuple[str, str, bool, bool, bool]:
    if row.get("download_status") == "image_found":
        return "ood_review_pool", "Outerview has images but material labels need visual audit", False, False, True
    return "metadata_only_pool", "Outerview metadata row without extracted image match", False, False, True


def assign(row: dict[str, str]) -> dict[str, Any]:
    source = row.get("source_id", "")
    if source == "realwaste_hf_uci":
        split, reason, classifier, yolo, audit = split_realwaste(row)
    elif source == "taco_official":
        split, reason, classifier, yolo, audit = split_taco(row)
    elif source == "outerview_global_trash_debris":
        split, reason, classifier, yolo, audit = split_outerview(row)
    else:
        split, reason, classifier, yolo, audit = "unknown_source_pool", "unknown source", False, False, True

    row = dict(row)
    row["recommended_split"] = split
    row["split_reason"] = reason
    row["usable_for_classifier"] = "yes" if classifier else "no"
    row["usable_for_yolo"] = "yes" if yolo else "no"
    row["needs_visual_audit"] = "yes" if audit else "no"
    row["final_decision"] = "pending_audit" if audit else "candidate"
    return row


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_IN)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    with args.input.open("r", encoding="utf-8", newline="") as handle:
        rows = [assign(row) for row in csv.DictReader(handle)]

    write_csv(args.output, rows)

    summary = {
        "records": len(rows),
        "by_source": dict(Counter(row["source_id"] for row in rows)),
        "by_recommended_split": dict(Counter(row["recommended_split"] for row in rows)),
        "by_classifier_usable": dict(Counter(row["usable_for_classifier"] for row in rows)),
        "by_yolo_usable": dict(Counter(row["usable_for_yolo"] for row in rows)),
        "needs_visual_audit": dict(Counter(row["needs_visual_audit"] for row in rows)),
        "output": str(args.output.relative_to(ROOT)),
    }
    summary_path = args.output.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=True), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
