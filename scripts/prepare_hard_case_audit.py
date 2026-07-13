"""Prepare a small trusted hard-case audit set.

This script intentionally downloads a small sample first. It is not a full
dataset merger. The output is meant for visual audit, label mapping checks, and
source-aware validation planning before retraining.
"""

from __future__ import annotations

import argparse
import csv
import html
import json
import random
import re
import time
from collections import defaultdict
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import quote
from urllib.request import Request, urlopen

try:
    from PIL import Image
except Exception:  # pragma: no cover - optional validation only
    Image = None


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = ROOT / "external_datasets" / "hard_case_audit"

REALWASTE_REPO = "shahzaibvohra/realwaste"
REALWASTE_INFO_URL = (
    "https://datasets-server.huggingface.co/info?"
    "dataset=shahzaibvohra%2Frealwaste"
)
OUTERVIEW_REPO = "Outerview/global-trash-and-debris-index"
OUTERVIEW_CSV = "1775001918223-Trash___Debris_Dataset.csv"

REALWASTE_LABEL_MAP = {
    "Cardboard": ("Cardboard", "direct"),
    "Food Organics": ("Organic", "direct"),
    "Glass": ("Glass", "direct"),
    "Metal": ("Metal", "direct"),
    "Miscellaneous Trash": ("review", "mixed-or-unknown-material"),
    "Paper": ("Paper", "direct"),
    "Plastic": ("Plastic", "direct"),
    "Textile Trash": ("review", "class-outside-current-taxonomy"),
    "Vegetation": ("Organic", "verify-as-organic-waste"),
}


def fetch_json(url: str, timeout: int = 60) -> dict[str, Any]:
    req = Request(url, headers={"User-Agent": "WasteWise-hard-case-audit/1.0"})
    with urlopen(req, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def fetch_bytes(url: str, timeout: int = 90, retries: int = 2) -> bytes:
    last_error: Exception | None = None
    for attempt in range(retries + 1):
        try:
            req = Request(url, headers={"User-Agent": "WasteWise-hard-case-audit/1.0"})
            with urlopen(req, timeout=timeout) as response:
                return response.read()
        except (HTTPError, URLError, TimeoutError) as exc:
            last_error = exc
            if attempt < retries:
                time.sleep(1.5 * (attempt + 1))
    raise RuntimeError(f"download failed: {url}") from last_error


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=True), encoding="utf-8")


def parse_hf_file_ref(ref: str) -> tuple[str, str]:
    match = re.match(r"hf://datasets/[^@]+@([^/]+)/(.+)$", ref)
    if not match:
        raise ValueError(f"unexpected Hugging Face file ref: {ref}")
    return match.group(1), match.group(2)


def label_from_realwaste_filename(filename: str) -> str:
    stem = Path(filename).stem
    match = re.match(r"(.+)_\d+$", stem)
    return match.group(1) if match else stem


def realwaste_download_url(filename: str, revision: str) -> str:
    encoded = quote(filename, safe="/")
    return f"https://huggingface.co/datasets/{REALWASTE_REPO}/resolve/{revision}/{encoded}"


def validate_image(path: Path) -> dict[str, Any]:
    if Image is None:
        return {"validated": False, "width": None, "height": None, "error": "PIL unavailable"}
    try:
        with Image.open(path) as image:
            image.verify()
        with Image.open(path) as image:
            width, height = image.size
        return {"validated": True, "width": width, "height": height, "error": ""}
    except Exception as exc:  # pragma: no cover - depends on downloaded data
        return {"validated": False, "width": None, "height": None, "error": str(exc)}


def prepare_realwaste(out_dir: Path, per_class: int, seed: int) -> list[dict[str, Any]]:
    info = fetch_json(REALWASTE_INFO_URL)
    checksums = info["dataset_info"]["default"]["download_checksums"]

    by_label: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for ref in checksums:
        revision, filename = parse_hf_file_ref(ref)
        label = label_from_realwaste_filename(filename)
        if filename.lower().endswith((".jpg", ".jpeg", ".png", ".webp")):
            by_label[label].append((revision, filename))

    rng = random.Random(seed)
    records: list[dict[str, Any]] = []
    images_dir = out_dir / "realwaste_hf_uci" / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    for source_label in sorted(by_label):
        candidates = by_label[source_label]
        rng.shuffle(candidates)
        target_class, audit_note = REALWASTE_LABEL_MAP.get(
            source_label, ("review", "unmapped-source-label")
        )
        for idx, (revision, filename) in enumerate(candidates[:per_class], start=1):
            suffix = Path(filename).suffix.lower() or ".jpg"
            safe_label = re.sub(r"[^A-Za-z0-9]+", "_", source_label).strip("_")
            image_name = f"{safe_label}_{idx:03d}{suffix}"
            image_path = images_dir / image_name
            status = "downloaded"
            error = ""
            if not image_path.exists():
                try:
                    image_path.write_bytes(fetch_bytes(realwaste_download_url(filename, revision)))
                except Exception as exc:
                    status = "failed"
                    error = str(exc)

            image_check = validate_image(image_path) if image_path.exists() else {}
            records.append(
                {
                    "source_id": "realwaste_hf_uci",
                    "source_name": "RealWaste",
                    "source_repo": REALWASTE_REPO,
                    "source_file": filename,
                    "source_label": source_label,
                    "target_class": target_class,
                    "audit_bucket": "review" if target_class == "review" else "train_candidate",
                    "audit_note": audit_note,
                    "image_path": str(image_path.relative_to(ROOT)) if image_path.exists() else "",
                    "metadata_path": "",
                    "download_status": status,
                    "download_error": error,
                    "width": image_check.get("width"),
                    "height": image_check.get("height"),
                    "validated_image": image_check.get("validated", False),
                    "license": "CC-BY-4.0 per registry/source card; verify before redistribution",
                }
            )
    return records


def prepare_outerview_metadata(out_dir: Path, rows: int, seed: int) -> list[dict[str, Any]]:
    source_dir = out_dir / "outerview_global_trash_debris"
    source_dir.mkdir(parents=True, exist_ok=True)
    csv_path = source_dir / OUTERVIEW_CSV
    if not csv_path.exists():
        url = f"https://huggingface.co/datasets/{OUTERVIEW_REPO}/resolve/main/{OUTERVIEW_CSV}"
        csv_path.write_bytes(fetch_bytes(url, timeout=120))

    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = list(csv.DictReader(handle))

    rng = random.Random(seed)
    sample = reader[:]
    rng.shuffle(sample)
    sample = sample[:rows]

    sample_path = source_dir / "metadata_sample.csv"
    if sample:
        with sample_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(sample[0].keys()))
            writer.writeheader()
            writer.writerows(sample)

    records: list[dict[str, Any]] = []
    for idx, row in enumerate(sample, start=1):
        label = (
            row.get("label")
            or row.get("class")
            or row.get("category")
            or row.get("object")
            or "unknown"
        )
        records.append(
            {
                "source_id": "outerview_global_trash_debris",
                "source_name": "Outerview Global Trash & Debris Index",
                "source_repo": OUTERVIEW_REPO,
                "source_file": OUTERVIEW_CSV,
                "source_label": label,
                "target_class": "review",
                "audit_bucket": "metadata_audit",
                "audit_note": "metadata-only sample; dataset-server image rows returned 500, so verify labels before image download",
                "image_path": "",
                "metadata_path": str(sample_path.relative_to(ROOT)),
                "download_status": "metadata_sampled",
                "download_error": "",
                "width": None,
                "height": None,
                "validated_image": False,
                "license": "CC-BY-4.0 per registry/source card; verify before redistribution",
                "metadata_row": idx,
            }
        )
    return records


def write_manifests(out_dir: Path, records: list[dict[str, Any]]) -> None:
    manifest_csv = out_dir / "manifest.csv"
    manifest_jsonl = out_dir / "manifest.jsonl"
    out_dir.mkdir(parents=True, exist_ok=True)

    audit_defaults = {
        "audit_decision": "pending",
        "corrected_class": "",
        "is_waste": "",
        "bbox_required": "",
        "final_split": "hard_case_candidate",
        "auditor_notes": "",
    }
    records = [{**audit_defaults, **record} for record in records]

    fieldnames = sorted({key for record in records for key in record})
    with manifest_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)

    with manifest_jsonl.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=True) + "\n")

    by_source: dict[str, int] = defaultdict(int)
    by_target: dict[str, int] = defaultdict(int)
    by_status: dict[str, int] = defaultdict(int)
    for record in records:
        by_source[record["source_id"]] += 1
        by_target[record["target_class"]] += 1
        by_status[record["download_status"]] += 1

    summary = {
        "record_count": len(records),
        "by_source": dict(sorted(by_source.items())),
        "by_target_class": dict(sorted(by_target.items())),
        "by_download_status": dict(sorted(by_status.items())),
        "manifest_csv": str(manifest_csv.relative_to(ROOT)),
        "manifest_jsonl": str(manifest_jsonl.relative_to(ROOT)),
    }
    write_json(out_dir / "source_summary.json", summary)

    readme = out_dir / "README.md"
    readme.write_text(
        "# Hard-Case Audit Sample\n\n"
        "This folder contains a small trusted-source sample for visual audit before retraining.\n\n"
        "- `manifest.csv`: spreadsheet-friendly audit manifest.\n"
        "- `manifest.jsonl`: script-friendly audit manifest.\n"
        "- `source_summary.json`: counts by source, target class, and status.\n\n"
        "Audit columns are included for `corrected_class`, `is_waste`, bounding-box need,\n"
        "final split, and reviewer notes.\n\n"
        "Open `review_gallery.html` to quickly inspect downloaded image samples.\n\n"
        "Do not merge these images directly into training until labels are visually checked and\n"
        "the final split is source-aware.\n",
        encoding="utf-8",
    )
    write_review_gallery(out_dir, records)


def write_review_gallery(out_dir: Path, records: list[dict[str, Any]]) -> None:
    cards: list[str] = []
    for record in records:
        image_path = record.get("image_path") or ""
        image_src = html.escape(image_path.replace("\\", "/"))
        image_html = (
            f'<img src="../../{image_src}" alt="">'
            if image_path
            else '<div class="no-image">metadata only</div>'
        )
        cards.append(
            "<article>"
            f"{image_html}"
            f"<strong>{html.escape(str(record.get('target_class', '')))}</strong>"
            f"<span>{html.escape(str(record.get('source_label', '')))}</span>"
            f"<small>{html.escape(str(record.get('source_id', '')))} / "
            f"{html.escape(str(record.get('audit_note', '')))}</small>"
            "</article>"
        )

    gallery = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>WasteWise Hard-Case Audit Gallery</title>
  <style>
    body { margin: 0; font-family: Arial, sans-serif; background: #f6f3ec; color: #151515; }
    header { position: sticky; top: 0; z-index: 1; padding: 18px 22px; background: #151515; color: #fff; }
    h1 { margin: 0 0 4px; font-size: 22px; }
    p { margin: 0; color: #d9d4c8; }
    main { display: grid; grid-template-columns: repeat(auto-fill, minmax(180px, 1fr)); gap: 12px; padding: 16px; }
    article { min-height: 250px; border: 1px solid #d8d1c4; background: #fff; padding: 10px; display: grid; gap: 8px; align-content: start; }
    img, .no-image { width: 100%; aspect-ratio: 1; object-fit: cover; background: #ece6da; display: grid; place-items: center; color: #756d60; }
    strong { font-size: 17px; }
    span { color: #5d554b; }
    small { color: #80776b; line-height: 1.35; }
  </style>
</head>
<body>
  <header>
    <h1>WasteWise Hard-Case Audit Gallery</h1>
    <p>Visual check first. Do not merge into training until labels and source split are verified.</p>
  </header>
  <main>
    __CARDS__
  </main>
</body>
</html>
"""
    (out_dir / "review_gallery.html").write_text(
        gallery.replace("__CARDS__", "\n    ".join(cards)),
        encoding="utf-8",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--realwaste-per-class", type=int, default=8)
    parser.add_argument("--outerview-rows", type=int, default=100)
    parser.add_argument(
        "--sources",
        nargs="+",
        default=["realwaste", "outerview"],
        choices=["realwaste", "outerview"],
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = args.out_dir.resolve()
    records: list[dict[str, Any]] = []

    if "realwaste" in args.sources:
        records.extend(prepare_realwaste(out_dir, args.realwaste_per_class, args.seed))

    if "outerview" in args.sources:
        records.extend(prepare_outerview_metadata(out_dir, args.outerview_rows, args.seed))

    write_manifests(out_dir, records)
    print(f"records: {len(records)}")
    print(f"output: {out_dir}")
    print(f"manifest: {out_dir / 'manifest.csv'}")


if __name__ == "__main__":
    main()
