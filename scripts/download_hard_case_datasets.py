"""Download trusted hard-case datasets for WasteWise.

This is the heavy-data counterpart to prepare_hard_case_audit.py. It downloads
full source datasets into external_datasets/, keeps source-specific manifests,
and avoids changing tracked training data until a later audit/split step.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import shutil
import time
import zipfile
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import quote
from urllib.request import Request, urlopen

try:
    from PIL import Image
except Exception:  # pragma: no cover
    Image = None


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = ROOT / "external_datasets" / "hard_case_full"
CHUNK = 8 * 1024 * 1024

REALWASTE_REPO = "shahzaibvohra/realwaste"
REALWASTE_INFO_URL = (
    "https://datasets-server.huggingface.co/info?"
    "dataset=shahzaibvohra%2Frealwaste"
)

TACO_RAW = "https://raw.githubusercontent.com/pedropro/TACO/master/data"
TACO_FILES = {
    "annotations.json": f"{TACO_RAW}/annotations.json",
    "annotations_unofficial.json": f"{TACO_RAW}/annotations_unofficial.json",
    "all_image_urls.csv": f"{TACO_RAW}/all_image_urls.csv",
}

OUTERVIEW_REPO = "Outerview/global-trash-and-debris-index"
OUTERVIEW_BASE = f"https://huggingface.co/datasets/{OUTERVIEW_REPO}/resolve/main"
OUTERVIEW_FILES = {
    "1775001918223-Trash___Debris_Dataset.csv": 3_576_102,
    "1775001918985-Trash___Debris_Dataset.geojson": 9_851_110,
    "1775001920664-Trash___Debris_Dataset.parquet": 3_385_712,
    "1775002681200-Trash___Debris_Dataset-image.zip": 1_073_068_085,
    "README.md": 4_147,
}

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

TACO_MATERIAL_KEYWORDS = {
    "Plastic": [
        "plastic",
        "bottle cap",
        "straw",
        "lid",
        "wrapper",
        "sachet",
        "rope",
        "styrofoam",
    ],
    "Glass": ["glass"],
    "Metal": ["aluminium", "aluminum", "metal", "can", "aerosol"],
    "Paper": ["paper", "magazine", "newspaper", "receipt", "paper bag"],
    "Cardboard": ["carton", "cardboard", "corrugated"],
    "Organic": ["food waste"],
}


def request(url: str, headers: dict[str, str] | None = None) -> Request:
    final_headers = {"User-Agent": "WasteWise-hard-case-downloader/1.0"}
    if headers:
        final_headers.update(headers)
    return Request(url, headers=final_headers)


def fetch_json(url: str, timeout: int = 90) -> dict[str, Any]:
    with urlopen(request(url), timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def download_file(url: str, target: Path, expected_size: int | None = None) -> str:
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() and target.stat().st_size > 0:
        if expected_size is None or target.stat().st_size == expected_size:
            return "exists"

    part = target.with_suffix(target.suffix + ".part")
    existing = part.stat().st_size if part.exists() else 0
    headers = {"Range": f"bytes={existing}-"} if existing else None

    for attempt in range(4):
        try:
            with urlopen(request(url, headers=headers), timeout=180) as response:
                status = getattr(response, "status", None)
                mode = "ab" if existing and status == 206 else "wb"
                if mode == "wb":
                    existing = 0
                total = expected_size or response.headers.get("Content-Length") or "unknown"
                print(f"download {target.name}: start={existing} total={total}")
                downloaded = existing
                last_log = time.time()
                with part.open(mode + "") as handle:
                    while True:
                        chunk = response.read(CHUNK)
                        if not chunk:
                            break
                        handle.write(chunk)
                        downloaded += len(chunk)
                        now = time.time()
                        if now - last_log > 10:
                            print(f"download {target.name}: {downloaded / 1024 / 1024:.1f} MB")
                            last_log = now
            part.replace(target)
            return "downloaded"
        except (HTTPError, URLError, TimeoutError, OSError) as exc:
            if attempt == 3:
                raise RuntimeError(f"failed to download {url}: {exc}") from exc
            time.sleep(2 * (attempt + 1))
            existing = part.stat().st_size if part.exists() else 0
            headers = {"Range": f"bytes={existing}-"} if existing else None
    return "failed"


def validate_image(path: Path) -> tuple[bool, int | None, int | None, str]:
    if Image is None:
        return False, None, None, "PIL unavailable"
    try:
        with Image.open(path) as image:
            image.verify()
        with Image.open(path) as image:
            width, height = image.size
        return True, width, height, ""
    except Exception as exc:  # pragma: no cover
        return False, None, None, str(exc)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


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


def download_realwaste(out_dir: Path, validate: bool, workers: int) -> dict[str, Any]:
    print("source realwaste: metadata")
    info = fetch_json(REALWASTE_INFO_URL)
    checksums = info["dataset_info"]["default"]["download_checksums"]
    source_dir = out_dir / "realwaste_hf_uci"
    rows: list[dict[str, Any]] = []

    tasks: list[tuple[str, dict[str, Any]]] = []
    for ref, meta in checksums.items():
        revision, filename = parse_hf_file_ref(ref)
        if not filename.lower().endswith((".jpg", ".jpeg", ".png", ".webp")):
            continue
        tasks.append((ref, meta))

    def one(task: tuple[str, dict[str, Any]]) -> dict[str, Any]:
        ref, meta = task
        revision, filename = parse_hf_file_ref(ref)
        source_label = label_from_realwaste_filename(filename)
        target_class, note = REALWASTE_LABEL_MAP.get(source_label, ("review", "unmapped-source-label"))
        target = source_dir / "images" / source_label / Path(filename).name
        try:
            status = download_file(
                realwaste_download_url(filename, revision),
                target,
                expected_size=meta.get("num_bytes"),
            )
            error = ""
        except Exception as exc:
            status = "failed"
            error = str(exc)
        ok, width, height, image_error = validate_image(target) if validate and target.exists() else (False, None, None, "")
        return {
            "source_id": "realwaste_hf_uci",
            "source_file": filename,
            "source_label": source_label,
            "target_class": target_class,
            "audit_note": note,
            "image_path": (
                str(target.relative_to(ROOT)) if target.exists() and ROOT in target.resolve().parents else
                str(target) if target.exists() else ""
            ),
            "download_status": status,
            "download_error": error or image_error,
            "width": width,
            "height": height,
            "validated_image": ok,
        }

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(one, task) for task in tasks]
        for index, future in enumerate(as_completed(futures), start=1):
            rows.append(future.result())
            if index % 100 == 0 or index == len(futures):
                print(f"source realwaste: {index}/{len(futures)} files processed")

    write_csv(source_dir / "manifest.csv", rows)
    return summarize("realwaste_hf_uci", rows)


def taco_class_to_material(name: str) -> tuple[str, str]:
    lower = name.lower()
    for material, keywords in TACO_MATERIAL_KEYWORDS.items():
        if any(keyword in lower for keyword in keywords):
            return material, "keyword-map"
    return "review", "manual-map-needed"


def download_taco(out_dir: Path, validate: bool, workers: int) -> dict[str, Any]:
    print("source taco: metadata")
    source_dir = out_dir / "taco_official"
    data_dir = source_dir / "data"
    for filename, url in TACO_FILES.items():
        download_file(url, data_dir / filename)

    annotations = json.loads((data_dir / "annotations.json").read_text(encoding="utf-8"))
    categories = {item["id"]: item["name"] for item in annotations["categories"]}
    anns_by_image: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for ann in annotations["annotations"]:
        anns_by_image[ann["image_id"]].append(ann)

    material_map = {
        str(category_id): {
            "source_label": label,
            "target_class": taco_class_to_material(label)[0],
            "mapping_note": taco_class_to_material(label)[1],
        }
        for category_id, label in categories.items()
    }
    write_json(source_dir / "material_map_draft.json", material_map)

    rows: list[dict[str, Any]] = []

    def one(image: dict[str, Any]) -> dict[str, Any]:
        file_name = image["file_name"]
        target = source_dir / "images" / file_name
        url = image.get("flickr_url") or image.get("flickr_640_url")
        fallback_url = image.get("flickr_640_url")
        image_anns = anns_by_image.get(image["id"], [])
        source_labels = sorted({categories[item["category_id"]] for item in image_anns})
        target_classes = sorted({taco_class_to_material(label)[0] for label in source_labels})
        try:
            status = download_file(url, target)
            error = ""
        except Exception as exc:
            error = str(exc)
            if fallback_url and fallback_url != url:
                try:
                    status = download_file(fallback_url, target)
                    status = "downloaded_fallback_640" if status == "downloaded" else status
                    url = fallback_url
                    error = ""
                except Exception as fallback_exc:
                    status = "failed"
                    error = f"{exc}; fallback failed: {fallback_exc}"
            else:
                status = "failed"
        ok, width, height, image_error = validate_image(target) if validate and target.exists() else (False, None, None, "")
        return {
            "source_id": "taco_official",
            "image_id": image["id"],
            "source_file": file_name,
            "source_label": "|".join(source_labels),
            "target_class": "|".join(target_classes) if target_classes else "review",
            "annotation_count": len(image_anns),
            "image_path": (
                str(target.relative_to(ROOT)) if target.exists() and ROOT in target.resolve().parents else
                str(target) if target.exists() else ""
            ),
            "download_status": status,
            "download_error": error or image_error,
            "width": width or image.get("width"),
            "height": height or image.get("height"),
            "validated_image": ok,
            "source_url": url,
        }

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(one, image) for image in annotations["images"]]
        for index, future in enumerate(as_completed(futures), start=1):
            rows.append(future.result())
            if index % 100 == 0 or index == len(futures):
                print(f"source taco: {index}/{len(futures)} images processed")

    write_csv(source_dir / "manifest.csv", rows)
    return summarize("taco_official", rows)


def download_outerview(out_dir: Path, extract: bool) -> dict[str, Any]:
    print("source outerview: files")
    source_dir = out_dir / "outerview_global_trash_debris"
    files_dir = source_dir / "files"
    for filename, expected_size in OUTERVIEW_FILES.items():
        url = f"{OUTERVIEW_BASE}/{quote(filename)}?download=true"
        download_file(url, files_dir / filename, expected_size=expected_size)

    images_dir = source_dir / "images"
    zip_path = files_dir / "1775002681200-Trash___Debris_Dataset-image.zip"
    if extract:
        marker = source_dir / ".images_extracted"
        if not marker.exists():
            print("source outerview: extracting image zip")
            images_dir.mkdir(parents=True, exist_ok=True)
            with zipfile.ZipFile(zip_path) as zf:
                zf.extractall(images_dir)
            marker.write_text(time.strftime("%Y-%m-%d %H:%M:%S"), encoding="utf-8")
        else:
            print("source outerview: extraction already marked complete")

    image_lookup = {path.name: path for path in images_dir.rglob("*") if path.is_file()} if images_dir.exists() else {}
    csv_path = files_dir / "1775001918223-Trash___Debris_Dataset.csv"
    rows: list[dict[str, Any]] = []
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for index, row in enumerate(reader, start=1):
            filename = row.get("filename", "")
            image_path = image_lookup.get(filename)
            rows.append(
                {
                    "source_id": "outerview_global_trash_debris",
                    "row_id": row.get("id", ""),
                    "source_file": filename,
                    "source_label": "trash_or_debris",
                    "target_class": "review",
                    "audit_note": "geotagged trash/debris; material label requires visual audit",
                    "latitude": row.get("latitude", ""),
                    "longitude": row.get("longitude", ""),
                    "region": row.get("region", ""),
                    "image_path": str(image_path.relative_to(ROOT)) if image_path else "",
                    "download_status": "image_found" if image_path else "metadata_only",
                    "metadata_row": index,
                }
            )
            if index % 5000 == 0:
                print(f"source outerview: {index} metadata rows processed")

    write_csv(source_dir / "manifest.csv", rows)
    return summarize("outerview_global_trash_debris", rows)


def summarize(source_id: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "source_id": source_id,
        "records": len(rows),
        "by_download_status": dict(Counter(row.get("download_status", "") for row in rows)),
        "by_target_class": dict(Counter(row.get("target_class", "") for row in rows)),
    }


def combine_manifests(out_dir: Path, summaries: list[dict[str, Any]]) -> None:
    combined_rows: list[dict[str, Any]] = []
    for manifest in out_dir.glob("*/manifest.csv"):
        with manifest.open("r", encoding="utf-8", newline="") as handle:
            combined_rows.extend(csv.DictReader(handle))
    if combined_rows:
        write_csv(out_dir / "combined_manifest.csv", combined_rows)
    if not summaries and combined_rows:
        by_source: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in combined_rows:
            by_source[row.get("source_id", "")].append(row)
        summaries = [summarize(source_id, rows) for source_id, rows in sorted(by_source.items())]
    write_json(
        out_dir / "download_summary.json",
        {
            "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "output_dir": str(out_dir),
            "sources": summaries,
            "combined_records": len(combined_rows),
        },
    )

    readme = out_dir / "README.md"
    readme.write_text(
        "# WasteWise Full Hard-Case Sources\n\n"
        "This folder stores heavy external hard-case datasets. It is ignored by git.\n\n"
        "- `realwaste_hf_uci/`: full RealWaste imagefolder pull from Hugging Face.\n"
        "- `taco_official/`: official TACO annotations, image URLs, and downloaded images.\n"
        "- `outerview_global_trash_debris/`: Outerview metadata and image archive.\n"
        "- `combined_manifest.csv`: combined source manifest for audit and split building.\n"
        "- `download_summary.json`: counts and download status.\n\n"
        "Next step: build a source-aware hard-case validation split. Do not merge all\n"
        "records directly into training without label audit and class mapping review.\n",
        encoding="utf-8",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument(
        "--sources",
        nargs="+",
        default=["realwaste", "taco", "outerview"],
        choices=["realwaste", "taco", "outerview"],
    )
    parser.add_argument("--no-validate", action="store_true")
    parser.add_argument("--no-extract-outerview", action="store_true")
    parser.add_argument("--skip-combine", action="store_true")
    parser.add_argument("--combine-only", action="store_true")
    parser.add_argument("--workers", type=int, default=8)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    summaries: list[dict[str, Any]] = []
    workers = max(1, args.workers)

    if args.combine_only:
        combine_manifests(out_dir, summaries)
        print(f"combined manifests in {out_dir}")
        return

    if "realwaste" in args.sources:
        summaries.append(download_realwaste(out_dir, validate=not args.no_validate, workers=workers))
    if "taco" in args.sources:
        summaries.append(download_taco(out_dir, validate=not args.no_validate, workers=workers))
    if "outerview" in args.sources:
        summaries.append(download_outerview(out_dir, extract=not args.no_extract_outerview))

    if not args.skip_combine:
        combine_manifests(out_dir, summaries)
    print(json.dumps({"output": str(out_dir), "sources": summaries}, indent=2))


if __name__ == "__main__":
    main()
