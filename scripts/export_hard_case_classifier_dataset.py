"""Export a classifier-ready dataset with hard cases mixed into the baseline.

The output uses the classification folder layout already supported by
ml_balanced_training.load_crops_and_balance:

data/hard_case_classifier_v1/
  train/<class>/*.jpg
  val/<class>/*.jpg
  test/<class>/*.jpg
  data.yaml
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import os
import random
import shutil
from pathlib import Path
from typing import Iterable


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BASELINE = ROOT / "data" / "merged_dataset_v5"
DEFAULT_SPLIT_MANIFEST = ROOT / "external_datasets" / "hard_case_full" / "source_aware_split_manifest.csv"
DEFAULT_OUT = ROOT / "data" / "hard_case_classifier_v1"

CLASSES = ["plastic", "glass", "metal", "paper", "cardboard", "organic", "Background"]
CLASS_MAP = {
    "Plastic": "plastic",
    "Glass": "glass",
    "Metal": "metal",
    "Paper": "paper",
    "Cardboard": "cardboard",
    "Organic": "organic",
    "Background": "Background",
}
SPLIT_MAP = {
    "classifier_train_candidate": "train",
    "classifier_hard_val": "val",
    "classifier_hard_test": "test",
}


def stable_name(prefix: str, source: Path) -> str:
    digest = hashlib.md5(str(source).encode("utf-8")).hexdigest()[:10]
    return f"{prefix}_{digest}{source.suffix.lower() or '.jpg'}"


def reset_output(path: Path) -> None:
    resolved = path.resolve()
    data_root = (ROOT / "data").resolve()
    if data_root not in resolved.parents:
        raise ValueError(f"refusing to clear output outside data/: {resolved}")
    if resolved.exists():
        shutil.rmtree(resolved)
    for split in ("train", "val", "test"):
        for class_name in CLASSES:
            (resolved / split / class_name).mkdir(parents=True, exist_ok=True)


def link_or_copy(source: Path, target: Path) -> str:
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        return "exists"
    try:
        os.link(source, target)
        return "hardlink"
    except OSError:
        shutil.copy2(source, target)
        return "copy"


def image_files(path: Path) -> list[Path]:
    return [
        item
        for item in path.iterdir()
        if item.is_file() and item.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    ]


def sample_paths(paths: list[Path], cap: int, seed: int, key: str) -> list[Path]:
    rng = random.Random(f"{seed}:{key}")
    selected = paths[:]
    rng.shuffle(selected)
    return selected[:cap]


def export_baseline(
    out_dir: Path,
    baseline_dir: Path,
    train_cap: int,
    val_cap: int,
    test_cap: int,
    seed: int,
) -> list[dict[str, str]]:
    records: list[dict[str, str]] = []

    for class_name in CLASSES:
        train_source = baseline_dir / "train" / class_name
        if train_source.exists():
            for source in sample_paths(image_files(train_source), train_cap, seed, f"train:{class_name}"):
                target = out_dir / "train" / class_name / stable_name("baseline_train", source)
                method = link_or_copy(source, target)
                records.append(
                    {
                        "split": "train",
                        "class": class_name,
                        "source_id": "merged_dataset_v5",
                        "source_path": str(source.relative_to(ROOT)),
                        "target_path": str(target.relative_to(ROOT)),
                        "link_method": method,
                    }
                )

        test_source = baseline_dir / "test" / class_name
        if not test_source.exists():
            continue
        heldout = sample_paths(image_files(test_source), val_cap + test_cap, seed, f"heldout:{class_name}")
        val_items = heldout[:val_cap]
        test_items = heldout[val_cap : val_cap + test_cap]
        for split, items in (("val", val_items), ("test", test_items)):
            for source in items:
                target = out_dir / split / class_name / stable_name(f"baseline_{split}", source)
                method = link_or_copy(source, target)
                records.append(
                    {
                        "split": split,
                        "class": class_name,
                        "source_id": "merged_dataset_v5",
                        "source_path": str(source.relative_to(ROOT)),
                        "target_path": str(target.relative_to(ROOT)),
                        "link_method": method,
                    }
                )
    return records


def export_hard_cases(out_dir: Path, split_manifest: Path) -> list[dict[str, str]]:
    records: list[dict[str, str]] = []
    with split_manifest.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if row.get("source_id") != "realwaste_hf_uci":
                continue
            split = SPLIT_MAP.get(row.get("recommended_split", ""))
            class_name = CLASS_MAP.get(row.get("target_class", ""))
            image_rel = row.get("image_path", "")
            if not split or not class_name or not image_rel:
                continue
            source = ROOT / image_rel
            if not source.exists():
                continue
            target = out_dir / split / class_name / stable_name("realwaste", source)
            method = link_or_copy(source, target)
            records.append(
                {
                    "split": split,
                    "class": class_name,
                    "source_id": "realwaste_hf_uci",
                    "source_path": str(source.relative_to(ROOT)),
                    "target_path": str(target.relative_to(ROOT)),
                    "link_method": method,
                    "recommended_split": row.get("recommended_split", ""),
                    "source_label": row.get("source_label", ""),
                }
            )
    return records


def write_manifest(out_dir: Path, records: Iterable[dict[str, str]]) -> None:
    rows = list(records)
    manifest = out_dir / "source_manifest.csv"
    fieldnames = sorted({key for row in rows for key in row})
    with manifest.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_data_yaml(out_dir: Path) -> None:
    names = "\n".join(f"- {name}" for name in CLASSES)
    (out_dir / "data.yaml").write_text(
        f"path: {out_dir}\n"
        "type: classification\n"
        "train: train\n"
        "val: val\n"
        "test: test\n"
        f"nc: {len(CLASSES)}\n"
        "names:\n"
        f"{names}\n",
        encoding="utf-8",
    )


def print_counts(out_dir: Path) -> None:
    for split in ("train", "val", "test"):
        print(f"[{split}]")
        for class_name in CLASSES:
            count = len(image_files(out_dir / split / class_name))
            print(f"  {class_name}: {count}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, default=DEFAULT_BASELINE)
    parser.add_argument("--split-manifest", type=Path, default=DEFAULT_SPLIT_MANIFEST)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--baseline-train-cap", type=int, default=1200)
    parser.add_argument("--baseline-val-cap", type=int, default=300)
    parser.add_argument("--baseline-test-cap", type=int, default=300)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = args.out_dir.resolve()
    if out_dir.exists() and not args.overwrite:
        raise SystemExit(f"output exists; pass --overwrite to rebuild: {out_dir}")

    reset_output(out_dir)
    records = []
    records.extend(
        export_baseline(
            out_dir,
            args.baseline.resolve(),
            args.baseline_train_cap,
            args.baseline_val_cap,
            args.baseline_test_cap,
            args.seed,
        )
    )
    records.extend(export_hard_cases(out_dir, args.split_manifest.resolve()))
    write_manifest(out_dir, records)
    write_data_yaml(out_dir)
    print(f"exported records: {len(records)}")
    print(f"output: {out_dir}")
    print_counts(out_dir)


if __name__ == "__main__":
    main()
