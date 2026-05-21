"""Source-level EDA and handcrafted feature analysis for merged YOLO datasets.

The lecturer concern is that a merged dataset can hide severe imbalance and
source bias. This script keeps the original source prefix from filenames
(`rf_taco_trash__...`) and reports each source separately.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml

from feature_ml_analysis import (
    ALL_FEATURE_NAMES,
    FEATURE_GROUPS,
    clamp_box,
    extract_combined_features,
    feature_slice,
)


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATA = ROOT / "merged_dataset_v3" / "data.yaml"
DEFAULT_OUT = ROOT / "runs" / "source_analysis" / "merged_dataset_v3"


@dataclass
class SourceStats:
    source: str
    images: dict[str, int] = field(default_factory=lambda: defaultdict(int))
    objects: dict[str, int] = field(default_factory=lambda: defaultdict(int))
    class_counts: dict[str, dict[str, int]] = field(default_factory=lambda: defaultdict(lambda: defaultdict(int)))
    widths: list[int] = field(default_factory=list)
    heights: list[int] = field(default_factory=list)
    box_area_ratios: list[float] = field(default_factory=list)
    boxes_per_image: list[int] = field(default_factory=list)


def infer_source_name(image_path: Path) -> str:
    name = image_path.name
    if "__" in name:
        return name.split("__", 1)[0]
    return "unknown"


def resolve_split_images(ds_root: Path, split_rel: str) -> Path:
    p = Path(split_rel)
    if p.is_absolute():
        return p
    direct = ds_root / p
    if direct.exists():
        return direct
    if split_rel.startswith("../"):
        fallback = ds_root / split_rel.replace("../", "", 1)
        if fallback.exists():
            return fallback
    return direct


def image_to_label_path(image_path: Path) -> Path:
    return Path(str(image_path).replace("\\images\\", "\\labels\\")).with_suffix(".txt")


def iter_images(image_dir: Path) -> list[Path]:
    paths = [
        p
        for p in image_dir.rglob("*")
        if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    ]
    paths.sort()
    return paths


def load_yolo_labels(label_path: Path, nc: int) -> list[tuple[int, float, float, float, float]]:
    out: list[tuple[int, float, float, float, float]] = []
    if not label_path.exists():
        return out
    for line in label_path.read_text(encoding="utf-8").splitlines():
        parts = line.strip().split()
        if len(parts) != 5:
            continue
        cid = int(float(parts[0]))
        if cid < 0 or cid >= nc:
            continue
        cx, cy, bw, bh = [float(x) for x in parts[1:]]
        out.append((cid, cx, cy, bw, bh))
    return out


def pct(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    return float(np.percentile(np.asarray(values, dtype=np.float64), q))


def collect_stats(cfg: dict, data_yaml: Path) -> tuple[list[str], dict[str, SourceStats]]:
    class_names = list(cfg["names"])
    ds_root = Path(cfg.get("path", data_yaml.parent))
    split_map = {
        "train": resolve_split_images(ds_root, cfg["train"]),
        "valid": resolve_split_images(ds_root, cfg.get("val", cfg.get("valid", cfg["train"]))),
        "test": resolve_split_images(ds_root, cfg.get("test", cfg.get("val", cfg["train"]))),
    }
    by_source: dict[str, SourceStats] = {}

    for split, image_dir in split_map.items():
        for img_path in iter_images(image_dir):
            source = infer_source_name(img_path)
            stats = by_source.setdefault(source, SourceStats(source=source))
            stats.images[split] += 1
            img = cv2.imread(str(img_path))
            if img is not None:
                h, w = img.shape[:2]
                stats.widths.append(w)
                stats.heights.append(h)
            labels = load_yolo_labels(image_to_label_path(img_path), len(class_names))
            stats.boxes_per_image.append(len(labels))
            stats.objects[split] += len(labels)
            for cid, _cx, _cy, bw, bh in labels:
                cname = class_names[cid]
                stats.class_counts[split][cname] += 1
                stats.box_area_ratios.append(float(bw * bh))

    return class_names, by_source


def class_imbalance_ratio(counts: dict[str, int]) -> float:
    nonzero = [v for v in counts.values() if v > 0]
    if not nonzero:
        return 0.0
    return max(nonzero) / max(1, min(nonzero))


def save_source_csv(class_names: list[str], by_source: dict[str, SourceStats], out_dir: Path) -> list[dict]:
    rows: list[dict] = []
    csv_path = out_dir / "source_summary.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        fields = [
            "source",
            "train_images",
            "valid_images",
            "test_images",
            "total_images",
            "train_objects",
            "valid_objects",
            "test_objects",
            "total_objects",
            "classes_present",
            "imbalance_ratio_total",
            "median_box_area_ratio",
            "median_boxes_per_image",
        ] + [f"objects_{name}" for name in class_names]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for source, stats in sorted(by_source.items(), key=lambda kv: sum(kv[1].objects.values()), reverse=True):
            total_by_class = {
                name: sum(stats.class_counts[split].get(name, 0) for split in ("train", "valid", "test"))
                for name in class_names
            }
            row = {
                "source": source,
                "train_images": int(stats.images.get("train", 0)),
                "valid_images": int(stats.images.get("valid", 0)),
                "test_images": int(stats.images.get("test", 0)),
                "total_images": int(sum(stats.images.values())),
                "train_objects": int(stats.objects.get("train", 0)),
                "valid_objects": int(stats.objects.get("valid", 0)),
                "test_objects": int(stats.objects.get("test", 0)),
                "total_objects": int(sum(stats.objects.values())),
                "classes_present": int(sum(1 for v in total_by_class.values() if v > 0)),
                "imbalance_ratio_total": round(class_imbalance_ratio(total_by_class), 3),
                "median_box_area_ratio": round(pct(stats.box_area_ratios, 50), 6),
                "median_boxes_per_image": round(pct([float(v) for v in stats.boxes_per_image], 50), 3),
            }
            for name in class_names:
                row[f"objects_{name}"] = int(total_by_class[name])
            writer.writerow(row)
            rows.append(row)
    return rows


def save_class_chart(class_names: list[str], by_source: dict[str, SourceStats], out_dir: Path) -> None:
    sources = [row[0] for row in sorted(by_source.items(), key=lambda kv: sum(kv[1].objects.values()), reverse=True)]
    data = np.zeros((len(sources), len(class_names)), dtype=np.float64)
    for i, source in enumerate(sources):
        stats = by_source[source]
        for j, cname in enumerate(class_names):
            data[i, j] = sum(stats.class_counts[split].get(cname, 0) for split in ("train", "valid", "test"))

    fig, ax = plt.subplots(figsize=(12, max(4, len(sources) * 0.65)))
    left = np.zeros(len(sources), dtype=np.float64)
    for j, cname in enumerate(class_names):
        ax.barh(sources, data[:, j], left=left, label=cname)
        left += data[:, j]
    ax.set_title("Object class distribution by source dataset")
    ax.set_xlabel("Objects")
    ax.invert_yaxis()
    ax.legend(loc="center left", bbox_to_anchor=(1.0, 0.5))
    fig.tight_layout()
    fig.savefig(out_dir / "chart_source_class_distribution.png", dpi=150)
    plt.close(fig)


def collect_feature_samples(
    cfg: dict,
    data_yaml: Path,
    source: str,
    max_objects: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    class_names = list(cfg["names"])
    ds_root = Path(cfg.get("path", data_yaml.parent))
    split_map = [
        resolve_split_images(ds_root, cfg["train"]),
        resolve_split_images(ds_root, cfg.get("val", cfg.get("valid", cfg["train"]))),
        resolve_split_images(ds_root, cfg.get("test", cfg.get("val", cfg["train"]))),
    ]
    items: list[tuple[Path, tuple[int, float, float, float, float]]] = []
    for image_dir in split_map:
        for img_path in iter_images(image_dir):
            if infer_source_name(img_path) != source:
                continue
            labels = load_yolo_labels(image_to_label_path(img_path), len(class_names))
            items.extend((img_path, label) for label in labels)
    rng = random.Random(seed)
    rng.shuffle(items)
    items = items[:max_objects]

    feats: list[np.ndarray] = []
    ys: list[int] = []
    for img_path, (cid, cx, cy, bw, bh) in items:
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        h, w = img.shape[:2]
        x1 = int((cx - bw / 2.0) * w)
        y1 = int((cy - bh / 2.0) * h)
        x2 = int((cx + bw / 2.0) * w)
        y2 = int((cy + bh / 2.0) * h)
        x1, y1, x2, y2 = clamp_box(x1, y1, x2, y2, w, h)
        if (x2 - x1) < 10 or (y2 - y1) < 10:
            continue
        feats.append(extract_combined_features(img[y1:y2, x1:x2]))
        ys.append(cid)
    if not feats:
        return np.empty((0, len(ALL_FEATURE_NAMES)), dtype=np.float32), np.empty((0,), dtype=np.int64), class_names
    return np.vstack(feats).astype(np.float32), np.asarray(ys, dtype=np.int64), class_names


def save_feature_summary(cfg: dict, data_yaml: Path, rows: list[dict], out_dir: Path, max_objects: int) -> None:
    feature_dir = out_dir / "features_by_source"
    feature_dir.mkdir(parents=True, exist_ok=True)
    summary_path = out_dir / "source_feature_summary.csv"
    fieldnames = ["source", "sampled_objects", "top_feature_1", "top_feature_2", "top_feature_3"]
    for group_name, _names in FEATURE_GROUPS:
        fieldnames.append(f"{group_name.lower()}_mean_abs")

    with summary_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            source = row["source"]
            x, y, class_names = collect_feature_samples(cfg, data_yaml, source, max_objects=max_objects, seed=42)
            if len(x) == 0:
                continue
            mu = np.mean(x, axis=0)
            sigma = np.std(x, axis=0)
            top_idx = np.argsort(-sigma)[:3].tolist()
            out_row = {
                "source": source,
                "sampled_objects": int(len(x)),
                "top_feature_1": ALL_FEATURE_NAMES[top_idx[0]],
                "top_feature_2": ALL_FEATURE_NAMES[top_idx[1]],
                "top_feature_3": ALL_FEATURE_NAMES[top_idx[2]],
            }
            for group_idx, (group_name, _names) in enumerate(FEATURE_GROUPS):
                out_row[f"{group_name.lower()}_mean_abs"] = round(float(np.mean(np.abs(mu[feature_slice(group_idx)]))), 6)
            writer.writerow(out_row)

            source_csv = feature_dir / f"{source}.csv"
            with source_csv.open("w", newline="", encoding="utf-8") as sf:
                sw = csv.writer(sf)
                sw.writerow(["feature", "mean", "std"])
                for i, name in enumerate(ALL_FEATURE_NAMES):
                    sw.writerow([name, round(float(mu[i]), 6), round(float(sigma[i]), 6)])


def write_report(data_yaml: Path, class_names: list[str], rows: list[dict], out_dir: Path) -> None:
    lines = [
        "# Dataset Source Analysis",
        "",
        "## Why This Exists",
        "The lecturer feedback is that the merged dataset is heavily imbalanced and hides source-level bias. "
        "This report keeps the original filename source prefix and analyses each dataset/source separately before any final merge.",
        "",
        f"- Data YAML: `{data_yaml}`",
        f"- Classes after previous merge: {', '.join(class_names)}",
        "",
        "## Source Summary",
        "| Source | Images | Objects | Classes Present | Imbalance Ratio | Main Risk |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        risk = "OK"
        if float(row["imbalance_ratio_total"]) >= 20:
            risk = "severe imbalance"
        elif int(row["classes_present"]) < len(class_names):
            risk = "missing classes"
        elif float(row["imbalance_ratio_total"]) >= 8:
            risk = "moderate imbalance"
        lines.append(
            f"| {row['source']} | {row['total_images']} | {row['total_objects']} | "
            f"{row['classes_present']} | {row['imbalance_ratio_total']} | {risk} |"
        )

    lines.extend(
        [
            "",
            "## Files Generated",
            "- `source_summary.csv`: image/object counts, class distribution, imbalance ratio.",
            "- `source_feature_summary.csv`: sampled handcrafted feature summary by source.",
            "- `features_by_source/*.csv`: mean/std of all 637 features for each source.",
            "- `chart_source_class_distribution.png`: stacked class distribution by source.",
            "",
            "## Interpretation For Report",
            "- Do not claim the merged dataset is balanced.",
            "- Present each source separately first.",
            "- Train ANN/CNN by source, then compare with the merged result.",
            "- If merging later, cap per class and per source so one source cannot dominate the final dataset.",
        ]
    )
    (out_dir / "DATASET_SOURCE_COMPARISON.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--max-feature-objects-per-source", type=int, default=1500)
    args = parser.parse_args()

    cfg = yaml.safe_load(args.data.read_text(encoding="utf-8"))
    args.out.mkdir(parents=True, exist_ok=True)

    class_names, by_source = collect_stats(cfg, args.data)
    rows = save_source_csv(class_names, by_source, args.out)
    save_class_chart(class_names, by_source, args.out)
    save_feature_summary(cfg, args.data, rows, args.out, max_objects=args.max_feature_objects_per_source)
    write_report(args.data.resolve(), class_names, rows, args.out)
    print(f"[OK] Source analysis written to {args.out}")


if __name__ == "__main__":
    main()
