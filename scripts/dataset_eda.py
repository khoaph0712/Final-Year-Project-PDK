"""EDA report for a YOLO-format waste dataset.

Outputs CSV/JSON summaries, plots, and an EDA_REPORT.md under runs/dataset_eda/.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import yaml


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATA = ROOT / "merged_dataset_v3" / "data.yaml"
DEFAULT_OUT = ROOT / "runs" / "dataset_eda" / "merged_dataset_v3"
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


@dataclass
class BoxRecord:
    split: str
    class_id: int
    class_name: str
    image: str
    w_rel: float
    h_rel: float
    area_rel: float
    aspect: float
    w_px: float
    h_px: float


def resolve_split(root: Path, rel: str) -> Path:
    p = Path(rel)
    return p if p.is_absolute() else root / p


def image_to_label_path(image_path: Path) -> Path:
    return Path(str(image_path).replace("\\images\\", "\\labels\\")).with_suffix(".txt")


def md5_file(path: Path) -> str:
    h = hashlib.md5()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def percentile(vals: list[float], pct: float) -> float:
    if not vals:
        return 0.0
    xs = sorted(vals)
    idx = min(len(xs) - 1, max(0, round((pct / 100.0) * (len(xs) - 1))))
    return float(xs[idx])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--hash-duplicates", action="store_true", help="Compute exact image hashes; slower but useful.")
    args = parser.parse_args()

    cfg = yaml.safe_load(args.data.read_text(encoding="utf-8"))
    ds_root = Path(cfg["path"])
    names = list(cfg["names"])
    splits = {"train": cfg["train"], "valid": cfg.get("val", cfg.get("valid", "valid/images")), "test": cfg["test"]}

    args.out.mkdir(parents=True, exist_ok=True)

    split_image_counts: Counter[str] = Counter()
    split_label_counts: Counter[str] = Counter()
    class_counts: dict[str, Counter[int]] = {s: Counter() for s in splits}
    invalid_labels: list[dict] = []
    missing_labels: list[str] = []
    empty_labels: list[str] = []
    unreadable_images: list[str] = []
    boxes: list[BoxRecord] = []
    image_hashes: dict[str, list[str]] = defaultdict(list)

    for split, rel in splits.items():
        img_dir = resolve_split(ds_root, rel)
        image_paths = sorted(p for p in img_dir.rglob("*") if p.suffix.lower() in IMAGE_EXTS)
        split_image_counts[split] = len(image_paths)
        for img_path in image_paths:
            img = cv2.imread(str(img_path))
            if img is None:
                unreadable_images.append(str(img_path))
                continue
            h_px, w_px = img.shape[:2]
            if args.hash_duplicates:
                image_hashes[md5_file(img_path)].append(str(img_path))

            label_path = image_to_label_path(img_path)
            if not label_path.exists():
                missing_labels.append(str(img_path))
                continue
            lines = label_path.read_text(encoding="utf-8").splitlines()
            if not lines:
                empty_labels.append(str(label_path))
                continue
            split_label_counts[split] += 1

            for line_no, line in enumerate(lines, start=1):
                parts = line.strip().split()
                if len(parts) != 5:
                    invalid_labels.append({"label": str(label_path), "line": line_no, "reason": "wrong_column_count"})
                    continue
                try:
                    cid = int(float(parts[0]))
                    cx, cy, bw, bh = [float(x) for x in parts[1:]]
                except ValueError:
                    invalid_labels.append({"label": str(label_path), "line": line_no, "reason": "parse_error"})
                    continue
                if cid < 0 or cid >= len(names) or not (0 <= cx <= 1 and 0 <= cy <= 1 and 0 < bw <= 1 and 0 < bh <= 1):
                    invalid_labels.append({"label": str(label_path), "line": line_no, "reason": "out_of_range"})
                    continue
                area_rel = bw * bh
                class_counts[split][cid] += 1
                boxes.append(
                    BoxRecord(
                        split=split,
                        class_id=cid,
                        class_name=names[cid],
                        image=str(img_path),
                        w_rel=bw,
                        h_rel=bh,
                        area_rel=area_rel,
                        aspect=bw / max(bh, 1e-9),
                        w_px=bw * w_px,
                        h_px=bh * h_px,
                    )
                )

    duplicate_groups = {h: paths for h, paths in image_hashes.items() if len(paths) > 1}

    summary = {
        "dataset": str(args.data),
        "root": str(ds_root),
        "classes": names,
        "image_counts": dict(split_image_counts),
        "label_file_counts": dict(split_label_counts),
        "box_counts_by_split": {s: {names[c]: int(v) for c, v in counts.items()} for s, counts in class_counts.items()},
        "missing_label_images": len(missing_labels),
        "empty_label_files": len(empty_labels),
        "invalid_label_rows": len(invalid_labels),
        "unreadable_images": len(unreadable_images),
        "duplicate_groups": len(duplicate_groups),
        "duplicate_images": sum(len(v) for v in duplicate_groups.values()),
    }
    (args.out / "eda_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (args.out / "invalid_labels.json").write_text(json.dumps(invalid_labels[:5000], indent=2), encoding="utf-8")
    (args.out / "duplicate_images.json").write_text(json.dumps(duplicate_groups, indent=2), encoding="utf-8")

    with (args.out / "class_counts.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["class", "train", "valid", "test", "total"])
        for cid, name in enumerate(names):
            row = [class_counts[s].get(cid, 0) for s in ("train", "valid", "test")]
            w.writerow([name, *row, sum(row)])

    small_by_class: dict[str, Counter[str]] = {s: Counter() for s in splits}
    for b in boxes:
        if b.w_px < 10 or b.h_px < 10 or b.area_rel < 0.0002:
            small_by_class[b.split][b.class_name] += 1

    with (args.out / "box_stats.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["class", "n", "area_p10", "area_p50", "area_p90", "aspect_p50", "small_boxes"])
        for cid, name in enumerate(names):
            cls_boxes = [b for b in boxes if b.class_id == cid]
            areas = [b.area_rel for b in cls_boxes]
            aspects = [b.aspect for b in cls_boxes]
            small = sum(small_by_class[split].get(name, 0) for split in splits)
            w.writerow([name, len(cls_boxes), percentile(areas, 10), percentile(areas, 50), percentile(areas, 90), percentile(aspects, 50), small])

    # Plots
    fig, ax = plt.subplots(figsize=(10, 5))
    x = range(len(names))
    width = 0.25
    for idx, split in enumerate(("train", "valid", "test")):
        ax.bar([i + (idx - 1) * width for i in x], [class_counts[split].get(i, 0) for i in x], width=width, label=split)
    ax.set_xticks(list(x))
    ax.set_xticklabels(names, rotation=30, ha="right")
    ax.set_title("Box counts by class and split")
    ax.set_ylabel("Boxes")
    ax.legend()
    fig.tight_layout()
    fig.savefig(args.out / "chart_class_balance.png", dpi=140)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist([b.area_rel for b in boxes], bins=60)
    ax.set_title("Bounding box area distribution")
    ax.set_xlabel("Relative box area")
    ax.set_ylabel("Count")
    fig.tight_layout()
    fig.savefig(args.out / "chart_box_area_hist.png", dpi=140)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 5))
    totals = Counter()
    smalls = Counter()
    for b in boxes:
        totals[b.class_name] += 1
        if b.w_px < 10 or b.h_px < 10 or b.area_rel < 0.0002:
            smalls[b.class_name] += 1
    vals = [100.0 * smalls[n] / max(1, totals[n]) for n in names]
    ax.bar(names, vals)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=30, ha="right")
    ax.set_title("Tiny/suspicious box ratio by class")
    ax.set_ylabel("% boxes")
    fig.tight_layout()
    fig.savefig(args.out / "chart_small_box_ratio.png", dpi=140)
    plt.close(fig)

    lines = [
        "# Dataset EDA Report",
        "",
        f"- Dataset config: `{args.data}`",
        f"- Dataset root: `{ds_root}`",
        f"- Classes: {', '.join(names)}",
        "",
        "## Split Summary",
        "",
        "| Split | Images | Label files | Boxes |",
        "|---|---:|---:|---:|",
    ]
    for split in ("train", "valid", "test"):
        lines.append(f"| {split} | {split_image_counts[split]} | {split_label_counts[split]} | {sum(class_counts[split].values())} |")
    lines.extend(
        [
            "",
            "## Data Quality Flags",
            "",
            f"- Missing label images: **{len(missing_labels)}**",
            f"- Empty label files: **{len(empty_labels)}**",
            f"- Invalid label rows: **{len(invalid_labels)}**",
            f"- Unreadable images: **{len(unreadable_images)}**",
            f"- Exact duplicate groups: **{len(duplicate_groups)}** (only computed when `--hash-duplicates` is used)",
            "",
            "## Class Balance",
            "",
            "| Class | Train | Valid | Test | Total |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for cid, name in enumerate(names):
        row = [class_counts[s].get(cid, 0) for s in ("train", "valid", "test")]
        lines.append(f"| {name} | {row[0]} | {row[1]} | {row[2]} | {sum(row)} |")
    lines.extend(
        [
            "",
            "## Recommended Dataset Fine-Tuning",
            "",
            "- Keep `merged_dataset_v3` unchanged as the locked baseline.",
            "- Build a tuned copy that removes invalid/tiny boxes and exact duplicate images.",
            "- Rebuild train/valid/test splits by dominant class so weak classes are represented more evenly.",
            "- Focus later data collection only on weak observed cases: organic, paper, real phone-camera images, messy backgrounds, and lighting variation.",
            "",
            "## Figures",
            "",
            "- `chart_class_balance.png`",
            "- `chart_box_area_hist.png`",
            "- `chart_small_box_ratio.png`",
        ]
    )
    (args.out / "EDA_REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"[OK] EDA written to {args.out}")


if __name__ == "__main__":
    main()
