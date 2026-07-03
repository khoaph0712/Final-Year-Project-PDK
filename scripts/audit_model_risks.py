#!/usr/bin/env python
"""Model risk audit: data leakage, bias, label noise, overfitting signals.

CPU-only, read-only over the datasets. Covers:
- exact-duplicate leakage across splits (MD5 of file bytes)
- near-duplicate leakage across splits (64-bit perceptual dHash, Hamming <= 4)
- cross-dataset eval contamination (hardcase val/test vs super/merged train)
- per-split class x box-count and box-size bias tables
- label noise: out-of-bounds/zero-area/duplicate boxes, polygons, empty labels,
  undecodable images
- overfitting signals parsed from the running/finished results.csv

Run: python scripts/audit_model_risks.py --dataset all
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

DATASETS = {
    "hardcase": {
        "path": ROOT / "external_datasets" / "yolo26_hardcase_dataset_v1",
        "layout": "yolo",
        "classes": ["plastic", "glass", "metal", "paper", "cardboard", "organic"],
    },
    "super": {
        "path": ROOT / "external_datasets" / "super_yolo_dataset",
        "layout": "yolo",
        "classes": ["plastic", "glass", "metal", "paper", "cardboard", "organic"],
    },
    "merged": {
        "path": ROOT / "data" / "merged_dataset_v5",
        "layout": "classification",
        "classes": ["plastic", "glass", "metal", "paper", "cardboard", "organic", "Background"],
    },
    "hardcase_classifier": {
        "path": ROOT / "data" / "hard_case_classifier_v1",
        "layout": "classification",
        "classes": ["plastic", "glass", "metal", "paper", "cardboard", "organic", "Background"],
    },
}
RESULTS_CSV = ROOT / "runs" / "detect" / "yolo26n_hardcase_v2_long" / "results.csv"
FIELD_MARKERS = ("_train_", "_test_", "_val_", "_valid_", "rf_")


# ---------------------------------------------------------------- hashing

def dhash64(gray: np.ndarray) -> int:
    small = cv2.resize(gray, (9, 8), interpolation=cv2.INTER_AREA)
    bits = small[:, 1:] > small[:, :-1]
    return int(np.packbits(bits).view(">u8")[0])


def hamming(a: int, b: int) -> int:
    return bin(a ^ b).count("1")


class NearDupIndex:
    """Multi-index for Hamming<=4 lookup: 5 chunks of ~13 bits; any pair within
    Hamming 4 must agree exactly on at least one chunk (pigeonhole)."""

    CHUNKS = [(0, 13), (13, 13), (26, 13), (39, 13), (52, 12)]

    def __init__(self) -> None:
        self.buckets: list[dict[int, list[int]]] = [defaultdict(list) for _ in self.CHUNKS]
        self.hashes: list[int] = []
        self.keys: list[str] = []

    def add(self, h: int, key: str) -> None:
        idx = len(self.hashes)
        self.hashes.append(h)
        self.keys.append(key)
        for b, (off, width) in enumerate(self.CHUNKS):
            self.buckets[b][(h >> off) & ((1 << width) - 1)].append(idx)

    def query(self, h: int, max_ham: int = 4) -> list[tuple[str, int]]:
        seen: set[int] = set()
        out: list[tuple[str, int]] = []
        for b, (off, width) in enumerate(self.CHUNKS):
            for idx in self.buckets[b].get((h >> off) & ((1 << width) - 1), []):
                if idx in seen:
                    continue
                seen.add(idx)
                d = hamming(h, self.hashes[idx])
                if d <= max_ham:
                    out.append((self.keys[idx], d))
        return out


def scan_images(paths: list[Path]) -> dict[str, dict]:
    """One decode pass per image: md5 (bytes), dhash + decodability (reduced decode)."""
    info: dict[str, dict] = {}
    for i, p in enumerate(paths):
        if i and i % 5000 == 0:
            print(f"    ... {i}/{len(paths)} images scanned")
        try:
            data = p.read_bytes()
        except OSError:
            info[str(p)] = {"md5": None, "dhash": None, "decodable": False}
            continue
        md5 = hashlib.md5(data).hexdigest()
        arr = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_REDUCED_GRAYSCALE_4)
        if arr is None:
            arr = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_GRAYSCALE)
        info[str(p)] = {
            "md5": md5,
            "dhash": dhash64(arr) if arr is not None else None,
            "decodable": arr is not None,
        }
    return info


# ---------------------------------------------------------------- dataset walking

def split_images(cfg: dict) -> dict[str, list[Path]]:
    root = cfg["path"]
    splits: dict[str, list[Path]] = {}
    for split_dir in sorted(d for d in root.iterdir() if d.is_dir()):
        if cfg["layout"] == "yolo":
            img_dir = split_dir / "images"
            if img_dir.exists():
                splits[split_dir.name] = [
                    p for p in img_dir.iterdir() if p.suffix.lower() in IMG_EXT
                ]
        else:
            imgs = [
                p
                for cls_dir in split_dir.iterdir()
                if cls_dir.is_dir()
                for p in cls_dir.iterdir()
                if p.suffix.lower() in IMG_EXT
            ]
            if imgs:
                splits[split_dir.name] = imgs
    return splits


def audit_labels(cfg: dict) -> dict:
    """Label-noise + bias stats for a YOLO-layout dataset."""
    root, classes = cfg["path"], cfg["classes"]
    out: dict[str, dict] = {}
    for split_dir in sorted(d for d in root.iterdir() if d.is_dir()):
        lbl_dir = split_dir / "labels"
        if not lbl_dir.exists():
            continue
        boxes = Counter()
        areas: dict[int, list[float]] = defaultdict(list)
        noise = Counter()
        single_class = 0
        n_files = 0
        for lf in lbl_dir.glob("*.txt"):
            n_files += 1
            lines = [ln.split() for ln in lf.read_text(encoding="utf-8").splitlines() if ln.strip()]
            if not lines:
                noise["empty_label_files"] += 1
                continue
            file_classes = set()
            seen_lines: set[tuple] = set()
            for parts in lines:
                if len(parts) > 5:
                    noise["polygon_rows"] += 1
                    continue
                if len(parts) < 5:
                    noise["malformed_rows"] += 1
                    continue
                c = int(float(parts[0]))
                x, y, w, h = (float(v) for v in parts[1:5])
                key = (c, round(x, 6), round(y, 6), round(w, 6), round(h, 6))
                if key in seen_lines:
                    noise["duplicate_boxes"] += 1
                seen_lines.add(key)
                if not all(0.0 <= v <= 1.0 for v in (x, y, w, h)):
                    noise["out_of_bounds"] += 1
                if w <= 0 or h <= 0:
                    noise["zero_area"] += 1
                    continue
                boxes[c] += 1
                areas[c].append(w * h)
                file_classes.add(c)
            if len(file_classes) == 1:
                single_class += 1
        cls_rows = []
        for c in sorted(boxes):
            a = sorted(areas[c])
            cls_rows.append(
                {
                    "class": classes[c] if c < len(classes) else str(c),
                    "boxes": boxes[c],
                    "median_area_pct": round(a[len(a) // 2] * 100, 2),
                    "tiny_lt1pct": round(sum(1 for v in a if v < 0.01) / len(a) * 100, 1),
                }
            )
        out[split_dir.name] = {
            "label_files": n_files,
            "classes": cls_rows,
            "noise": dict(noise),
            "single_class_image_pct": round(single_class / max(n_files, 1) * 100, 1),
        }
    return out


def merged_domain_bias(cfg: dict) -> dict:
    out: dict[str, dict] = {}
    for split_dir in sorted(d for d in cfg["path"].iterdir() if d.is_dir()):
        per_class = {}
        for cls_dir in sorted(d for d in split_dir.iterdir() if d.is_dir()):
            names = [p.name for p in cls_dir.iterdir() if p.suffix.lower() in IMG_EXT]
            field = sum(1 for n in names if any(m in n for m in FIELD_MARKERS))
            per_class[cls_dir.name] = {
                "total": len(names),
                "field": field,
                "studio": len(names) - field,
            }
        out[split_dir.name] = per_class
    return out


# ---------------------------------------------------------------- leakage

def cross_split_dups(splits: dict[str, list[Path]], info: dict[str, dict]) -> dict:
    md5_owner: dict[str, tuple[str, str]] = {}
    exact: list[dict] = []
    near_index: dict[str, NearDupIndex] = {}
    near: list[dict] = []

    for split, paths in splits.items():
        idx = NearDupIndex()
        for p in paths:
            rec = info[str(p)]
            if rec["md5"]:
                prev = md5_owner.get(rec["md5"])
                if prev and prev[0] != split:
                    exact.append({"a": prev[1], "b": str(p), "splits": f"{prev[0]}<->{split}"})
                else:
                    md5_owner.setdefault(rec["md5"], (split, str(p)))
            if rec["dhash"] is not None:
                for other_split, other_idx in near_index.items():
                    for key, d in other_idx.query(rec["dhash"]):
                        near.append(
                            {"a": key, "b": str(p), "splits": f"{other_split}<->{split}", "hamming": d}
                        )
                idx.add(rec["dhash"], str(p))
        near_index[split] = idx

    # near list includes exact copies too; drop pairs already reported as exact
    exact_pairs = {(e["a"], e["b"]) for e in exact} | {(e["b"], e["a"]) for e in exact}
    near = [n for n in near if (n["a"], n["b"]) not in exact_pairs]
    return {"exact": exact, "near": near}


def cross_dataset_contamination(
    eval_splits: dict[str, list[Path]],
    train_paths: list[Path],
    info_eval: dict[str, dict],
    info_train: dict[str, dict],
) -> dict:
    train_md5 = {info_train[str(p)]["md5"] for p in train_paths} - {None}
    train_idx = NearDupIndex()
    for p in train_paths:
        h = info_train[str(p)]["dhash"]
        if h is not None:
            train_idx.add(h, str(p))
    out = {}
    for split in ("val", "test", "valid"):
        if split not in eval_splits:
            continue
        exact = near = 0
        examples = []
        for p in eval_splits[split]:
            rec = info_eval[str(p)]
            if rec["md5"] in train_md5:
                exact += 1
                if len(examples) < 5:
                    examples.append(str(p))
            elif rec["dhash"] is not None and train_idx.query(rec["dhash"]):
                near += 1
                if len(examples) < 5:
                    examples.append(str(p))
        out[split] = {"n": len(eval_splits[split]), "exact_in_train": exact, "near_in_train": near, "examples": examples}
    return out


# ---------------------------------------------------------------- overfitting

def parse_results_csv(path: Path) -> dict:
    if not path.exists():
        return {"available": False}
    with path.open() as fh:
        rows = list(csv.DictReader(fh))
    if not rows:
        return {"available": False}

    last = rows[-1]
    best = max(rows, key=lambda r: float(r.get("metrics/mAP50-95(B)", 0) or 0))
    return {
        "available": True,
        "epochs_done": int(float(last["epoch"])),
        "last": {
            "train_cls_loss": float(last["train/cls_loss"]),
            "val_cls_loss": float(last["val/cls_loss"]),
            "precision": float(last["metrics/precision(B)"]),
            "recall": float(last["metrics/recall(B)"]),
            "mAP50": float(last["metrics/mAP50(B)"]),
        },
        "best": {
            "epoch": int(float(best["epoch"])),
            "mAP50": float(best["metrics/mAP50(B)"]),
            "mAP50_95": float(best["metrics/mAP50-95(B)"]),
        },
        "val_train_cls_gap": round(float(last["val/cls_loss"]) - float(last["train/cls_loss"]), 3),
    }


# ---------------------------------------------------------------- controls

def self_test() -> None:
    # structured images (gradients + shapes) - dHash is only stable for natural
    # low-frequency content, pure noise would not survive a resize round-trip
    xx, yy = np.meshgrid(np.linspace(0, 255, 320), np.linspace(0, 255, 240))
    img = ((xx + yy) / 2).astype(np.uint8)
    cv2.circle(img, (100, 120), 60, 255, -1)
    other = ((255 - xx) * 0.7 + yy * 0.3).astype(np.uint8)
    cv2.rectangle(other, (200, 30), (300, 200), 0, -1)
    resized = cv2.resize(cv2.resize(img, (200, 150)), (320, 240))
    assert hamming(dhash64(img), dhash64(resized)) <= 4, "positive control failed"
    assert hamming(dhash64(img), dhash64(other)) > 4, "negative control failed"
    idx = NearDupIndex()
    idx.add(dhash64(img), "a")
    assert idx.query(dhash64(resized)), "index positive control failed"
    assert not idx.query(dhash64(other)), "index negative control failed"
    print("[OK] self-test passed (dHash + index controls)")


# ---------------------------------------------------------------- report

def write_report(out_dir: Path, results: dict) -> None:
    lines = ["# Model Risk Audit", ""]
    lines += [
        "Static provenance (verified by code reading, see plan/session notes):",
        "",
        "- **Model inbreeding: CLEAR** - all labels trace to human annotations "
        "(TACO COCO via `export_taco_yolo_hardcase.py`, Roboflow-source annotations via "
        "`archive/build_super_yolo_dataset.py`); no script converts model predictions into labels. "
        "Upstream Roboflow community annotation quality is unverifiable.",
        "- **Split handling: CLEAR** - manifest-driven / source-preserving; no random re-splitting by default.",
        "- **Filename-marker cross-split contamination (merged_v5): CLEAR** - 0 in both directions.",
        "",
    ]
    for ds, r in results.get("datasets", {}).items():
        lines.append(f"## Dataset: {ds}")
        lines.append("")
        leak = r.get("leakage", {})
        lines.append(
            f"- Cross-split duplicates: **{len(leak.get('exact', []))} exact**, "
            f"**{len(leak.get('near', []))} near (Hamming<=4)**"
        )
        und = r.get("undecodable", 0)
        lines.append(f"- Undecodable images: **{und}**")
        for pair in leak.get("exact", [])[:10]:
            lines.append(f"  - EXACT {pair['splits']}: `{Path(pair['a']).name}` == `{Path(pair['b']).name}`")
        for pair in leak.get("near", [])[:10]:
            lines.append(
                f"  - NEAR ham={pair['hamming']} {pair['splits']}: `{Path(pair['a']).name}` ~ `{Path(pair['b']).name}`"
            )
        labels = r.get("labels")
        if labels:
            lines += ["", "| split | class | boxes | median area | tiny <1% img |", "|---|---|---:|---:|---:|"]
            for split, srow in labels.items():
                for c in srow["classes"]:
                    lines.append(
                        f"| {split} | {c['class']} | {c['boxes']} | {c['median_area_pct']}% | {c['tiny_lt1pct']}% |"
                    )
            lines += ["", "| split | label files | single-class img | noise |", "|---|---:|---:|---|"]
            for split, srow in labels.items():
                lines.append(
                    f"| {split} | {srow['label_files']} | {srow['single_class_image_pct']}% | "
                    f"{json.dumps(srow['noise'])} |"
                )
        domains = r.get("domains")
        if domains:
            lines += ["", "| split | class | total | field | studio |", "|---|---|---:|---:|---:|"]
            for split, per_class in domains.items():
                for cls, d in per_class.items():
                    lines.append(f"| {split} | {cls} | {d['total']} | {d['field']} | {d['studio']} |")
        lines.append("")

    cont = results.get("contamination", {})
    if cont:
        lines += ["## Cross-dataset eval contamination (hardcase eval vs other train sets)", ""]
        for name, per_split in cont.items():
            for split, row in per_split.items():
                lines.append(
                    f"- hardcase/{split} vs {name}/train: {row['exact_in_train']} exact + "
                    f"{row['near_in_train']} near of {row['n']} images"
                )
        lines.append("")

    fit = results.get("overfitting", {})
    if fit.get("available"):
        lines += [
            "## Overfitting signals (yolo26n_hardcase_v2_long, may be mid-training)",
            "",
            f"- Epochs done: {fit['epochs_done']}; best mAP50-95 {fit['best']['mAP50_95']:.4f} at epoch {fit['best']['epoch']}",
            f"- Last epoch: P {fit['last']['precision']:.3f} R {fit['last']['recall']:.3f} mAP50 {fit['last']['mAP50']:.3f}",
            f"- val-train cls-loss gap: {fit['val_train_cls_gap']} "
            "(mild gap normal; watch for val loss rising while train falls)",
            "",
        ]

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "MODEL_RISK_AUDIT.md").write_text("\n".join(lines), encoding="utf-8")
    (out_dir / "model_risk_audit.json").write_text(json.dumps(results, indent=2), encoding="utf-8")


# ---------------------------------------------------------------- main

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", choices=["hardcase", "super", "merged", "hardcase_classifier", "all"], default="all")
    ap.add_argument("--out", type=Path, default=ROOT / "runs" / "audits")
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()

    self_test()
    if args.self_test:
        return

    targets = list(DATASETS) if args.dataset == "all" else [args.dataset]
    results: dict = {"datasets": {}}
    scan_cache: dict[str, tuple[dict, dict]] = {}

    for name in targets:
        cfg = DATASETS[name]
        if not cfg["path"].exists():
            print(f"[WARN] {name}: missing {cfg['path']}")
            continue
        print(f"[INFO] {name}: listing images...")
        splits = split_images(cfg)
        all_paths = [p for ps in splits.values() for p in ps]
        print(f"[INFO] {name}: scanning {len(all_paths)} images ({ {k: len(v) for k, v in splits.items()} })")
        info = scan_images(all_paths)
        scan_cache[name] = (splits, info)

        r: dict = {
            "leakage": cross_split_dups(splits, info),
            "undecodable": sum(1 for v in info.values() if not v["decodable"]),
        }
        if cfg["layout"] == "yolo":
            r["labels"] = audit_labels(cfg)
        else:
            r["domains"] = merged_domain_bias(cfg)
        results["datasets"][name] = r
        print(
            f"[INFO] {name}: exact dups {len(r['leakage']['exact'])}, "
            f"near dups {len(r['leakage']['near'])}, undecodable {r['undecodable']}"
        )

    if "hardcase" in scan_cache:
        hc_splits, hc_info = scan_cache["hardcase"]
        contamination = {}
        for other in ("super", "merged"):
            if other in scan_cache:
                o_splits, o_info = scan_cache[other]
                train = o_splits.get("train", [])
                contamination[other] = cross_dataset_contamination(hc_splits, train, hc_info, o_info)
        results["contamination"] = contamination

    results["overfitting"] = parse_results_csv(RESULTS_CSV)
    write_report(args.out, results)
    print(f"[OK] Audit written to {args.out / 'MODEL_RISK_AUDIT.md'}")


if __name__ == "__main__":
    main()
