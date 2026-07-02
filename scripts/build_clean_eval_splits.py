#!/usr/bin/env python
"""Build leakage-quarantined eval splits from the model-risk audit.

Reads `runs/audits/model_risk_audit.json` (produced by audit_model_risks.py) and
creates CLEAN copies of the evaluation splits with every leaked image removed:

* hardcase detector: `external_datasets/yolo26_hardcase_clean_eval/{val,test}` +
  a `data.yaml` whose train still points at the ORIGINAL train set, so
  `model.val(data=..., split='val'|'test')` gives honest numbers.
* merged classifier: `data/merged_dataset_v5_clean_test/test/<class>/`.

Originals are never modified. Quarantined filenames + reasons are written to
CSV manifests for the report.
"""

from __future__ import annotations

import csv
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
AUDIT_JSON = ROOT / "runs" / "audits" / "model_risk_audit.json"
HARDCASE = ROOT / "external_datasets" / "yolo26_hardcase_dataset_v1"
HARDCASE_CLEAN = ROOT / "external_datasets" / "yolo26_hardcase_clean_eval"
MERGED = ROOT / "data" / "merged_dataset_v5"
MERGED_CLEAN = ROOT / "data" / "merged_dataset_v5_clean_test"
OUT_DIR = ROOT / "runs" / "audits"


def leaked_eval_files(audit: dict) -> tuple[dict[str, set[Path]], list[dict]]:
    """Split-scan order in the audit was alphabetical (test, train, val), so in a
    pair 'X<->Y' side a belongs to X and side b to Y. An eval image is quarantined
    when it near/exact-duplicates an image in ANY other split (train leakage, or
    val<->test cross-contamination which breaks test independence)."""
    quarantine: dict[str, set[Path]] = {"hardcase_val": set(), "hardcase_test": set(), "merged_test": set()}
    manifest: list[dict] = []

    def add(bucket: str, path_str: str, match: str, kind: str, pair: str) -> None:
        p = Path(path_str)
        if p not in quarantine[bucket]:
            quarantine[bucket].add(p)
            manifest.append(
                {"bucket": bucket, "file": p.name, "kind": kind, "pair": pair, "matched": Path(match).name}
            )

    hc = audit["datasets"]["hardcase"]["leakage"]
    for e in hc["near"] + hc["exact"]:
        kind = "near" if "hamming" in e else "exact"
        if e["splits"] == "train<->val":
            add("hardcase_val", e["b"], e["a"], kind, e["splits"])
        elif e["splits"] == "test<->train":
            add("hardcase_test", e["a"], e["b"], kind, e["splits"])
        elif e["splits"] == "test<->val":
            add("hardcase_test", e["a"], e["b"], kind, e["splits"])
            add("hardcase_val", e["b"], e["a"], kind, e["splits"])

    mg = audit["datasets"]["merged"]["leakage"]
    for e in mg["exact"] + mg["near"]:
        kind = "near" if "hamming" in e else "exact"
        if e["splits"] == "test<->train":
            add("merged_test", e["a"], e["b"], kind, e["splits"])

    return quarantine, manifest


def build_hardcase_clean(quarantine: dict[str, set[Path]]) -> dict:
    stats = {}
    for split in ("val", "test"):
        bad = quarantine[f"hardcase_{split}"]
        src_img = HARDCASE / split / "images"
        src_lbl = HARDCASE / split / "labels"
        dst_img = HARDCASE_CLEAN / split / "images"
        dst_lbl = HARDCASE_CLEAN / split / "labels"
        dst_img.mkdir(parents=True, exist_ok=True)
        dst_lbl.mkdir(parents=True, exist_ok=True)
        kept = removed = 0
        for img in src_img.iterdir():
            if img in bad:
                removed += 1
                continue
            shutil.copy2(img, dst_img / img.name)
            lbl = src_lbl / (img.stem + ".txt")
            if lbl.exists():
                shutil.copy2(lbl, dst_lbl / lbl.name)
            kept += 1
        stats[split] = {"kept": kept, "removed": removed}

    names = ["plastic", "glass", "metal", "paper", "cardboard", "organic"]
    yaml_text = (
        f"# Leakage-quarantined eval splits; train points at the ORIGINAL train set.\n"
        f"train: {HARDCASE / 'train' / 'images'}\n"
        f"val: {HARDCASE_CLEAN / 'val' / 'images'}\n"
        f"test: {HARDCASE_CLEAN / 'test' / 'images'}\n"
        f"nc: {len(names)}\n"
        f"names: {names}\n"
    )
    (HARDCASE_CLEAN / "data.yaml").write_text(yaml_text, encoding="utf-8")
    return stats


def build_merged_clean(quarantine: dict[str, set[Path]]) -> dict:
    bad = quarantine["merged_test"]
    kept = removed = 0
    for cls_dir in sorted(d for d in (MERGED / "test").iterdir() if d.is_dir()):
        dst = MERGED_CLEAN / "test" / cls_dir.name
        dst.mkdir(parents=True, exist_ok=True)
        for img in cls_dir.iterdir():
            if img in bad:
                removed += 1
                continue
            shutil.copy2(img, dst / img.name)
            kept += 1
    return {"test": {"kept": kept, "removed": removed}}


def main() -> None:
    audit = json.loads(AUDIT_JSON.read_text(encoding="utf-8"))
    quarantine, manifest = leaked_eval_files(audit)

    print("[INFO] Quarantine sizes:", {k: len(v) for k, v in quarantine.items()})
    hc_stats = build_hardcase_clean(quarantine)
    mg_stats = build_merged_clean(quarantine)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with (OUT_DIR / "quarantine_manifest.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=["bucket", "file", "kind", "pair", "matched"])
        writer.writeheader()
        writer.writerows(manifest)

    summary = {"hardcase": hc_stats, "merged": mg_stats, "quarantined": {k: len(v) for k, v in quarantine.items()}}
    (OUT_DIR / "clean_eval_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"[OK] Clean hardcase eval: {HARDCASE_CLEAN / 'data.yaml'}")
    print(f"[OK] Clean merged test:   {MERGED_CLEAN / 'test'}")
    print(f"[OK] Manifest:            {OUT_DIR / 'quarantine_manifest.csv'}")


if __name__ == "__main__":
    main()
