"""Count labels per class in a YOLO dataset folder (default merged_dataset_v3)."""
from pathlib import Path
from collections import Counter

import sys

ROOT = Path(sys.argv[1] if len(sys.argv) > 1 else r"C:\FYP_v2\merged_dataset_v3")
names = ["plastic", "glass", "metal", "paper", "cardboard", "organic", "other"]

totals = {split: Counter() for split in ("train", "valid", "test")}
img_totals = {split: 0 for split in totals}

for split in totals:
    lbl_dir = ROOT / split / "labels"
    if not lbl_dir.exists():
        continue
    for f in lbl_dir.glob("*.txt"):
        img_totals[split] += 1
        try:
            for line in f.read_text().splitlines():
                parts = line.strip().split()
                if not parts:
                    continue
                cid = int(parts[0])
                totals[split][cid] += 1
        except Exception:
            pass

print(f"{'class':<12}{'train':>10}{'valid':>10}{'test':>10}{'TOTAL':>12}")
print("-" * 54)
grand = Counter()
for cid, name in enumerate(names):
    row = [totals[s].get(cid, 0) for s in ("train", "valid", "test")]
    total = sum(row)
    grand[cid] = total
    print(f"{name:<12}{row[0]:>10}{row[1]:>10}{row[2]:>10}{total:>12}")

print("-" * 54)
tt = [sum(totals[s].values()) for s in ("train", "valid", "test")]
print(f"{'boxes':<12}{tt[0]:>10}{tt[1]:>10}{tt[2]:>10}{sum(tt):>12}")
print(f"{'images':<12}{img_totals['train']:>10}{img_totals['valid']:>10}{img_totals['test']:>10}{sum(img_totals.values()):>12}")

print("\n--- Train share (%) ---")
train_total = sum(totals['train'].values()) or 1
for cid, name in enumerate(names):
    pct = 100 * totals['train'].get(cid, 0) / train_total
    bar = "#" * int(pct / 2)
    print(f"{name:<12}{pct:6.2f}%  {bar}")
