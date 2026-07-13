from __future__ import annotations

import csv
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def count_classifier(root: Path) -> dict[str, dict[str, int]]:
    out: dict[str, dict[str, int]] = {}
    for split in ("train", "val", "test"):
        split_dir = root / split
        out[split] = {}
        if not split_dir.exists():
            continue
        for class_dir in sorted(item for item in split_dir.iterdir() if item.is_dir()):
            out[split][class_dir.name] = sum(1 for item in class_dir.iterdir() if item.suffix.lower() in IMAGE_EXTS)
    return out


def parse_yolo_names(data_yaml: Path) -> tuple[Path, list[str]]:
    root = data_yaml.parent
    names: list[str] = []
    in_names = False
    for raw in data_yaml.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if line.startswith("path:"):
            value = line.split(":", 1)[1].strip()
            root = Path(value)
        elif line == "names:":
            in_names = True
        elif in_names and line.startswith("- "):
            names.append(line[2:].strip())
        elif in_names and line and ":" in line:
            break
    if not root.is_absolute():
        root = data_yaml.parent / root
    return root, names


def count_yolo(data_yaml: Path) -> dict[str, dict[str, int]]:
    root, names = parse_yolo_names(data_yaml)
    out: dict[str, dict[str, int]] = {}
    for split in ("train", "val", "test"):
        counts = Counter()
        label_dir = root / split / "labels"
        if label_dir.exists():
            for label_file in label_dir.glob("*.txt"):
                for line in label_file.read_text(encoding="utf-8").splitlines():
                    parts = line.strip().split()
                    if not parts:
                        continue
                    class_id = int(float(parts[0]))
                    label = names[class_id] if 0 <= class_id < len(names) else str(class_id)
                    counts[label] += 1
        out[split] = dict(counts)
    return out


def print_table(title: str, data: dict[str, dict[str, int]]) -> None:
    print(f"\n{title}")
    classes = sorted({class_name for counts in data.values() for class_name in counts})
    print("class,train,val,test,total")
    totals = []
    for class_name in classes:
        row = [data.get(split, {}).get(class_name, 0) for split in ("train", "val", "test")]
        total = sum(row)
        totals.append(total)
        print(f"{class_name},{row[0]},{row[1]},{row[2]},{total}")
    if totals:
        print(f"min_total,{min(totals)}")
        print(f"max_total,{max(totals)}")
        print(f"max/min,{max(totals) / max(1, min(totals)):.2f}x")


def main() -> None:
    print_table("classifier:data/hard_case_classifier_v1", count_classifier(ROOT / "data" / "hard_case_classifier_v1"))
    print_table("yolo:external_datasets/yolo26_hardcase_dataset_v1", count_yolo(ROOT / "external_datasets" / "yolo26_hardcase_dataset_v1" / "data.yaml"))

    manifest = ROOT / "data" / "hard_case_classifier_v1" / "source_manifest.csv"
    if manifest.exists():
        by_source = defaultdict(Counter)
        with manifest.open("r", encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                by_source[row.get("source_id", "unknown")][row.get("class", "unknown")] += 1
        print("\nclassifier source mix")
        for source, counts in by_source.items():
            print(source, dict(counts))


if __name__ == "__main__":
    main()
