"""Download extra Roboflow datasets to fix weak classes.

- Clean food-waste  -> replaces noisy `organic`
- Cigarette butts   -> targeted `other`
- Waste-detection   -> styrofoam / wrappers / non-biodegradable for `other`
"""
from pathlib import Path
from roboflow import Roboflow

RF_API_KEY = "kPxvBKgvQPxtChunxphQ"
ROOT = Path(r"C:\FYP_v2")

TARGETS = [
    # (workspace, project, out_folder)
    ("amir-wsfdf", "food-waste-detection-yolo-v8-ksmgh", "rf_food_waste"),
    ("using-ai-to-detect-cigarettes", "cigarette-butts-1fukc-p2oxb", "rf_cigarettes"),
    ("yolov8-ofcbj", "waste-detection-saah1", "rf_waste_saah1"),
]


def download_latest(rf: Roboflow, workspace: str, project_slug: str, out: Path) -> bool:
    try:
        project = rf.workspace(workspace).project(project_slug)
    except Exception as e:
        print(f"  [SKIP] cannot access {workspace}/{project_slug}: {e}")
        return False

    versions = []
    try:
        versions = project.versions()
    except Exception as e:
        print(f"  [WARN] could not list versions: {e}; will try v1..v5")

    candidates = []
    if versions:
        for v in versions:
            vid = None
            for attr in ("version", "id"):
                try:
                    val = getattr(v, attr, None)
                    if val is None and isinstance(v, dict):
                        val = v.get(attr)
                    if val is not None:
                        vid = int(str(val).split("/")[-1])
                        break
                except Exception:
                    continue
            if vid is not None:
                candidates.append(vid)
        candidates = sorted(set(candidates), reverse=True)
    if not candidates:
        candidates = [5, 4, 3, 2, 1]

    for vnum in candidates:
        try:
            print(f"  -> trying v{vnum}")
            project.version(vnum).download("yolov8", location=str(out))
            print(f"  [OK] downloaded v{vnum} to {out}")
            return True
        except Exception as e:
            print(f"  v{vnum} failed: {e}")
    return False


def main() -> None:
    rf = Roboflow(api_key=RF_API_KEY)
    for workspace, project_slug, folder in TARGETS:
        out = ROOT / folder
        if out.exists() and any(out.iterdir()):
            print(f"[SKIP] {folder} already exists")
            continue
        print(f"\n[+] Downloading {workspace}/{project_slug} -> {folder}")
        download_latest(rf, workspace, project_slug, out)


if __name__ == "__main__":
    main()
