from __future__ import annotations

import json
import os
import shutil
import sys
import urllib.error
import urllib.request
import stat
from pathlib import Path

from huggingface_hub import HfApi


ROOT = Path(__file__).resolve().parents[1]
DEPLOY_DIR = Path(r"C:\tmp\wastewise-hf-space")
MODEL_FILES = [
    ROOT / "runs" / "dl" / "convnext_ensemble_tuned" / "best_convnext_ensemble_tuned.pth",
    ROOT / "runs" / "dl" / "convnext_ensemble_tuned" / "handcrafted_scaler.npz",
    ROOT / "models" / "trained" / "yolov11_detector" / "best.pt",
    ROOT / "models" / "places365" / "resnet18_places365.pth.tar",
    ROOT / "models" / "places365" / "categories_places365.txt",
    ROOT / "models" / "places365" / "IO_places365.txt",
]
SUPPORT_FILES = [
    ROOT / "scripts" / "custom_feature_extractor.py",
    ROOT / "scripts" / "stage2_model.py",
]


def request_json(url: str, token: str, method: str = "GET", payload: dict | None = None) -> dict:
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=data,
        method=method,
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        },
    )
    with urllib.request.urlopen(request, timeout=60) as response:
        return json.loads(response.read().decode("utf-8"))


def write_file(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8", newline="\n")


def copy_tree_item(source: Path, target: Path) -> None:
    if source.is_dir():
        if target.exists():
            shutil.rmtree(target)
        shutil.copytree(source, target)
    else:
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)


def remove_tree(path: Path) -> None:
    def handle_error(function, value, exc_info):
        try:
            os.chmod(value, stat.S_IWRITE)
            function(value)
        except Exception:
            raise exc_info[1]

    shutil.rmtree(path, onerror=handle_error)


def main() -> int:
    token = os.environ.get("HF_TOKEN", "").strip()
    if not token:
        print("HF_TOKEN is missing", file=sys.stderr)
        return 2

    whoami = request_json("https://huggingface.co/api/whoami-v2", token)
    username = whoami["name"]
    print(f"Authenticated as: {username}")
    space_name = "wastewise-ai"
    repo_id = f"{username}/{space_name}"

    try:
        request_json(
            "https://huggingface.co/api/repos/create",
            token,
            method="POST",
            payload={
                "name": space_name,
                "type": "space",
                "sdk": "docker",
                "private": False,
            },
        )
        print(f"Created Space: {repo_id}")
    except urllib.error.HTTPError as error:
        body = error.read().decode("utf-8", "replace")
        if error.code == 409 or "already exists" in body.lower():
            print(f"Space already exists: {repo_id}")
        elif error.code == 402:
            # Free-tier accounts can't provision a NEW docker Space, but pushing to an
            # EXISTING one (already created, already has its hardware tier set) doesn't
            # need this create call - only creation is paywalled. Assume it exists and
            # let the upload_folder call below fail loudly if it actually doesn't.
            print(f"Skipping create (402 Payment Required - assuming {repo_id} already exists)")
        else:
            print(body, file=sys.stderr)
            raise

    if DEPLOY_DIR.exists():
        remove_tree(DEPLOY_DIR)
    DEPLOY_DIR.mkdir(parents=True)

    # Backend-only Space: the frontend is served by Vercel (wastewise-fyp.vercel.app,
    # which proxies /api/* here via vercel.json rewrites). Shipping the frontend too
    # duplicated the site on the *.hf.space domain.
    copy_tree_item(ROOT / "web" / "server.py", DEPLOY_DIR / "web" / "server.py")

    for model_file in MODEL_FILES:
        relative = model_file.relative_to(ROOT)
        copy_tree_item(model_file, DEPLOY_DIR / relative)
    for support_file in SUPPORT_FILES:
        relative = support_file.relative_to(ROOT)
        copy_tree_item(support_file, DEPLOY_DIR / relative)

    write_file(
        DEPLOY_DIR / "Dockerfile",
        """FROM python:3.11-slim

WORKDIR /app

ENV TF_CPP_MIN_LOG_LEVEL=3
ENV MPLCONFIGDIR=/tmp/matplotlib
ENV HF_HOME=/tmp/huggingface
ENV WASTEWISE_API_ONLY=1

RUN apt-get update \\
    && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 git git-lfs \\
    && rm -rf /var/lib/apt/lists/*

COPY requirements-space.txt .
RUN pip install --no-cache-dir -r requirements-space.txt

COPY . .

EXPOSE 7860
CMD ["python", "web/server.py", "--host", "0.0.0.0", "--port", "7860"]
""",
    )
    write_file(
        DEPLOY_DIR / "requirements-space.txt",
        """numpy>=1.26.0,<2.0
opencv-python-headless>=4.9.0,<5
torch==2.4.1
torchvision==0.19.1
ultralytics==8.4.92
""",
    )
    write_file(
        DEPLOY_DIR / "README.md",
        f"""---
title: WasteWise AI
emoji: ♻️
colorFrom: green
colorTo: blue
sdk: docker
pinned: false
---

# WasteWise AI

Backend model API for the WasteWise FYP application. The frontend is served
by Vercel (https://wastewise-fyp.vercel.app), which proxies `/api/*` here.

- Backend: `web/server.py` (API-only mode)
- Models: hard-case ConvNeXt + 637-feature classifier and YOLO26m localizer
- API: `GET /api/health`, `POST /api/predict`
""",
    )
    write_file(
        DEPLOY_DIR / ".gitattributes",
        """*.h5 filter=lfs diff=lfs merge=lfs -text
*.pt filter=lfs diff=lfs merge=lfs -text
*.pth filter=lfs diff=lfs merge=lfs -text
*.jpg filter=lfs diff=lfs merge=lfs -text
*.jpeg filter=lfs diff=lfs merge=lfs -text
*.png filter=lfs diff=lfs merge=lfs -text
*.webp filter=lfs diff=lfs merge=lfs -text
""",
    )

    # Upload via the Hub HTTP API (token auth), not a git push. The previous git-over-HTTPS
    # path fed HF_TOKEN to git through a .bat GIT_ASKPASS shim, which is fragile on Windows:
    # git couldn't reliably invoke the .bat, so the push authenticated with no token and HF
    # rejected it ("Password authentication in git is no longer supported"). upload_folder
    # handles auth, large-file (LFS) upload and the commit in one call, and skips unchanged
    # files (so the ~100 MB of model weights are not re-sent when only code changed). Requires
    # the token to have WRITE permission on the Space.
    #
    # delete_patterns="*" restores the mirror semantics of the old `git push --force`: any
    # remote file no longer present in DEPLOY_DIR (e.g. a retired script or renamed model
    # checkpoint) is deleted from the Space instead of lingering forever. `.gitattributes` is
    # exempt from deletion by the library itself.
    HfApi(token=token).upload_folder(
        folder_path=str(DEPLOY_DIR),
        repo_id=repo_id,
        repo_type="space",
        commit_message="Deploy WasteWise AI full model app",
        delete_patterns="*",
    )

    print(f"https://huggingface.co/spaces/{repo_id}")
    print(f"https://{username}-{space_name}.hf.space")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
