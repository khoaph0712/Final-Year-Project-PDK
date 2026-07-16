from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import urllib.error
import urllib.request
import stat
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEPLOY_DIR = Path(r"C:\tmp\wastewise-hf-space")
MODEL_FILES = [
    ROOT / "runs" / "dl" / "convnext_ensemble_tuned" / "best_convnext_ensemble_tuned.pth",
    ROOT / "runs" / "dl" / "convnext_ensemble_tuned" / "handcrafted_scaler.npz",
    ROOT / "models" / "trained" / "yolov11_detector" / "best.pt",
]
SUPPORT_FILES = [
    ROOT / "scripts" / "custom_feature_extractor.py",
    ROOT / "scripts" / "stage2_model.py",
]


def run(args: list[str], cwd: Path | None = None, env: dict[str, str] | None = None) -> None:
    subprocess.run(args, cwd=str(cwd) if cwd else None, env=env, check=True)


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
            # let the git push below fail loudly if it actually doesn't.
            print(f"Skipping create (402 Payment Required - assuming {repo_id} already exists)")
        else:
            print(body, file=sys.stderr)
            raise

    if DEPLOY_DIR.exists():
        remove_tree(DEPLOY_DIR)
    DEPLOY_DIR.mkdir(parents=True)

    for name in ["index.html", "styles.css", "app.js", "README.md", ".nojekyll"]:
        copy_tree_item(ROOT / "web" / name, DEPLOY_DIR / "web" / name)
    copy_tree_item(ROOT / "web" / "assets", DEPLOY_DIR / "web" / "assets")
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
opencv-python-headless>=4.9.0
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

Full web deployment for the WasteWise FYP application.

- Frontend: `web/index.html`, `web/styles.css`, `web/app.js`
- Backend: `web/server.py`
- Models: hard-case ConvNeXt + 637-feature classifier and YOLO26m localizer
- API: `/api/predict`
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

    run(["git", "init"], DEPLOY_DIR)
    run(["git", "checkout", "-B", "main"], DEPLOY_DIR)
    run(["git", "lfs", "install", "--local"], DEPLOY_DIR)
    run(["git", "config", "user.name", "Khoa Phung"], DEPLOY_DIR)
    run(["git", "config", "user.email", "91509922+letouch07@users.noreply.github.com"], DEPLOY_DIR)
    run(["git", "add", "."], DEPLOY_DIR)
    run(["git", "commit", "-m", "Deploy WasteWise AI full model app"], DEPLOY_DIR)
    # Username lives in the remote URL, so git only asks for the password; GIT_ASKPASS
    # answers it from the HF_TOKEN env var. Unlike the previous credential.helper=store
    # approach, the token is never written to ~/.git-credentials (which persisted the
    # plaintext token on disk if the push crashed before cleanup).
    run(["git", "remote", "add", "origin", f"https://__token__@huggingface.co/spaces/{repo_id}"], DEPLOY_DIR)

    askpass = DEPLOY_DIR.parent / "hf_askpass.bat"
    askpass.write_text("@echo off\necho %HF_TOKEN%\n", encoding="ascii")
    push_env = dict(os.environ)
    push_env["GIT_TERMINAL_PROMPT"] = "0"
    push_env["GIT_ASKPASS"] = str(askpass)
    push_env["HF_TOKEN"] = token
    try:
        run(["git", "push", "origin", "main", "--force"], DEPLOY_DIR, env=push_env)
    finally:
        askpass.unlink(missing_ok=True)

    print(f"https://huggingface.co/spaces/{repo_id}")
    print(f"https://{username}-{space_name}.hf.space")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
