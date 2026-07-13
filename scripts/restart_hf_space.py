from __future__ import annotations

import os

from huggingface_hub import HfApi


def main() -> None:
    token = os.environ.get("HF_TOKEN", "").strip()
    if not token:
        raise SystemExit("HF_TOKEN is missing")
    HfApi(token=token).restart_space(repo_id="khoaphung/wastewise-ai")
    print("restart requested")


if __name__ == "__main__":
    main()
