"""Concatenate report chapters and convert to .docx via pandoc.

Chapters are joined in filename order, so the numeric prefixes are the
table of contents. Evidence comments (<!-- src: ... -->) are left in the
intermediate Markdown; pandoc strips HTML comments on conversion.

    python build.py            # build docx
    python build.py --md-only  # just the concatenated markdown
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
BUILD = HERE / "_build"
MERGED = BUILD / "REPORT.md"
DOCX = HERE.parent / "WasteWise_FYP_Final_Report.docx"
# Styles only -- pandoc reads styles.xml from the reference doc and ignores its
# body, so the previous report doubles as the style template.
REFERENCE = HERE.parent / "UNDERGRADUATE FINAL YEAR PROJECT REPORT (updated).docx"


def chapters() -> list[Path]:
    return sorted(p for p in HERE.glob("*.md") if p.name[0].isdigit() or p.name.startswith("appendix"))


def concat() -> Path:
    BUILD.mkdir(exist_ok=True)
    parts = [p.read_text(encoding="utf-8").rstrip() for p in chapters()]
    MERGED.write_text("\n\n".join(parts) + "\n", encoding="utf-8")
    return MERGED


def to_docx() -> None:
    cmd = [
        "pandoc", str(MERGED), "-o", str(DOCX),
        f"--resource-path={ROOT}{';' if sys.platform == 'win32' else ':'}{HERE}",
        "--toc", "--toc-depth=3",
    ]
    if REFERENCE.exists():
        cmd.append(f"--reference-doc={REFERENCE}")
    subprocess.run(cmd, check=True, cwd=ROOT)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--md-only", action="store_true")
    a = ap.parse_args()

    found = chapters()
    if not found:
        sys.exit("no chapters found")
    concat()
    words = len(MERGED.read_text(encoding="utf-8").split())
    print(f"{len(found)} chapters -> {MERGED.relative_to(ROOT)} ({words:,} words)")

    if not a.md_only:
        to_docx()
        print(f"wrote {DOCX.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
