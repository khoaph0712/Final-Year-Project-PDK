"""Build a styles-only reference.docx from the previous report.

pandoc starts the output archive from the reference doc, so any media the
reference carries is copied into every build as orphaned parts (74 images /
7 MB in the original). Only the style definitions are wanted, so this strips
media and its relationships. Re-run only if the source styling changes.
"""

from __future__ import annotations

import re
import shutil
import zipfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
SOURCE = HERE.parent / "UNDERGRADUATE FINAL YEAR PROJECT REPORT (updated).docx"
TARGET = HERE / "reference.docx"

DROP_PREFIXES = ("word/media/", "word/embeddings/")
# Relationship entries pointing at the dropped parts would dangle otherwise.
REL_DROP = re.compile(rb'<Relationship[^>]*Target="(?:media|embeddings)/[^"]*"[^>]*/>')


def main() -> None:
    if not SOURCE.exists():
        raise SystemExit(f"source not found: {SOURCE}")

    dropped = 0
    with zipfile.ZipFile(SOURCE) as src, zipfile.ZipFile(
        TARGET, "w", zipfile.ZIP_DEFLATED
    ) as out:
        for item in src.infolist():
            if item.filename.startswith(DROP_PREFIXES):
                dropped += 1
                continue
            data = src.read(item.filename)
            if item.filename.endswith(".rels"):
                data = REL_DROP.sub(b"", data)
            out.writestr(item, data)

    src_mb = SOURCE.stat().st_size / 1e6
    out_mb = TARGET.stat().st_size / 1e6
    print(f"dropped {dropped} media parts: {src_mb:.1f} MB -> {out_mb:.1f} MB")
    print(f"wrote {TARGET.name}")


if __name__ == "__main__":
    main()
