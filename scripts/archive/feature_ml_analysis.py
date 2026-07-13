"""Compatibility wrapper for the active feature ML analysis script.

The implementation is now kept in `scripts/feature_ml_analysis.py` because it
is still part of the active ML pipeline. This wrapper keeps older archive tools
that import or execute `scripts/archive/feature_ml_analysis.py` working.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path


_ACTIVE_SCRIPT = Path(__file__).resolve().parents[1] / "feature_ml_analysis.py"
_SPEC = importlib.util.spec_from_file_location("_active_feature_ml_analysis", _ACTIVE_SCRIPT)
if _SPEC is None or _SPEC.loader is None:
    raise ImportError(f"Cannot load active feature ML script: {_ACTIVE_SCRIPT}")

_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)

for _name in dir(_MODULE):
    if not _name.startswith("__"):
        globals()[_name] = getattr(_MODULE, _name)


if __name__ == "__main__":
    _MODULE.main()
