"""Moved to scripts/custom_feature_extractor.py (it is a live production
dependency of web/server.py, not an archived experiment). This shim keeps the
archived scripts that append scripts/archive to sys.path working unchanged.

Loaded by explicit file path because this shim shares the real module's name:
a plain `from custom_feature_extractor import ...` here would import itself."""

import importlib.util
from pathlib import Path

_real_path = Path(__file__).resolve().parents[1] / "custom_feature_extractor.py"
_spec = importlib.util.spec_from_file_location("_custom_feature_extractor_real", _real_path)
_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_module)
globals().update({name: value for name, value in vars(_module).items() if not name.startswith("_")})
