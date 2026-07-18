"""Regression tests for the pure helper logic in web/server.py.

Pins class-key normalization and the routing/threshold constant invariants so a refactor
can't silently change them. The S6 waste-state decision layer was removed 2026-07-18, so its
tier tests are gone with it. No models are loaded - everything here is pure.

Run with either:
    .venv311\\Scripts\\python.exe web\\test_server_decisions.py
    .venv311\\Scripts\\python.exe -m pytest web\\test_server_decisions.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import server


def test_normalize_class_key():
    assert server.normalize_class_key("plastic") == "plastic"
    assert server.normalize_class_key("PLASTIC ") == "plastic"
    assert server.normalize_class_key("bottle") == "plastic"
    assert server.normalize_class_key("plastic_bottle") == "plastic"
    assert server.normalize_class_key("Aluminium") == "metal"
    assert server.normalize_class_key("cardboard box") == "cardboard"
    assert server.normalize_class_key("food waste") == "organic"
    assert server.normalize_class_key("glass jar") == "glass"  # substring match
    # Generic labels carry no material information.
    assert server.normalize_class_key("trash") is None
    assert server.normalize_class_key("garbage") is None
    assert server.normalize_class_key("") is None
    assert server.normalize_class_key(None) is None
    assert server.normalize_class_key("shoe") is None


def test_constants_invariants():
    # Every classifier class must have a bin route.
    assert set(server.ROUTES) == set(server.CLASSIFIER_CLASSES)
    # The box-objectness gate must sit above the candidate-generation threshold.
    assert server.YOLO_GATE_CONF > server.YOLO_CONF
    # The reliable-detection bar is a probability in (0, 1).
    assert 0.0 < server.MIN_RELIABLE_CONF < 1.0


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"[OK] {name}")
    print("All decision-logic tests passed.")
