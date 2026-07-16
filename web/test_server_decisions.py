"""Regression tests for the pure decision logic in web/server.py.

These functions carry A/B-measured tuning (see runs/audits/pipeline_bin_decision_eval_*.json);
the tests pin the tier ordering and constants so a refactor can't silently change them.
No models are loaded - everything here is pure.

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
    # The waste gate must be stricter than the review bar.
    assert server.WASTE_GATE_CONF > server.WASTE_REVIEW_CONF
    # The box-objectness gate must sit above the candidate-generation threshold,
    # otherwise every proposed box would auto-qualify as "confident waste".
    assert server.YOLO_GATE_CONF > server.YOLO_CONF


def state(label_key="plastic", label_conf=0.9, crop_key="plastic", crop_conf=0.9,
          yolo_key="plastic", yolo_score=0.9, scene_is_empty=False, class_threshold=0.25):
    return server.estimate_detection_waste_state(
        label_key, label_conf, crop_key, crop_conf, yolo_key, yolo_score,
        scene_is_empty, class_threshold,
    )


def test_waste_state_tiers():
    # Tier 1: empty scene wins over everything.
    assert state(scene_is_empty=True)[0] == "not_waste"
    # Tier 2: background label - confident background crop with no box is not waste...
    assert state(label_key="Background", crop_key="Background", crop_conf=0.8, yolo_key=None)[0] == "not_waste"
    # ...anything less certain goes to review.
    assert state(label_key="Background", crop_key="Background", crop_conf=0.5, yolo_key=None)[0] == "review"
    assert state(label_key="Background", crop_conf=0.8)[0] == "review"  # yolo box present
    # Tier 3: below the per-class threshold -> review.
    assert state(label_conf=0.10, class_threshold=0.25)[0] == "review"
    # Tier 4: joint confidence gate (classifier >= WASTE_GATE_CONF and box >= YOLO_GATE_CONF).
    assert state(label_conf=server.WASTE_GATE_CONF, yolo_score=server.YOLO_GATE_CONF)[0] == "waste"
    # Tier 5: strong localizer agreement rescues a mid-confidence label.
    assert state(label_conf=0.40, class_threshold=0.25, yolo_key="plastic", yolo_score=0.50)[0] == "waste"
    # Weak evidence tiers -> review, never waste.
    assert state(label_conf=0.40, class_threshold=0.25, yolo_key="metal", yolo_score=0.20)[0] == "review"
    assert state(label_conf=0.40, class_threshold=0.25, crop_key="Background", yolo_key=None, yolo_score=0.0)[0] == "review"


def det(waste_state="waste", confidence=90):
    return {"wasteState": waste_state, "confidence": confidence}


def test_final_decision():
    review_bar = int(server.WASTE_REVIEW_CONF * 100)
    # Empty scene short-circuits.
    assert server.choose_final_decision([det()], "plastic", 0.9, True)[0] == "not_waste"
    # One strong waste detection -> waste.
    assert server.choose_final_decision([det(confidence=review_bar)], "plastic", 0.9, False)[0] == "waste"
    # Waste detections exist but all below the review bar -> review, not waste.
    assert server.choose_final_decision([det(confidence=review_bar - 1)], "plastic", 0.9, False)[0] == "review"
    # All detections judged not_waste -> not_waste.
    assert server.choose_final_decision([det("not_waste"), det("not_waste")], "plastic", 0.9, False)[0] == "not_waste"
    # Mixed not_waste + review -> review.
    assert server.choose_final_decision([det("not_waste"), det("review")], "plastic", 0.9, False)[0] == "review"
    # No detections: confident background scene -> not_waste, anything else -> review.
    assert server.choose_final_decision([], "Background", 0.8, False)[0] == "not_waste"
    assert server.choose_final_decision([], "Background", 0.5, False)[0] == "review"
    assert server.choose_final_decision([], "plastic", 0.9, False)[0] == "review"


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"[OK] {name}")
    print("All decision-logic tests passed.")
