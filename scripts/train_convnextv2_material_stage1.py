"""Train a ConvNeXtV2 material classifier for Stage 1.

This intentionally expects a material-labelled folder dataset. Do not feed
Trashify Stage 0 labels here, because `trash`, `hand`, and `bin` are not
material classes.
"""

from __future__ import annotations

import argparse
from train_convnextv2_stage0_gate import main as stage0_main


if __name__ == "__main__":
    # The Stage 0 trainer is generic enough for image-folder classification;
    # this wrapper documents the intended Stage 1 entry point. Use with:
    #   --data data/hard_case_classifier_v1/data.yaml --out runs/dl/convnextv2_material_stage1
    stage0_main()
