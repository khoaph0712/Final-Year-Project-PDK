"""Stage 2 crop-classifier architecture (ConvNeXt-Tiny + 637 handcrafted features).

Single source of truth for the deployed classifier's architecture: web/server.py and
the eval scripts import from here so the class can't drift from the checkpoint
(runs/dl/convnext_ensemble_tuned/best_convnext_ensemble_tuned.pth). The historical
training scripts keep their inline copies as a record of what was actually run.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torchvision.models as models


class ConvNeXtFeatureExtractor(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.backbone = models.convnext_tiny(weights=None)
        self.backbone.classifier = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.flatten(self.backbone(x), 1)


class Stage3EnsembleClassifier(nn.Module):
    def __init__(self, num_classes: int = 7, dropout_rate: float = 0.3) -> None:
        super().__init__()
        self.convnext_extractor = ConvNeXtFeatureExtractor()
        self.classifier = nn.Sequential(
            nn.Linear(768 + 637, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate),
            nn.Linear(256, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(64, num_classes),
        )

    def forward(self, image_tensor: torch.Tensor, handcrafted_features_tensor: torch.Tensor) -> torch.Tensor:
        deep_features = self.convnext_extractor(image_tensor)
        fused_features = torch.cat((deep_features, handcrafted_features_tensor), dim=1)
        return self.classifier(fused_features)
