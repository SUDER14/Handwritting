"""CNN character classifier architecture (future_work.md's top recognition item).

Small conv net trained on EMNIST-letters to classify a single normalized
28x28 glyph crop into one of 26 lowercase classes. This is what lets the
pipeline genuinely *look at the pixels* to decide which letter was written,
instead of trusting a user-supplied label (`LabeledRecognizer`) or matching
against a fixed printed font (`TemplateMatchingRecognizer`) — see
`src/recognition/cnn_recognizer.py` for the `Recognizer` wrapper and
`scripts/train_cnn_recognizer.py` for training (with augmentation so the
model generalizes to slanted, variable-stroke-width handwriting rather than
only upright EMNIST-style printing).
"""
from __future__ import annotations

import torch
from torch import nn


class CharCNN(nn.Module):
    """1x28x28 glyph -> logits over `num_classes` letter classes."""

    def __init__(self, num_classes: int = 26):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 28 -> 14
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 14 -> 7
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.conv(x))
