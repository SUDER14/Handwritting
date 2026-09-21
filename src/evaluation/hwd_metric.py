"""HWD (Handwriting Distance, Pippi et al., BMVC 2023) via the OFFICIAL implementation (third_party/HWD, aimagelab/HWD).

Official definition (hwd.scores.HWDScore): images -> PaddingSquareHeight -> resize to height 32 (nearest) -> the
ImageNet-VGG16 backbone pretrained on 10,400 font classes (`VGG16_class_10400.pth`, downloaded by the official code) ->
every spatial position of the conv feature map is one 512-d token -> HWD(A, B) = Euclidean distance between the MEAN
token of set A and the MEAN token of set B. We call the official backbone and transforms unchanged; the only thing added
here is a per-line cache of (sum of tokens, token count), which makes the mean of any subset of lines exact and cheap
(the mean over a set's tokens = sum of line sums / sum of line counts) -- identical to the official value.

Images: uint8 grayscale, dark ink on light paper (IAM native polarity), any width.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from src.utils.config import resolve_path

HWD_REPO = resolve_path("third_party/HWD")
HEIGHT = 32


def _official():
    if str(HWD_REPO) not in sys.path:
        sys.path.insert(0, str(HWD_REPO))
    from hwd.scores import HWDScore           # noqa: E402  (official code)
    return HWDScore


class HWDFeatures:
    """Line -> (feature sum [512], token count), using the official HWD backbone/transforms."""

    def __init__(self, height: int = HEIGHT):
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.score = _official()(height=height)
        self.backbone = self.score.backbone
        self.transforms = self.score.transforms
        self.backbone.model.eval()

    @torch.inference_mode()
    def line(self, gray: np.ndarray) -> tuple[np.ndarray, int]:
        img = Image.fromarray(np.asarray(gray, np.uint8)).convert("RGB")
        x = self.transforms(img).unsqueeze(0)
        _, _, h, w = x.shape
        tokens = self.backbone.model(x)[0]                    # (H'*W', 512): official AdjustDims flattening
        return tokens.sum(0).numpy().astype(np.float64), int(tokens.shape[0])


def set_mean(entries) -> np.ndarray:
    """Mean token over a set of lines, from their (sum, count) entries."""
    total = sum(s for s, _ in entries)
    n = sum(c for _, c in entries)
    return total / n


def hwd(mean_a: np.ndarray, mean_b: np.ndarray) -> float:
    """Official HWD = Euclidean distance between the two sets' mean tokens."""
    return float(np.linalg.norm(mean_a - mean_b))
