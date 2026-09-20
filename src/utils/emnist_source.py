"""EMNIST-letters loader via the Hugging Face Hub.

torchvision's built-in `torchvision.datasets.EMNIST` downloads from a
NIST-hosted mirror that has served an EXPIRED TLS certificate for an
extended period (a known upstream issue, not fixable client-side by
updating a CA bundle, and not something we work around by disabling TLS
verification). Hugging Face's Hub mirror of the same academic dataset
(`tanganke/emnist_letters`) serves the same 26-class, 28x28,
case-merged letters data over a normal valid-cert connection, so we use
that instead. See docs/limitations.md.

Note: this dataset (like the original NIST-format EMNIST) stores images
rotated 90 degrees and mirrored relative to normal reading orientation —
we apply the standard EMNIST fix (rotate -90, then horizontal flip),
verified visually against known labels during development.
"""
from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset

HF_DATASET_ID = "tanganke/emnist_letters"


def _fix_emnist_orientation(pil_img):
    return pil_img.rotate(-90, expand=True).transpose(0)  # 0 == PIL.Image.FLIP_LEFT_RIGHT


class EMNISTLettersHF(Dataset):
    """PyTorch Dataset wrapper around the Hugging Face EMNIST-letters port.

    Yields (image_tensor[1,28,28] in [0,1], class_idx in [0,25]) where
    class_idx 0..25 corresponds to a..z (EMNIST letters merges upper/lower
    case into one class per letter).
    """

    def __init__(self, split: str = "train", cache_dir: str | None = None):
        from datasets import load_dataset  # lazy import: optional/heavy dependency

        self.hf_dataset = load_dataset(HF_DATASET_ID, split=split, cache_dir=cache_dir)

    def __len__(self) -> int:
        return len(self.hf_dataset)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        item = self.hf_dataset[idx]
        img = _fix_emnist_orientation(item["image"].convert("L"))
        arr = np.array(img, dtype=np.float32) / 255.0
        tensor = torch.from_numpy(arr).unsqueeze(0)
        label = int(item["label"])  # already 0..25, A..Z ClassLabel order
        return tensor, label


def load_emnist_letters(split: str = "train", cache_dir: str | None = None) -> EMNISTLettersHF:
    return EMNISTLettersHF(split=split, cache_dir=cache_dir)
