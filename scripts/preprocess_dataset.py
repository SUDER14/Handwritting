"""Fetches/caches the EMNIST-letters training dataset and writes a quick
sanity-check contact sheet + class-balance report to data/processed/.

Run this once before training (scripts/train_style_encoder.py also
downloads on demand, but running this first surfaces dataset problems
— e.g. connectivity, orientation — independently of the training loop).

Usage:
    python scripts/preprocess_dataset.py
"""
from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2
import numpy as np

from src.utils.config import resolve_path
from src.utils.emnist_source import EMNISTLettersHF, load_emnist_letters
from src.utils.logging_setup import get_logger

logger = get_logger(__name__)


def build_contact_sheet(dataset: EMNISTLettersHF, n_per_class: int = 4) -> np.ndarray:
    """One row per letter (a-z), `n_per_class` example glyphs each — for eyeballing
    orientation/label correctness before spending time training.
    """
    buckets: dict[int, list[np.ndarray]] = {i: [] for i in range(26)}
    for i in range(len(dataset)):
        tensor, label = dataset[i]
        if len(buckets[label]) >= n_per_class:
            continue
        buckets[label].append((tensor.squeeze(0).numpy() * 255).astype(np.uint8))
        if all(len(v) >= n_per_class for v in buckets.values()):
            break

    rows = []
    for label in range(26):
        imgs = buckets[label] or [np.zeros((28, 28), dtype=np.uint8)] * n_per_class
        while len(imgs) < n_per_class:
            imgs.append(np.zeros((28, 28), dtype=np.uint8))
        rows.append(np.hstack(imgs))
    return np.vstack(rows)


def report_class_balance(dataset: EMNISTLettersHF, sample_size: int = 5000) -> Counter:
    n = min(sample_size, len(dataset))
    counts = Counter(dataset[i][1] for i in range(n))
    return counts


def main():
    cache_dir = resolve_path("data/raw") / "hf_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Loading EMNIST-letters (train split) via Hugging Face Hub...")
    train_ds = load_emnist_letters(split="train", cache_dir=str(cache_dir))
    logger.info("Loaded %d training images.", len(train_ds))

    out_dir = resolve_path("data/processed")
    out_dir.mkdir(parents=True, exist_ok=True)

    sheet = build_contact_sheet(train_ds)
    sheet_path = out_dir / "emnist_contact_sheet.png"
    cv2.imwrite(str(sheet_path), sheet)
    logger.info("Wrote orientation/label sanity-check contact sheet to %s", sheet_path)

    counts = report_class_balance(train_ds)
    logger.info(
        "Class balance over %d sampled images: min=%d max=%d (26 classes)",
        sum(counts.values()), min(counts.values()), max(counts.values()),
    )


if __name__ == "__main__":
    main()
