"""Train the CNN character recognizer (future_work.md's top recognition item).

Trains `src/recognition/cnn_model.py::CharCNN` on EMNIST-letters to classify
a single normalized glyph crop as one of 26 lowercase letters. Unlike the
style VAE (scripts/train_style_encoder.py), which learns to *generate*
glyphs, this trains a plain discriminative classifier so
`src/recognition/cnn_recognizer.py::CNNRecognizer` can actually detect what
was written from pixels, instead of trusting a user-supplied label.

Augmentation (random affine warps + morphological erosion/dilation + noise)
is applied during training so the classifier generalizes beyond EMNIST's
upright, uniform-stroke printing to the slanted, variable-stroke-width
letterforms typical of real (including cursive-influenced) handwriting —
see docs/limitations.md and docs/future_work.md for why full
segmentation-free cursive-word recognition (CRNN+CTC) remains a larger,
separate future item; this model answers "which of a-z is this segmented
glyph", robustly, which is what the current per-character pipeline needs.

Usage:
    python scripts/train_cnn_recognizer.py
"""
from __future__ import annotations

import random
import string
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, Subset

from src.recognition.cnn_model import CharCNN
from src.utils.config import load_config, resolve_path
from src.utils.emnist_source import load_emnist_letters
from src.utils.logging_setup import get_logger

logger = get_logger(__name__)

LABELS = list(string.ascii_lowercase)


class AugmentedEMNIST(Dataset):
    """Wraps an EMNIST-letters split with random affine warp + morphology + noise.

    Applied only to the training split (see `get_dataloaders`) — this is what
    pushes the classifier to tolerate slant, variable stroke width, and minor
    stroke breaks/joins rather than memorizing EMNIST's clean, upright style.
    """

    def __init__(self, base: Dataset, augment: bool, seed: int = 0):
        self.base = base
        self.augment = augment
        self.rng = random.Random(seed)

    def __len__(self) -> int:
        return len(self.base)

    def _augment(self, img_u8: np.ndarray) -> np.ndarray:
        h, w = img_u8.shape
        angle = self.rng.uniform(-18, 18)
        shear = self.rng.uniform(-14, 14)
        scale = self.rng.uniform(0.85, 1.15)
        tx = self.rng.uniform(-0.08, 0.08) * w
        ty = self.rng.uniform(-0.08, 0.08) * h

        center = (w / 2.0, h / 2.0)
        rot = cv2.getRotationMatrix2D(center, angle, scale)
        shear_rad = np.radians(shear)
        shear_mat = np.array([[1, np.tan(shear_rad), 0], [0, 1, 0]], dtype=np.float32)
        affine = shear_mat @ np.vstack([rot, [0, 0, 1]]).astype(np.float32)
        affine[0, 2] += tx
        affine[1, 2] += ty
        warped = cv2.warpAffine(img_u8, affine, (w, h), flags=cv2.INTER_LINEAR, borderValue=0)

        # Random erosion/dilation to vary stroke width — real ink strokes
        # vary far more than EMNIST's rendered characters.
        if self.rng.random() < 0.5:
            k = self.rng.choice([2, 3])
            kernel = np.ones((k, k), np.uint8)
            warped = cv2.dilate(warped, kernel, iterations=1) if self.rng.random() < 0.5 \
                else cv2.erode(warped, kernel, iterations=1)

        if self.rng.random() < 0.5:
            noise = np.random.default_rng(self.rng.randrange(2 ** 31)).normal(0, 12, warped.shape)
            warped = np.clip(warped.astype(np.float32) + noise, 0, 255).astype(np.uint8)

        return warped

    def __getitem__(self, idx: int):
        tensor, label = self.base[idx]
        img_u8 = (tensor.squeeze(0).numpy() * 255).astype(np.uint8)
        if self.augment:
            img_u8 = self._augment(img_u8)
        out = torch.from_numpy(img_u8.astype(np.float32) / 255.0).unsqueeze(0)
        return out, label


def get_dataloaders(cfg: dict):
    train_cfg = cfg["recognition_training"]
    cache_dir = resolve_path("data/raw") / "hf_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    full_train = load_emnist_letters(split="train", cache_dir=str(cache_dir))

    max_n = train_cfg.get("max_train_samples", 20000)
    n = min(max_n, len(full_train))
    g = torch.Generator().manual_seed(train_cfg["seed"])
    perm = torch.randperm(len(full_train), generator=g)[:n]

    val_frac = train_cfg.get("val_fraction", 0.05)
    n_val = max(1, int(n * val_frac))
    val_idx = perm[:n_val].tolist()
    train_idx = perm[n_val:].tolist()

    augment = train_cfg.get("augment", True)
    train_ds = AugmentedEMNIST(Subset(full_train, train_idx), augment=augment, seed=train_cfg["seed"])
    val_ds = AugmentedEMNIST(Subset(full_train, val_idx), augment=False)

    train_loader = DataLoader(train_ds, batch_size=train_cfg["batch_size"], shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=train_cfg["batch_size"], shuffle=False, num_workers=0)
    return train_loader, val_loader


@torch.no_grad()
def _accuracy(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    correct, total = 0, 0
    for images, labels in loader:
        images, labels = images.to(device), labels.clamp(0, len(LABELS) - 1).to(device)
        preds = model(images).argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
    return correct / max(1, total)


def train():
    cfg = load_config()
    train_cfg = cfg["recognition_training"]

    torch.manual_seed(train_cfg["seed"])
    random.seed(train_cfg["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Training CNN recognizer on device: %s", device)

    train_loader, val_loader = get_dataloaders(cfg)
    logger.info("Train batches: %d, Val batches: %d", len(train_loader), len(val_loader))

    model = CharCNN(num_classes=len(LABELS)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=train_cfg["lr"])
    criterion = nn.CrossEntropyLoss()

    ckpt_path = resolve_path(train_cfg["checkpoint_path"])
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)

    best_val_acc = 0.0
    for epoch in range(1, train_cfg["epochs"] + 1):
        t0 = time.time()
        model.train()
        loss_sum, n_batches = 0.0, 0
        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.clamp(0, len(LABELS) - 1).to(device)

            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            loss_sum += loss.item()
            n_batches += 1

        train_loss = loss_sum / max(1, n_batches)
        val_acc = _accuracy(model, val_loader, device)
        dt = time.time() - t0
        logger.info(
            "Epoch %d/%d | train_loss=%.4f | val_acc=%.4f | %.1fs",
            epoch, train_cfg["epochs"], train_loss, val_acc, dt,
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "labels": LABELS,
                    "num_classes": len(LABELS),
                    "epoch": epoch,
                    "val_acc": val_acc,
                },
                ckpt_path,
            )
            logger.info("Saved new best checkpoint to %s (val_acc=%.4f)", ckpt_path, val_acc)

    logger.info("Training complete. Best val_acc=%.4f", best_val_acc)


if __name__ == "__main__":
    train()
