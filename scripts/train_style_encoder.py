"""Train the conditional-VAE style encoder / generator on EMNIST letters.

TRAINING regime (section 11 of the design brief):
    many writers -> learn a general handwriting style representation
    (EMNIST's "letters" split is assembled from tens of thousands of NIST
    Special Database 19 writers, so a single image's label carries no
    writer ID, but across the ~145k images the model still sees enormous
    style diversity per character class — enough for the style latent to
    learn general, transferable visual-style factors.)

INFERENCE (see src/style_encoder/inference.py) then only needs a handful
of glyphs from a brand-new writer to estimate their style embedding —
no retraining.

Usage:
    python scripts/train_style_encoder.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from torch.utils.data import DataLoader, Subset

from src.style_encoder.model import HandwritingStyleVAE, vae_loss
from src.utils.config import load_config, resolve_path
from src.utils.emnist_source import load_emnist_letters
from src.utils.logging_setup import get_logger

logger = get_logger(__name__)


def get_dataloaders(cfg: dict):
    train_cfg = cfg["training"]
    cache_dir = resolve_path("data/raw") / "hf_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    # Orientation fix and label indexing (0..25 -> a..z) are handled inside
    # EMNISTLettersHF — see src/utils/emnist_source.py for why we source
    # from Hugging Face rather than torchvision's built-in EMNIST downloader.
    full_train = load_emnist_letters(split="train", cache_dir=str(cache_dir))

    max_n = train_cfg.get("max_train_samples", 40000)
    n = min(max_n, len(full_train))
    g = torch.Generator().manual_seed(train_cfg["seed"])
    perm = torch.randperm(len(full_train), generator=g)[:n]

    val_frac = train_cfg.get("val_fraction", 0.05)
    n_val = max(1, int(n * val_frac))
    val_idx = perm[:n_val].tolist()
    train_idx = perm[n_val:].tolist()

    train_ds = Subset(full_train, train_idx)
    val_ds = Subset(full_train, val_idx)

    train_loader = DataLoader(train_ds, batch_size=train_cfg["batch_size"], shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=train_cfg["batch_size"], shuffle=False, num_workers=0)
    return train_loader, val_loader


def train():
    cfg = load_config()
    train_cfg = cfg["training"]
    se_cfg = cfg["style_encoder"]

    torch.manual_seed(train_cfg["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Training on device: %s", device)

    train_loader, val_loader = get_dataloaders(cfg)
    logger.info("Train batches: %d, Val batches: %d", len(train_loader), len(val_loader))

    model = HandwritingStyleVAE(
        num_classes=se_cfg["num_classes"],
        latent_dim=se_cfg["latent_dim"],
        content_embed_dim=se_cfg["content_embed_dim"],
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=train_cfg["lr"])

    ckpt_path = resolve_path(train_cfg["checkpoint_path"])
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)

    best_val = float("inf")
    for epoch in range(1, train_cfg["epochs"] + 1):
        t0 = time.time()
        model.train()
        train_loss_sum, train_recon_sum, train_kl_sum, n_batches = 0.0, 0.0, 0.0, 0
        for images, labels in train_loader:
            images = images.to(device)
            class_idx = labels.clamp(0, se_cfg["num_classes"] - 1).to(device)

            optimizer.zero_grad()
            recon, mu, logvar = model(images, class_idx)
            loss, recon_loss, kl = vae_loss(recon, images, mu, logvar)
            loss.backward()
            optimizer.step()

            train_loss_sum += loss.item()
            train_recon_sum += recon_loss.item()
            train_kl_sum += kl.item()
            n_batches += 1

        model.eval()
        val_loss_sum, val_batches = 0.0, 0
        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device)
                class_idx = labels.clamp(0, se_cfg["num_classes"] - 1).to(device)
                recon, mu, logvar = model(images, class_idx)
                loss, _, _ = vae_loss(recon, images, mu, logvar)
                val_loss_sum += loss.item()
                val_batches += 1

        train_loss = train_loss_sum / max(1, n_batches)
        val_loss = val_loss_sum / max(1, val_batches)
        dt = time.time() - t0
        logger.info(
            "Epoch %d/%d | train_loss=%.2f (recon=%.2f kl=%.2f) | val_loss=%.2f | %.1fs",
            epoch, train_cfg["epochs"], train_loss,
            train_recon_sum / max(1, n_batches), train_kl_sum / max(1, n_batches),
            val_loss, dt,
        )

        if val_loss < best_val:
            best_val = val_loss
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "config": se_cfg,
                    "epoch": epoch,
                    "val_loss": val_loss,
                },
                ckpt_path,
            )
            logger.info("Saved new best checkpoint to %s (val_loss=%.2f)", ckpt_path, val_loss)

    logger.info("Training complete. Best val_loss=%.2f", best_val)


if __name__ == "__main__":
    train()
