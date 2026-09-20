"""Few-shot inference with the trained HandwritingStyleVAE.

This is the module that realizes section 10 of the design brief
("few-shot learning" / "support increasing the amount of reference
handwriting without changing the fundamental pipeline"):

    new writer -> a few handwritten glyphs -> average their style
    latents into one writer embedding -> decode any of the 26 letters
    conditioned on [writer_embedding, letter_content_embedding]

Adding more reference words simply adds more glyphs to average over;
nothing else in the pipeline changes.
"""
from __future__ import annotations

import string
from pathlib import Path

import cv2
import numpy as np
import torch

from src.style_encoder.model import HandwritingStyleVAE
from src.utils.config import load_config, resolve_path
from src.utils.logging_setup import get_logger

logger = get_logger(__name__)

CHARSET = string.ascii_lowercase
CHAR_TO_IDX = {c: i for i, c in enumerate(CHARSET)}


def glyph_to_tensor(glyph_bin: np.ndarray, image_size: int = 28) -> torch.Tensor:
    """Resize/pad a binary glyph crop (ink=255) into a normalized 1x1xHxW tensor.

    Matches the EMNIST convention the model was trained on: ink=1.0 (white)
    on a 0.0 (black) background, centered in a square canvas before resize.
    """
    h, w = glyph_bin.shape[:2]
    if h == 0 or w == 0:
        square = np.zeros((image_size, image_size), dtype=np.uint8)
    else:
        side = max(h, w)
        square = np.zeros((side, side), dtype=np.uint8)
        y0, x0 = (side - h) // 2, (side - w) // 2
        square[y0:y0 + h, x0:x0 + w] = glyph_bin
        pad = side // 5
        square = cv2.copyMakeBorder(square, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=0)
        square = cv2.resize(square, (image_size, image_size), interpolation=cv2.INTER_AREA)

    tensor = torch.from_numpy(square.astype(np.float32) / 255.0)
    return tensor.unsqueeze(0).unsqueeze(0)  # 1,1,H,W


def tensor_to_glyph(tensor: torch.Tensor, threshold: float | None = None) -> np.ndarray:
    """Convert a model output tensor (1,1,H,W) back into a uint8 ink=255 image."""
    arr = tensor.squeeze().detach().cpu().numpy()
    arr = np.clip(arr, 0, 1)
    if threshold is not None:
        arr = (arr > threshold).astype(np.float32)
    return (arr * 255).astype(np.uint8)


class StyleEncoderInference:
    """Loads a trained checkpoint and exposes few-shot style extraction + generation."""

    def __init__(self, checkpoint_path: str | Path | None = None, config: dict | None = None):
        cfg = config or load_config()
        se_cfg = cfg["style_encoder"]
        train_cfg = cfg["training"]
        ckpt_path = Path(checkpoint_path) if checkpoint_path else resolve_path(train_cfg["checkpoint_path"])

        self.device = torch.device("cpu")  # inference is cheap; keep it CPU-portable
        self.model = HandwritingStyleVAE(
            num_classes=se_cfg["num_classes"],
            latent_dim=se_cfg["latent_dim"],
            content_embed_dim=se_cfg["content_embed_dim"],
        ).to(self.device)

        if not ckpt_path.exists():
            raise FileNotFoundError(
                f"No trained checkpoint at {ckpt_path}. Run `python scripts/train_style_encoder.py` first."
            )
        checkpoint = torch.load(ckpt_path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()
        self.image_size = se_cfg["image_size"]
        logger.info(
            "Loaded style encoder checkpoint (epoch %s, val_loss %.2f) from %s",
            checkpoint.get("epoch"), checkpoint.get("val_loss", float("nan")), ckpt_path,
        )

    def encode_writer_style(self, observed_glyphs: dict[str, np.ndarray]) -> torch.Tensor:
        """Average the style latents of all observed (char -> glyph image) pairs into one embedding."""
        valid = {c: g for c, g in observed_glyphs.items() if c in CHAR_TO_IDX}
        if not valid:
            raise ValueError("No observed glyphs with recognizable lowercase a-z labels were provided.")

        latents = []
        with torch.no_grad():
            for char, glyph in valid.items():
                tensor = glyph_to_tensor(glyph, self.image_size).to(self.device)
                mu = self.model.encode_style(tensor)
                latents.append(mu)
        stacked = torch.cat(latents, dim=0)
        writer_style = stacked.mean(dim=0, keepdim=True)
        logger.info("Estimated writer style from %d observed glyph(s): %s", len(valid), list(valid.keys()))
        return writer_style

    def generate_char(self, char: str, writer_style: torch.Tensor, sample_noise: float = 0.0) -> np.ndarray:
        """Decode one character conditioned on a writer style embedding.

        `sample_noise` > 0 adds small Gaussian jitter to the latent so repeated
        calls produce slightly different (but style-consistent) variants —
        used for natural, non-pixel-identical repeated characters (section 14).
        """
        if char not in CHAR_TO_IDX:
            raise ValueError(f"Character {char!r} is outside the supported charset (a-z).")
        idx = torch.tensor([CHAR_TO_IDX[char]], dtype=torch.long)
        z = writer_style
        if sample_noise > 0:
            z = z + torch.randn_like(z) * sample_noise
        with torch.no_grad():
            recon = self.model.generate(z, idx)
        return tensor_to_glyph(recon, threshold=0.35)

    def generate_alphabet(
        self, writer_style: torch.Tensor, charset: str = CHARSET, variants: int = 1, sample_noise: float = 0.05
    ) -> dict[str, list[np.ndarray]]:
        alphabet = {}
        for char in charset:
            imgs = [
                self.generate_char(char, writer_style, sample_noise=sample_noise if v > 0 else 0.0)
                for v in range(variants)
            ]
            alphabet[char] = imgs
        return alphabet
