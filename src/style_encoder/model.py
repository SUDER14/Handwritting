"""Conditional VAE: the genuine ML component (Stage 2-4 of the design brief).

Architecture (chosen per section 24 — simplest architecture that
demonstrates the research idea; see docs/architecture.md for the full
rationale):

    IMAGE = CHARACTER CONTENT + WRITER STYLE

- `StyleEncoder` (CNN): image -> (mu, logvar) of a Gaussian style latent.
  Trained WITHOUT seeing the character label, so it is pushed to encode
  only the label-independent residual — visual style (slant, stroke
  width, roundness, proportions) — rather than "which letter is this".
- `CharacterEmbedding`: a small learned embedding table, one vector per
  class (a-z). This is the CONTENT representation.
- `Decoder`: takes [style_latent ; character_embedding] and reconstructs
  the glyph image. Because content is supplied explicitly and separately
  from the style latent, the model is architecturally forced to keep the
  two disentangled — the decoder cannot recover "which letter" from the
  style latent alone.

Why a conditional VAE and not a GAN/diffusion model (section 24/25):
a VAE trains stably in a handful of epochs on CPU, has a smooth,
averageable latent space (critical for the few-shot step: averaging
several observed glyphs' style latents into one writer embedding is
exactly VAE-latent arithmetic), and needs no adversarial-training
tuning. A GAN or diffusion model would very likely need a GPU and much
longer training to converge on this 8GB-RAM / CPU-only machine (see
docs/architecture.md, section "Hardware-driven choice").
"""
from __future__ import annotations

import torch
from torch import nn


class StyleEncoder(nn.Module):
    """CNN encoder: 1x28x28 glyph -> (mu, logvar) of the style latent.

    Deliberately does NOT take the character class as input — this is what
    forces the latent to capture writer style rather than letter identity.
    """

    def __init__(self, latent_dim: int = 32):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, 3, stride=2, padding=1),   # 28 -> 14
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),  # 14 -> 7
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),  # 7 -> 4
            nn.ReLU(inplace=True),
        )
        flat_dim = 128 * 4 * 4
        self.fc_mu = nn.Linear(flat_dim, latent_dim)
        self.fc_logvar = nn.Linear(flat_dim, latent_dim)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.conv(x)
        h = h.flatten(1)
        return self.fc_mu(h), self.fc_logvar(h)


class Decoder(nn.Module):
    """Transposed-conv decoder: [style_latent ; char_embedding] -> 1x28x28 glyph."""

    def __init__(self, latent_dim: int = 32, content_embed_dim: int = 16):
        super().__init__()
        in_dim = latent_dim + content_embed_dim
        self.fc = nn.Linear(in_dim, 128 * 4 * 4)
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 3, stride=2, padding=1, output_padding=0),  # 4 -> 7
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1),   # 7 -> 14
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, 1, 3, stride=2, padding=1, output_padding=1),    # 14 -> 28
        )

    def forward(self, z: torch.Tensor, content: torch.Tensor) -> torch.Tensor:
        h = torch.cat([z, content], dim=1)
        h = self.fc(h)
        h = h.view(-1, 128, 4, 4)
        logits = self.deconv(h)
        return torch.sigmoid(logits)


class HandwritingStyleVAE(nn.Module):
    """Full conditional VAE tying together the style encoder, content embedding, and decoder."""

    def __init__(self, num_classes: int = 26, latent_dim: int = 32, content_embed_dim: int = 16):
        super().__init__()
        self.latent_dim = latent_dim
        self.encoder = StyleEncoder(latent_dim)
        self.content_embedding = nn.Embedding(num_classes, content_embed_dim)
        self.decoder = Decoder(latent_dim, content_embed_dim)

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x: torch.Tensor, class_idx: torch.Tensor):
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        content = self.content_embedding(class_idx)
        recon = self.decoder(z, content)
        return recon, mu, logvar

    @torch.no_grad()
    def encode_style(self, x: torch.Tensor) -> torch.Tensor:
        """Deterministic style latent (posterior mean) for inference / few-shot averaging."""
        mu, _ = self.encoder(x)
        return mu

    @torch.no_grad()
    def generate(self, style_latent: torch.Tensor, class_idx: torch.Tensor) -> torch.Tensor:
        content = self.content_embedding(class_idx)
        return self.decoder(style_latent, content)


def vae_loss(
    recon: torch.Tensor, target: torch.Tensor, mu: torch.Tensor, logvar: torch.Tensor, kl_weight: float = 0.5
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Binary-cross-entropy reconstruction + KL divergence to a standard normal prior."""
    recon_loss = nn.functional.binary_cross_entropy(recon, target, reduction="sum") / target.size(0)
    kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / target.size(0)
    total = recon_loss + kl_weight * kl
    return total, recon_loss, kl
