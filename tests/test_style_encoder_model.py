"""Unit tests for the conditional VAE architecture (src/style_encoder/model.py).

These test the model's shapes and basic training-step mechanics (not
learned quality, which is evaluated separately after training — see
scripts/train_style_encoder.py and docs/experiments.md).
"""
from __future__ import annotations

import torch

from src.style_encoder.model import Decoder, HandwritingStyleVAE, StyleEncoder, vae_loss


def test_style_encoder_output_shapes():
    encoder = StyleEncoder(latent_dim=32)
    x = torch.rand(4, 1, 28, 28)
    mu, logvar = encoder(x)
    assert mu.shape == (4, 32)
    assert logvar.shape == (4, 32)


def test_decoder_output_shape():
    decoder = Decoder(latent_dim=32, content_embed_dim=16)
    z = torch.randn(4, 32)
    content = torch.randn(4, 16)
    out = decoder(z, content)
    assert out.shape == (4, 1, 28, 28)
    assert torch.all((out >= 0) & (out <= 1))  # sigmoid output


def test_full_vae_forward_pass():
    model = HandwritingStyleVAE(num_classes=26, latent_dim=32, content_embed_dim=16)
    x = torch.rand(8, 1, 28, 28)
    class_idx = torch.randint(0, 26, (8,))
    recon, mu, logvar = model(x, class_idx)
    assert recon.shape == x.shape
    assert mu.shape == (8, 32)
    assert logvar.shape == (8, 32)


def test_vae_loss_is_finite_and_positive():
    model = HandwritingStyleVAE()
    x = torch.rand(4, 1, 28, 28)
    class_idx = torch.randint(0, 26, (4,))
    recon, mu, logvar = model(x, class_idx)
    total, recon_loss, kl = vae_loss(recon, x, mu, logvar)
    assert torch.isfinite(total)
    assert total.item() > 0


def test_encode_style_is_deterministic():
    model = HandwritingStyleVAE()
    model.eval()
    x = torch.rand(2, 1, 28, 28)
    z1 = model.encode_style(x)
    z2 = model.encode_style(x)
    assert torch.allclose(z1, z2)


def test_generate_produces_valid_image_shape():
    model = HandwritingStyleVAE()
    model.eval()
    z = torch.randn(1, model.latent_dim)
    class_idx = torch.tensor([5])
    out = model.generate(z, class_idx)
    assert out.shape == (1, 1, 28, 28)


def test_gradient_flows_through_full_model():
    model = HandwritingStyleVAE()
    x = torch.rand(2, 1, 28, 28)
    class_idx = torch.tensor([0, 1])
    recon, mu, logvar = model(x, class_idx)
    loss, _, _ = vae_loss(recon, x, mu, logvar)
    loss.backward()
    grad_norm = sum(p.grad.abs().sum().item() for p in model.parameters() if p.grad is not None)
    assert grad_norm > 0
