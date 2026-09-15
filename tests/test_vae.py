"""
Unit tests for Medical AutoencoderKL (VAE).
"""
import pytest
import torch

from src.models.vae import MedicalAutoencoderKL, DiagonalGaussianDistribution


def test_diagonal_gaussian():
    B, C, H, W = 2, 8, 30, 30
    params = torch.randn(B, C, H, W)
    dist = DiagonalGaussianDistribution(params)

    sample = dist.sample()
    assert sample.shape == (B, 4, H, W)

    mode = dist.mode()
    assert mode.shape == (B, 4, H, W)

    kl = dist.kl()
    assert kl.shape == (B,)
    assert (kl >= 0).all()


def test_vae_forward_pass():
    vae = MedicalAutoencoderKL(
        in_channels=1,
        out_channels=1,
        latent_channels=4,
        base_channels=16,
        channel_multipliers=[1, 2, 4, 4],
        num_res_blocks=1,
    )

    B = 2
    x = torch.randn(B, 1, 240, 240)
    recon, posterior = vae(x, sample_posterior=True)

    assert recon.shape == (B, 1, 240, 240)
    assert posterior.sample().shape == (B, 4, 30, 30)

    # Verify loss and gradient flow
    l1_loss = torch.abs(recon - x).mean()
    kl_loss = posterior.kl().mean()
    total_loss = l1_loss + 1e-4 * kl_loss

    total_loss.backward()

    for p in vae.parameters():
        if p.requires_grad:
            assert p.grad is not None


def test_vae_encode_decode():
    vae = MedicalAutoencoderKL(
        in_channels=1,
        out_channels=1,
        latent_channels=4,
        base_channels=16,
        channel_multipliers=[1, 2, 4, 4],
        num_res_blocks=1,
    )

    x = torch.randn(1, 1, 240, 240)
    with torch.no_grad():
        dist = vae.encode(x)
        z = dist.sample() * vae.scale_factor
        assert z.shape == (1, 4, 30, 30)

        recon = vae.decode(z)
        assert recon.shape == (1, 1, 240, 240)
        assert recon.min() >= -1.0
        assert recon.max() <= 1.0
