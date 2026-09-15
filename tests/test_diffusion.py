"""
Unit tests for Diffusion Noise Scheduler, Conditional Denoising U-Net, and Reconstruction Pipeline.
"""
import pytest
import torch

from src.models.positional_embedding import SliceDepthEmbedding
from src.models.slice_encoder import SharedSliceEncoder
from src.models.vae import MedicalAutoencoderKL
from src.models.unet_diffusion import ConditionalUNet2D
from src.models.scheduler import DiffusionNoiseScheduler
from src.models.pipeline import SparseToDenseDiffusionPipeline


def test_noise_scheduler():
    scheduler = DiffusionNoiseScheduler(num_train_timesteps=1000, beta_schedule="cosine")
    x_0 = torch.randn(4, 4, 30, 30)
    noise = torch.randn_like(x_0)
    t = torch.tensor([0, 250, 500, 999])

    x_t = scheduler.add_noise(x_0, noise, t)
    assert x_t.shape == x_0.shape

    # At t=0, x_t should be almost identical to x_0
    assert torch.isclose(x_t[0], x_0[0], atol=0.1).all()

    # DDIM timesteps
    ddim_ts = scheduler.get_ddim_timesteps(num_inference_steps=10)
    assert len(ddim_ts) == 10
    assert ddim_ts[0] > ddim_ts[-1]


def test_conditional_unet_forward():
    unet = ConditionalUNet2D(
        in_channels=4,
        spatial_context_channels=8,
        out_channels=4,
        base_channels=32,
        channel_multipliers=[1, 2, 4],
        num_res_blocks=1,
        context_dim=128,
        num_heads=2,
    )

    B = 2
    z_t = torch.randn(B, 4, 30, 30)
    t = torch.randint(0, 1000, (B,))
    context_tokens = torch.randn(B, 17, 128)
    spatial_context = torch.randn(B, 8, 30, 30)

    noise_pred = unet(z_t, t, context_tokens, spatial_context)
    assert noise_pred.shape == (B, 4, 30, 30)
    assert not torch.isnan(noise_pred).any()

    # Test backward pass
    loss = noise_pred.sum()
    loss.backward()

    for p in unet.parameters():
        if p.requires_grad:
            assert p.grad is not None


def test_end_to_end_pipeline():
    encoder = SharedSliceEncoder(
        in_channels=1,
        base_channels=16,
        context_dim=128,
        out_spatial_channels=8,
    )
    vae = MedicalAutoencoderKL(
        in_channels=1,
        out_channels=1,
        latent_channels=4,
        base_channels=16,
        channel_multipliers=[1, 2, 4, 4],
        num_res_blocks=1,
    )
    unet = ConditionalUNet2D(
        in_channels=4,
        spatial_context_channels=8,
        out_channels=4,
        base_channels=32,
        channel_multipliers=[1, 2, 4],
        num_res_blocks=1,
        context_dim=128,
        num_heads=2,
    )
    scheduler = DiffusionNoiseScheduler(num_train_timesteps=1000, beta_schedule="cosine")

    pipeline = SparseToDenseDiffusionPipeline(encoder, vae, unet, scheduler)

    # Test slice synthesis
    B = 2
    prev_s = torch.randn(B, 1, 240, 240)
    next_s = torch.randn(B, 1, 240, 240)
    norm_z = torch.tensor([0.25, 0.75])
    alpha = torch.tensor([0.5, 0.5])
    gap = torch.tensor([4.0, 4.0])

    gen = pipeline.generate_slices_batch(
        prev_s, next_s, norm_z, alpha, gap, num_inference_steps=5
    )
    assert gen.shape == (B, 1, 240, 240)
    assert not torch.isnan(gen).any()
