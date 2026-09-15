"""
Unit tests for Continuous Positional Embeddings and Shared Slice Encoder.
"""
import pytest
import torch

from src.models.positional_embedding import ContinuousFourierPositionalEmbedding, SliceDepthEmbedding
from src.models.slice_encoder import SharedSliceEncoder


def test_positional_embeddings():
    fourier_emb = ContinuousFourierPositionalEmbedding(emb_dim=64, num_frequencies=32)
    coords = torch.tensor([0.0, 0.25, 0.5, 0.75, 1.0])
    emb = fourier_emb(coords)

    assert emb.shape == (5, 64)
    # Output for distinct coordinates should be distinct
    assert not torch.allclose(emb[0], emb[1])
    assert not torch.isnan(emb).any()


def test_slice_depth_embedding():
    depth_embedder = SliceDepthEmbedding(out_dim=256)
    B = 4
    z = torch.linspace(0, 1, B)
    alpha = torch.full((B,), 0.5)
    gap = torch.full((B,), 4.0)

    out = depth_embedder(z, alpha, gap)
    assert out.shape == (B, 256)
    assert not torch.isnan(out).any()


def test_shared_slice_encoder():
    encoder = SharedSliceEncoder(
        in_channels=1,
        base_channels=16,
        context_dim=128,
        out_spatial_channels=8,
    )

    B = 2
    prev_slice = torch.randn(B, 1, 240, 240)
    next_slice = torch.randn(B, 1, 240, 240)
    norm_z = torch.tensor([0.2, 0.8])
    alpha = torch.tensor([0.25, 0.75])
    gap_dist = torch.tensor([4.0, 8.0])

    context_tokens, spatial_context = encoder(prev_slice, next_slice, norm_z, alpha, gap_dist)

    # 1 global token + 16 spatial tokens = 17 tokens
    assert context_tokens.shape == (B, 17, 128)
    assert spatial_context.shape == (B, 8, 30, 30)
    assert not torch.isnan(context_tokens).any()
    assert not torch.isnan(spatial_context).any()

    # Verify gradients flow
    loss = context_tokens.sum() + spatial_context.sum()
    loss.backward()

    for p in encoder.parameters():
        if p.requires_grad:
            assert p.grad is not None
