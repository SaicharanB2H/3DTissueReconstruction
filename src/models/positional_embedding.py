"""
Continuous Fourier and Sinusoidal Positional Embeddings for Medical Slice Depth Coordinates.
"""
import math
import torch
import torch.nn as nn


class ContinuousFourierPositionalEmbedding(nn.Module):
    """
    Encodes continuous depth coordinates z in [0, 1] and relative positions alpha in [0, 1]
    into rich harmonic frequency feature vectors.
    """

    def __init__(self, emb_dim: int = 64, num_frequencies: int = 32, max_period: float = 10000.0):
        super().__init__()
        self.emb_dim = emb_dim
        self.num_frequencies = num_frequencies

        half_dim = num_frequencies // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half_dim, dtype=torch.float32) / half_dim
        )
        self.register_buffer("freqs", freqs)

        # Projection MLP: sin and cos concatenation has dimension num_frequencies
        in_dim = num_frequencies
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, emb_dim),
            nn.SiLU(),
            nn.Linear(emb_dim, emb_dim),
        )

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        """
        coords: Tensor of shape (B,) or (B, 1) or (B, N) with values in [0, 1]
        Returns: Tensor of shape (B, emb_dim)
        """
        if coords.ndim == 1:
            coords = coords.unsqueeze(1) # (B, 1)

        # (B, 1) * (1, half_dim) -> (B, half_dim)
        args = coords * self.freqs.unsqueeze(0) * (2.0 * math.pi)
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1) # (B, num_frequencies)

        # If input had extra dimensions, flatten and project
        out = self.mlp(emb)
        return out


class SliceDepthEmbedding(nn.Module):
    """
    Comprehensive multi-coordinate embedding combining:
    1. Global normalized target depth z_target / (D - 1)
    2. Relative position alpha = (z_target - z_prev) / (z_next - z_prev)
    3. Normalized gap distance delta_z / 16 (checkpoint-compatible convention)
    """

    def __init__(self, out_dim: int = 256, base_freq_dim: int = 32):
        super().__init__()
        self.z_embedder = ContinuousFourierPositionalEmbedding(emb_dim=64, num_frequencies=base_freq_dim)
        self.alpha_embedder = ContinuousFourierPositionalEmbedding(emb_dim=64, num_frequencies=base_freq_dim)
        self.gap_embedder = ContinuousFourierPositionalEmbedding(emb_dim=64, num_frequencies=base_freq_dim)

        self.proj = nn.Sequential(
            nn.Linear(64 * 3, out_dim),
            nn.SiLU(),
            nn.Linear(out_dim, out_dim),
        )

    def forward(
        self,
        normalized_z: torch.Tensor,
        alpha: torch.Tensor,
        gap_dist: torch.Tensor,
    ) -> torch.Tensor:
        """
        Returns a rich depth-conditioning vector of shape (B, out_dim).
        """
        emb_z = self.z_embedder(normalized_z)
        emb_alpha = self.alpha_embedder(alpha)
        # Normalize gap_dist by 16.0
        emb_gap = self.gap_embedder(gap_dist / 16.0)

        concat = torch.cat([emb_z, emb_alpha, emb_gap], dim=-1)
        return self.proj(concat)
