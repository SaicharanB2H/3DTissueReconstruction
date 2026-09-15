"""
Conditional Denoising U-Net for Latent Diffusion on Medical Slice Latents (30x30).
Combines Timestep Embeddings, Cross-Attention on Context Tokens, and Spatial Feature Fusion.
"""
import math
from typing import List, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class SinusoidalTimeEmbedding(nn.Module):
    """Sinusoidal timestep embedding for diffusion time t."""

    def __init__(self, emb_dim: int):
        super().__init__()
        self.emb_dim = emb_dim

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        half_dim = self.emb_dim // 2
        exponent = -math.log(10000) * torch.arange(half_dim, dtype=torch.float32, device=timesteps.device) / half_dim
        freqs = torch.exp(exponent)
        args = timesteps.unsqueeze(1).float() * freqs.unsqueeze(0)
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        if self.emb_dim % 2 == 1:
            emb = F.pad(emb, (0, 1))
        return emb


class SpatialCrossAttention(nn.Module):
    """
    Cross-Attention mechanism: Queries from spatial latent features,
    Keys and Values from Context Conditioning Tokens.
    """

    def __init__(self, query_dim: int, context_dim: int, num_heads: int = 4):
        super().__init__()
        self.query_dim = query_dim
        self.context_dim = context_dim
        self.num_heads = num_heads
        self.head_dim = query_dim // num_heads
        self.scale = 1.0 / math.sqrt(self.head_dim)

        self.norm = nn.GroupNorm(min(8, query_dim), query_dim)
        self.to_q = nn.Linear(query_dim, query_dim, bias=False)
        self.to_k = nn.Linear(context_dim, query_dim, bias=False)
        self.to_v = nn.Linear(context_dim, query_dim, bias=False)
        self.to_out = nn.Sequential(
            nn.Linear(query_dim, query_dim),
            nn.Dropout(0.0),
        )

    def forward(self, x: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        """
        x: (B, C, H, W)
        context: (B, N_tokens, context_dim)
        """
        B, C, H, W = x.shape
        h = self.norm(x)
        h_flat = h.permute(0, 2, 3, 1).reshape(B, H * W, C) # (B, HW, C)

        q = self.to_q(h_flat).reshape(B, H * W, self.num_heads, self.head_dim).permute(0, 2, 1, 3) # (B, heads, HW, head_dim)
        k = self.to_k(context).reshape(B, -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)  # (B, heads, N, head_dim)
        v = self.to_v(context).reshape(B, -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)  # (B, heads, N, head_dim)

        # Scaled dot-product attention: (HW, N)
        attn = torch.softmax((q @ k.transpose(-1, -2)) * self.scale, dim=-1)
        out = attn @ v # (B, heads, HW, head_dim)

        out = out.permute(0, 2, 1, 3).reshape(B, H * W, C)
        out = self.to_out(out)
        out = out.reshape(B, H, W, C).permute(0, 3, 1, 2)

        return x + out


class ResnetBlockConditioned(nn.Module):
    """Residual Block conditioned on Timestep Embedding."""

    def __init__(self, in_channels: int, out_channels: int, time_emb_dim: int, num_groups: int = 16):
        super().__init__()
        self.norm1 = nn.GroupNorm(min(num_groups, in_channels), in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)

        self.time_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, out_channels),
        )

        self.norm2 = nn.GroupNorm(min(num_groups, out_channels), out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)

        self.shortcut = (
            nn.Conv2d(in_channels, out_channels, kernel_size=1)
            if in_channels != out_channels
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor, time_emb: torch.Tensor) -> torch.Tensor:
        h = self.conv1(F.silu(self.norm1(x)))
        time_proj = self.time_mlp(time_emb).unsqueeze(-1).unsqueeze(-1)
        h = h + time_proj
        h = self.conv2(F.silu(self.norm2(h)))
        return h + self.shortcut(x)


class ConditionalUNet2D(nn.Module):
    """
    Conditional Denoising U-Net for Slice-Latent Diffusion.
    Takes noisy latent z_t (B, 4, 30, 30), spatial context (B, 8, 30, 30),
    timestep t (B,), and context tokens (B, 17, 256).
    """

    def __init__(
        self,
        in_channels: int = 4,
        spatial_context_channels: int = 8,
        out_channels: int = 4,
        base_channels: int = 64,
        channel_multipliers: List[int] = [1, 2, 4], # 30x30 -> 15x15 -> 8x8
        num_res_blocks: int = 2,
        context_dim: int = 256,
        num_heads: int = 4,
        num_groups: int = 16,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.spatial_context_channels = spatial_context_channels
        self.out_channels = out_channels
        self.num_res_blocks = num_res_blocks

        # Timestep embedding MLP
        time_dim = base_channels * 4
        self.time_proj = nn.Sequential(
            SinusoidalTimeEmbedding(base_channels),
            nn.Linear(base_channels, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim),
        )

        # Input projection: noisy latent + spatial context concatenated along channels
        total_in_c = in_channels + spatial_context_channels
        self.conv_in = nn.Conv2d(total_in_c, base_channels, kernel_size=3, padding=1)

        # Downsampling path
        self.down_blocks = nn.ModuleList()
        self.down_attns = nn.ModuleList()
        self.downsamples = nn.ModuleList()

        channels = base_channels
        skip_channels = [channels]

        for i, mult in enumerate(channel_multipliers):
            out_ch = base_channels * mult
            for _ in range(num_res_blocks):
                self.down_blocks.append(
                    ResnetBlockConditioned(channels, out_ch, time_dim, num_groups=num_groups)
                )
                self.down_attns.append(
                    SpatialCrossAttention(out_ch, context_dim, num_heads=num_heads)
                )
                channels = out_ch
                skip_channels.append(channels)

            if i < len(channel_multipliers) - 1:
                self.downsamples.append(
                    nn.Conv2d(channels, channels, kernel_size=3, stride=2, padding=1)
                )
                skip_channels.append(channels)
            else:
                self.downsamples.append(None)

        # Mid-level blocks
        self.mid_block1 = ResnetBlockConditioned(channels, channels, time_dim, num_groups=num_groups)
        self.mid_attn = SpatialCrossAttention(channels, context_dim, num_heads=num_heads)
        self.mid_block2 = ResnetBlockConditioned(channels, channels, time_dim, num_groups=num_groups)

        # Upsampling path
        self.up_blocks = nn.ModuleList()
        self.up_attns = nn.ModuleList()
        self.upsamples = nn.ModuleList()

        for i, mult in reversed(list(enumerate(channel_multipliers))):
            out_ch = base_channels * mult
            for _ in range(num_res_blocks + 1):
                skip_ch = skip_channels.pop()
                self.up_blocks.append(
                    ResnetBlockConditioned(channels + skip_ch, out_ch, time_dim, num_groups=num_groups)
                )
                self.up_attns.append(
                    SpatialCrossAttention(out_ch, context_dim, num_heads=num_heads)
                )
                channels = out_ch

            if i > 0:
                self.upsamples.append(
                    nn.Sequential(
                        nn.Upsample(scale_factor=2.0, mode="nearest"),
                        nn.Conv2d(channels, channels, kernel_size=3, padding=1),
                    )
                )
            else:
                self.upsamples.append(None)

        self.norm_out = nn.GroupNorm(min(num_groups, channels), channels)
        self.conv_out = nn.Conv2d(channels, out_channels, kernel_size=3, padding=1)

    def forward(
        self,
        z_t: torch.Tensor,
        timesteps: torch.Tensor,
        context_tokens: torch.Tensor,
        spatial_context: torch.Tensor,
    ) -> torch.Tensor:
        """
        Inputs:
        - z_t: (B, in_channels, 30, 30) - noisy latent
        - timesteps: (B,) - integer timesteps in [0, T-1]
        - context_tokens: (B, N_tokens, context_dim)
        - spatial_context: (B, spatial_context_channels, 30, 30)

        Returns:
        - predicted_noise: (B, out_channels, 30, 30)
        """
        time_emb = self.time_proj(timesteps)

        # Concatenate noisy latent and spatial context features
        x = torch.cat([z_t, spatial_context], dim=1) # (B, in_channels + spatial_c, 30, 30)
        h = self.conv_in(x)

        skips = [h]
        down_idx = 0
        for i in range(len(self.downsamples)):
            num_blocks = self.num_res_blocks
            for _ in range(num_blocks):
                h = self.down_blocks[down_idx](h, time_emb)
                h = self.down_attns[down_idx](h, context_tokens)
                skips.append(h)
                down_idx += 1

            if self.downsamples[i] is not None:
                h = self.downsamples[i](h)
                skips.append(h)

        # Mid blocks
        h = self.mid_block1(h, time_emb)
        h = self.mid_attn(h, context_tokens)
        h = self.mid_block2(h, time_emb)

        # Up blocks
        up_idx = 0
        for i in range(len(self.upsamples)):
            num_blocks = self.num_res_blocks + 1
            for _ in range(num_blocks):
                skip = skips.pop()
                # Handle potential spatial dimension mismatch if odd size
                if h.shape[2:] != skip.shape[2:]:
                    h = F.interpolate(h, size=skip.shape[2:], mode="nearest")
                h = torch.cat([h, skip], dim=1)
                h = self.up_blocks[up_idx](h, time_emb)
                h = self.up_attns[up_idx](h, context_tokens)
                up_idx += 1

            if self.upsamples[i] is not None:
                h = self.upsamples[i](h)

        h = F.silu(self.norm_out(h))
        return self.conv_out(h)
