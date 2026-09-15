"""
Shared Contextual Slice Encoder with Weight Tying and Positional Embedding Fusion.
"""
from typing import Tuple, Dict, Any, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.positional_embedding import SliceDepthEmbedding


class ConvBlock(nn.Module):
    def __init__(self, in_c: int, out_c: int, stride: int = 1):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.GroupNorm(num_groups=min(8, out_c), num_channels=out_c),
            nn.SiLU(inplace=True),
            nn.Conv2d(out_c, out_c, kernel_size=3, stride=1, padding=1, bias=False),
            nn.GroupNorm(num_groups=min(8, out_c), num_channels=out_c),
            nn.SiLU(inplace=True),
        )
        self.shortcut = (
            nn.Sequential(
                nn.Conv2d(in_c, out_c, kernel_size=1, stride=stride, bias=False),
                nn.GroupNorm(num_groups=min(8, out_c), num_channels=out_c),
            )
            if (in_c != out_c or stride != 1)
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x) + self.shortcut(x)


class SharedSliceEncoder(nn.Module):
    """
    Weight-tied Slice Encoder that processes upper and lower context slices (I_prev, I_next),
    fuses them with continuous depth positional embeddings, and outputs:
    1. context_tokens: (B, N_tokens, context_dim) for Cross-Attention in Denoising U-Net.
    2. spatial_context: (B, out_spatial_c, 30, 30) for channel-wise concatenation with VAE latents.
    """

    def __init__(
        self,
        in_channels: int = 1,
        base_channels: int = 32,
        context_dim: int = 256,
        num_context_tokens: int = 16,
        out_spatial_channels: int = 8,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.context_dim = context_dim
        self.num_context_tokens = num_context_tokens

        # Depth Positional Embedding Module
        self.depth_embedder = SliceDepthEmbedding(out_dim=context_dim)

        # Shared Feature Extractor (Downsamples 240x240 -> 120x120 -> 60x60 -> 30x30)
        self.shared_backbone = nn.Sequential(
            ConvBlock(in_channels, base_channels, stride=2),       # 240 -> 120
            ConvBlock(base_channels, base_channels * 2, stride=2), # 120 -> 60
            ConvBlock(base_channels * 2, base_channels * 4, stride=2), # 60 -> 30
            ConvBlock(base_channels * 4, base_channels * 4, stride=1), # 30 -> 30
        )

        backbone_out_c = base_channels * 4 # e.g. 128

        # Spatial fusion of prev and next features + alpha weighting
        self.spatial_fusion = nn.Sequential(
            nn.Conv2d(backbone_out_c * 2 + 1, backbone_out_c, kernel_size=3, padding=1),
            nn.GroupNorm(num_groups=8, num_channels=backbone_out_c),
            nn.SiLU(inplace=True),
            nn.Conv2d(backbone_out_c, out_spatial_channels, kernel_size=3, padding=1),
        )

        # Token projection for Cross-Attention
        # Adaptive pooling to (4, 4) -> 16 tokens
        self.token_pool = nn.AdaptiveAvgPool2d((4, 4))
        self.token_proj = nn.Linear(backbone_out_c, context_dim)

        # Global conditioning token projection
        self.global_token_proj = nn.Linear(context_dim, context_dim)

    def forward(
        self,
        prev_slice: torch.Tensor,
        next_slice: torch.Tensor,
        normalized_z: torch.Tensor,
        alpha: torch.Tensor,
        gap_dist: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Inputs:
        - prev_slice: (B, C, 240, 240)
        - next_slice: (B, C, 240, 240)
        - normalized_z: (B,) in [0, 1]
        - alpha: (B,) in [0, 1]
        - gap_dist: (B,)

        Outputs:
        - context_tokens: (B, num_tokens + 1, context_dim) for Cross-Attention
        - spatial_context: (B, out_spatial_c, 30, 30) for Latent Concatenation
        """
        B = prev_slice.shape[0]

        # 1. Extract shared features for prev and next slices (weight-tied)
        feat_prev = self.shared_backbone(prev_slice) # (B, C_feat, 30, 30)
        feat_next = self.shared_backbone(next_slice) # (B, C_feat, 30, 30)

        # 2. Linear blend weighted by alpha as guide
        alpha_map = alpha.view(B, 1, 1, 1).expand(B, 1, 30, 30)
        spatial_input = torch.cat([feat_prev, feat_next, alpha_map], dim=1) # (B, C_feat*2 + 1, 30, 30)
        spatial_context = self.spatial_fusion(spatial_input) # (B, out_spatial_c, 30, 30)

        # 3. Compute continuous depth positional embedding
        depth_emb = self.depth_embedder(normalized_z, alpha, gap_dist) # (B, context_dim)
        global_token = self.global_token_proj(depth_emb).unsqueeze(1) # (B, 1, context_dim)

        # 4. Tokenize spatial features
        # Alpha-interpolated feature map for tokens
        feat_blend = (1.0 - alpha_map) * feat_prev + alpha_map * feat_next
        pooled = self.token_pool(feat_blend) # (B, C_feat, 4, 4)
        tokens = pooled.flatten(2).transpose(1, 2) # (B, 16, C_feat)
        tokens = self.token_proj(tokens) # (B, 16, context_dim)

        # Add positional embedding to tokens
        tokens = tokens + depth_emb.unsqueeze(1)

        # Prepend global depth token -> (B, 17, context_dim)
        context_tokens = torch.cat([global_token, tokens], dim=1)

        return context_tokens, spatial_context
