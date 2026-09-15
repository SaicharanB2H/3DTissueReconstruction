"""
Medical Variational Autoencoder (AutoencoderKL) for 2D/3D MRI Latent Space Compression.
Compresses (B, C, 240, 240) into compact (B, latent_channels, 30, 30) latent space.
"""
from typing import Tuple, List, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


class DiagonalGaussianDistribution:
    """
    Diagonal Gaussian distribution with reparameterization trick and analytical KL divergence.
    """

    def __init__(self, parameters: torch.Tensor, deterministic: bool = False):
        self.parameters = parameters
        self.mean, self.logvar = torch.chunk(parameters, 2, dim=1)
        self.logvar = torch.clamp(self.logvar, -30.0, 20.0)
        self.deterministic = deterministic
        self.std = torch.exp(0.5 * self.logvar)
        self.var = torch.exp(self.logvar)

    def sample(self) -> torch.Tensor:
        if self.deterministic:
            return self.mean
        eps = torch.randn_like(self.mean)
        return self.mean + self.std * eps

    def kl(self, other: Optional["DiagonalGaussianDistribution"] = None) -> torch.Tensor:
        if other is None:
            # KL divergence against Standard Normal N(0, I)
            return 0.5 * torch.sum(
                torch.pow(self.mean, 2) + self.var - 1.0 - self.logvar,
                dim=[1, 2, 3],
            )
        return 0.5 * torch.sum(
            torch.pow(self.mean - other.mean, 2) / other.var
            + self.var / other.var
            - 1.0
            - self.logvar
            + other.logvar,
            dim=[1, 2, 3],
        )

    def mode(self) -> torch.Tensor:
        return self.mean


class ResnetBlock2D(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, num_groups: int = 16):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        groups = min(num_groups, in_channels, out_channels)
        self.norm1 = nn.GroupNorm(groups, in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)

        groups_out = min(num_groups, out_channels)
        self.norm2 = nn.GroupNorm(groups_out, out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)

        if in_channels != out_channels:
            self.nin_shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        else:
            self.nin_shortcut = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.conv1(F.silu(self.norm1(x)))
        h = self.conv2(F.silu(self.norm2(h)))
        return h + self.nin_shortcut(x)


class AttentionBlock2D(nn.Module):
    def __init__(self, channels: int, num_heads: int = 4, num_groups: int = 16):
        super().__init__()
        self.channels = channels
        self.num_heads = num_heads
        groups = min(num_groups, channels)
        self.norm = nn.GroupNorm(groups, channels)
        self.qkv = nn.Conv2d(channels, channels * 3, kernel_size=1)
        self.proj_out = nn.Conv2d(channels, channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        h = self.norm(x)
        qkv = self.qkv(h)
        q, k, v = torch.chunk(qkv, 3, dim=1)

        # Reshape for multi-head attention: (B, heads, H*W, dim_head)
        dim_head = C // self.num_heads
        q = q.reshape(B, self.num_heads, dim_head, H * W).transpose(-1, -2)
        k = k.reshape(B, self.num_heads, dim_head, H * W).transpose(-1, -2)
        v = v.reshape(B, self.num_heads, dim_head, H * W).transpose(-1, -2)

        # Scaled dot-product attention
        scale = 1.0 / (dim_head ** 0.5)
        attn = torch.softmax((q @ k.transpose(-1, -2)) * scale, dim=-1)
        out = attn @ v # (B, heads, H*W, dim_head)

        out = out.transpose(-1, -2).reshape(B, C, H, W)
        return x + self.proj_out(out)


class Downsample2D(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class Upsample2D(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, scale_factor=2.0, mode="nearest")
        return self.conv(x)


class MedicalVAEEncoder(nn.Module):
    """
    Downsamples from 240x240 to 30x30 (8x spatial compression).
    """

    def __init__(
        self,
        in_channels: int = 1,
        latent_channels: int = 4,
        base_channels: int = 32,
        channel_multipliers: List[int] = [1, 2, 4, 4],
        num_res_blocks: int = 2,
        num_groups: int = 16,
    ):
        super().__init__()
        self.conv_in = nn.Conv2d(in_channels, base_channels, kernel_size=3, padding=1)

        channels = base_channels
        self.down_blocks = nn.ModuleList()

        for i, mult in enumerate(channel_multipliers):
            out_ch = base_channels * mult
            for _ in range(num_res_blocks):
                self.down_blocks.append(ResnetBlock2D(channels, out_ch, num_groups=num_groups))
                channels = out_ch
            if i < len(channel_multipliers) - 1: # 3 downsamples -> 240 -> 120 -> 60 -> 30
                self.down_blocks.append(Downsample2D(channels))

        # Mid-level blocks with attention
        self.mid_block1 = ResnetBlock2D(channels, channels, num_groups=num_groups)
        self.mid_attn = AttentionBlock2D(channels, num_groups=num_groups)
        self.mid_block2 = ResnetBlock2D(channels, channels, num_groups=num_groups)

        # Output projection for mean and log-var (2 * latent_channels)
        self.norm_out = nn.GroupNorm(min(num_groups, channels), channels)
        self.conv_out = nn.Conv2d(channels, 2 * latent_channels, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.conv_in(x)
        for block in self.down_blocks:
            h = block(h)
        h = self.mid_block1(h)
        h = self.mid_attn(h)
        h = self.mid_block2(h)
        h = F.silu(self.norm_out(h))
        return self.conv_out(h)


class MedicalVAEDecoder(nn.Module):
    """
    Upsamples from 30x30 to 240x240.
    """

    def __init__(
        self,
        out_channels: int = 1,
        latent_channels: int = 4,
        base_channels: int = 32,
        channel_multipliers: List[int] = [4, 4, 2, 1],
        num_res_blocks: int = 2,
        num_groups: int = 16,
    ):
        super().__init__()
        in_ch = base_channels * channel_multipliers[0]
        self.conv_in = nn.Conv2d(latent_channels, in_ch, kernel_size=3, padding=1)

        # Mid-level blocks
        self.mid_block1 = ResnetBlock2D(in_ch, in_ch, num_groups=num_groups)
        self.mid_attn = AttentionBlock2D(in_ch, num_groups=num_groups)
        self.mid_block2 = ResnetBlock2D(in_ch, in_ch, num_groups=num_groups)

        channels = in_ch
        self.up_blocks = nn.ModuleList()

        for i, mult in enumerate(channel_multipliers):
            out_ch = base_channels * mult
            for _ in range(num_res_blocks + 1):
                self.up_blocks.append(ResnetBlock2D(channels, out_ch, num_groups=num_groups))
                channels = out_ch
            if i < len(channel_multipliers) - 1: # 3 upsamples -> 30 -> 60 -> 120 -> 240
                self.up_blocks.append(Upsample2D(channels))

        self.norm_out = nn.GroupNorm(min(num_groups, channels), channels)
        self.conv_out = nn.Conv2d(channels, out_channels, kernel_size=3, padding=1)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        h = self.conv_in(z)
        h = self.mid_block1(h)
        h = self.mid_attn(h)
        h = self.mid_block2(h)
        for block in self.up_blocks:
            h = block(h)
        h = F.silu(self.norm_out(h))
        # Use tanh to ensure output is strictly within [-1, 1]
        return torch.tanh(self.conv_out(h))


class MedicalAutoencoderKL(nn.Module):
    """
    Full Variational Autoencoder (AutoencoderKL).
    """

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        latent_channels: int = 4,
        base_channels: int = 32,
        channel_multipliers: List[int] = [1, 2, 4, 4],
        num_res_blocks: int = 2,
        scale_factor: float = 0.18215,
    ):
        super().__init__()
        self.scale_factor = scale_factor
        self.encoder = MedicalVAEEncoder(
            in_channels=in_channels,
            latent_channels=latent_channels,
            base_channels=base_channels,
            channel_multipliers=channel_multipliers,
            num_res_blocks=num_res_blocks,
        )
        self.decoder = MedicalVAEDecoder(
            out_channels=out_channels,
            latent_channels=latent_channels,
            base_channels=base_channels,
            channel_multipliers=list(reversed(channel_multipliers)),
            num_res_blocks=num_res_blocks,
        )

    def encode(self, x: torch.Tensor) -> DiagonalGaussianDistribution:
        h = self.encoder(x)
        return DiagonalGaussianDistribution(h)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        # Scale back from latent distribution space to unscaled latent
        z_unscaled = z / self.scale_factor
        return self.decoder(z_unscaled)

    def forward(
        self, x: torch.Tensor, sample_posterior: bool = True
    ) -> Tuple[torch.Tensor, DiagonalGaussianDistribution]:
        posterior = self.encode(x)
        if sample_posterior:
            z = posterior.sample()
        else:
            z = posterior.mode()
        # Scale latent for diffusion stability
        z_scaled = z * self.scale_factor
        recon = self.decode(z_scaled)
        return recon, posterior
