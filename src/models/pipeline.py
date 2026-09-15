"""
End-to-End Sparse-to-Dense 3D Medical Volume Reconstruction Pipeline.
Coordinates Context Slice Encoding, Latent Diffusion Sampling, VAE Decoding, and 3D Assembly.
"""
from typing import List, Dict, Any, Optional, Tuple, Union
import numpy as np
import torch
import torch.nn as nn
import scipy.ndimage
from tqdm import tqdm

from src.models.slice_encoder import SharedSliceEncoder
from src.models.vae import MedicalAutoencoderKL
from src.models.unet_diffusion import ConditionalUNet2D
from src.models.scheduler import DiffusionNoiseScheduler
from src.data.sparsifier import VolumeSparsifier


class SparseToDenseDiffusionPipeline(nn.Module):
    """
    Complete generative 3D medical volume reconstructor.
    """

    def __init__(
        self,
        slice_encoder: SharedSliceEncoder,
        vae: MedicalAutoencoderKL,
        unet: ConditionalUNet2D,
        scheduler: DiffusionNoiseScheduler,
    ):
        super().__init__()
        self.slice_encoder = slice_encoder
        self.vae = vae
        self.unet = unet
        self.scheduler = scheduler

    @torch.no_grad()
    def generate_slices_batch(
        self,
        prev_slices: torch.Tensor,
        next_slices: torch.Tensor,
        normalized_zs: torch.Tensor,
        alphas: torch.Tensor,
        gap_dists: torch.Tensor,
        num_inference_steps: int = 25,
        eta: float = 0.0,
    ) -> torch.Tensor:
        """
        Synthesizes a batch of missing 2D slices given their bounding context.
        """
        B = prev_slices.shape[0]
        device = prev_slices.device

        # 1. Extract context tokens and spatial conditioning from slice encoder
        context_tokens, spatial_context = self.slice_encoder(
            prev_slices, next_slices, normalized_zs, alphas, gap_dists
        )

        # 2. Initialize latent noise z_T ~ N(0, I) of shape (B, 4, 30, 30)
        latent_shape = (B, self.unet.in_channels, 30, 30)
        z_t = torch.randn(latent_shape, device=device)

        # 3. Fast DDIM Reverse Diffusion Loop
        timesteps = self.scheduler.get_ddim_timesteps(num_inference_steps)

        for i, t in enumerate(timesteps):
            t_prev = timesteps[i + 1] if i + 1 < len(timesteps) else -1
            t_tensor = torch.full((B,), t, device=device, dtype=torch.long)

            # Predict noise epsilon
            noise_pred = self.unet(z_t, t_tensor, context_tokens, spatial_context)

            # Take reverse DDIM step
            z_t, _ = self.scheduler.ddim_step(
                model_output=noise_pred,
                timestep=t,
                prev_timestep=t_prev,
                sample=z_t,
                eta=eta,
            )

        # 4. Decode denoised latents back to image space (B, C, 240, 240)
        generated_slices = self.vae.decode(z_t)
        return generated_slices

    @torch.no_grad()
    def reconstruct_volume(
        self,
        sparse_volume: Union[np.ndarray, torch.Tensor],
        acquired_indices: List[int],
        stride: int = 4,
        total_slices: int = 155,
        num_inference_steps: int = 25,
        batch_size: int = 8,
        apply_smoothing: bool = True,
    ) -> torch.Tensor:
        """
        Takes a sparse 3D volume, reconstructs all missing slices,
        and assembles the complete dense 3D volume.
        """
        is_tensor = isinstance(sparse_volume, torch.Tensor)
        device = next(self.parameters()).device

        if is_tensor:
            vol = sparse_volume.clone().to(device)
        else:
            vol = torch.from_numpy(np.copy(sparse_volume)).to(device)

        sparsifier = VolumeSparsifier(stride=stride, total_slices=total_slices)
        missing_indices = sparsifier.get_missing_indices(total_slices=total_slices, stride=stride)

        if len(missing_indices) == 0:
            return vol

        # Collect metadata for all missing slices
        missing_contexts = []
        for target_z in missing_indices:
            ctx = sparsifier.get_context_for_slice(target_z, acquired_indices, total_slices)
            missing_contexts.append(ctx)

        # Process missing slices in mini-batches
        dense_vol = vol.clone()

        for start_idx in range(0, len(missing_contexts), batch_size):
            batch_ctx = missing_contexts[start_idx : start_idx + batch_size]
            B = len(batch_ctx)

            prev_slices = torch.stack([vol[c["prev_z"]] for c in batch_ctx], dim=0) # (B, C, H, W)
            next_slices = torch.stack([vol[c["next_z"]] for c in batch_ctx], dim=0) # (B, C, H, W)
            norm_zs = torch.tensor([c["normalized_z"] for c in batch_ctx], device=device, dtype=torch.float32)
            alphas = torch.tensor([c["alpha"] for c in batch_ctx], device=device, dtype=torch.float32)
            gap_dists = torch.tensor([c["gap_distance"] for c in batch_ctx], device=device, dtype=torch.float32)

            gen_slices = self.generate_slices_batch(
                prev_slices=prev_slices,
                next_slices=next_slices,
                normalized_zs=norm_zs,
                alphas=alphas,
                gap_dists=gap_dists,
                num_inference_steps=num_inference_steps,
            )

            # Assign generated slices into volume
            for i, c in enumerate(batch_ctx):
                target_z = c["target_z"]
                dense_vol[target_z] = gen_slices[i]

        # Optional inter-slice boundary smoothing
        if apply_smoothing:
            # Apply subtle 1D filter along z-axis to missing slices to smooth boundary transitions
            vol_np = dense_vol.cpu().numpy()
            smoothed_np = scipy.ndimage.gaussian_filter(vol_np, sigma=(0.5, 0.0, 0.0, 0.0))
            # Preserve original acquired slices exactly
            smoothed_np[acquired_indices] = vol_np[acquired_indices]
            dense_vol = torch.from_numpy(smoothed_np).to(device)

        return dense_vol
