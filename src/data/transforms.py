"""
Intensity Normalization and Spatial Transforms for Medical MRI Slices.
"""
from typing import Tuple, Optional, Union
import numpy as np
import torch


class MedicalIntensityNormalizer:
    """
    Robust intensity normalizer for MRI / CT volumes and slices.
    Maps foreground intensities to [-1, 1] range required for Diffusion & VAE models,
    with an approximate inverse; percentile clipping discards extreme intensities.
    """

    def __init__(
        self,
        clip_percentiles: Tuple[float, float] = (0.5, 99.5),
        target_range: Tuple[float, float] = (-1.0, 1.0),
        epsilon: float = 1e-7,
    ):
        self.p_low, self.p_high = clip_percentiles
        self.target_min, self.target_max = target_range
        self.epsilon = epsilon

    def normalize(
        self,
        img: Union[np.ndarray, torch.Tensor],
        stats: Optional[Tuple[float, float]] = None,
    ) -> Tuple[Union[np.ndarray, torch.Tensor], Tuple[float, float]]:
        """
        Normalize an image or volume to [target_min, target_max].
        Returns (normalized_img, (vmin, vmax)).
        """
        is_tensor = isinstance(img, torch.Tensor)
        if is_tensor:
            arr = img.detach().cpu().numpy()
        else:
            arr = np.copy(img)

        if stats is not None:
            vmin, vmax = stats
        else:
            non_zeros = arr[arr > self.epsilon]
            if len(non_zeros) > 0:
                vmin = float(np.percentile(non_zeros, self.p_low))
                vmax = float(np.percentile(non_zeros, self.p_high))
            else:
                vmin = float(arr.min())
                vmax = float(arr.max())

            if vmax - vmin < self.epsilon:
                vmax = vmin + 1.0

        if not np.isfinite(arr).all() or not np.isfinite([vmin, vmax]).all() or vmax <= vmin:
            raise ValueError("Intensities and normalization bounds must be finite; maximum must exceed minimum")
        # Clip and scale
        arr_clipped = np.clip(arr, vmin, vmax)
        arr_norm = (arr_clipped - vmin) / (vmax - vmin + self.epsilon)
        arr_scaled = arr_norm * (self.target_max - self.target_min) + self.target_min

        # Keep true background (exact zero) at target_min
        arr_scaled[arr <= self.epsilon] = self.target_min

        if is_tensor:
            res = torch.from_numpy(arr_scaled).to(img.device, dtype=img.dtype)
        else:
            res = arr_scaled.astype(np.float32)

        return res, (vmin, vmax)

    def denormalize(
        self,
        norm_img: Union[np.ndarray, torch.Tensor],
        stats: Tuple[float, float],
    ) -> Union[np.ndarray, torch.Tensor]:
        """
        Invert normalization back to original physical intensity range.
        """
        vmin, vmax = stats
        is_tensor = isinstance(norm_img, torch.Tensor)
        if is_tensor:
            arr = norm_img.detach().cpu().numpy()
        else:
            arr = np.copy(norm_img)

        # Scale from [target_min, target_max] to [0, 1]
        arr_01 = (arr - self.target_min) / (self.target_max - self.target_min + self.epsilon)
        arr_01 = np.clip(arr_01, 0.0, 1.0)
        orig = arr_01 * (vmax - vmin) + vmin

        # Preserve background zeroes
        orig[arr <= (self.target_min + 0.01)] = 0.0

        if is_tensor:
            return torch.from_numpy(orig).to(norm_img.device, dtype=norm_img.dtype)
        return orig.astype(np.float32)


class MedicalSpatialAugmenter:
    """
    Lightweight 2D spatial augmentations (flips, rotations) preserving anatomical fidelity.
    """

    def __init__(self, p_flip_h: float = 0.5, p_flip_v: float = 0.0):
        self.p_flip_h = p_flip_h
        self.p_flip_v = p_flip_v

    def __call__(
        self,
        *images: Union[np.ndarray, torch.Tensor]
    ) -> Tuple[Union[np.ndarray, torch.Tensor], ...]:
        do_h = np.random.rand() < self.p_flip_h
        do_v = np.random.rand() < self.p_flip_v

        results = []
        for img in images:
            if isinstance(img, torch.Tensor):
                if do_h:
                    img = torch.flip(img, dims=[-1])
                if do_v:
                    img = torch.flip(img, dims=[-2])
            else:
                if do_h:
                    img = np.flip(img, axis=-1)
                if do_v:
                    img = np.flip(img, axis=-2)
            results.append(img)

        return tuple(results) if len(results) > 1 else results[0]
