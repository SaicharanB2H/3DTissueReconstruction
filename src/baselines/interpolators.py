"""
Classical Slice-to-Volume Interpolation Baselines.
Implements Nearest Neighbor, Linear, Cubic Spline, and 3D B-Spline volumetric interpolation.
"""
from typing import List, Optional, Union
import numpy as np
import torch
from scipy.interpolate import interp1d, CubicSpline
import scipy.ndimage


class BaseVolumeInterpolator:
    """
    Base class for volumetric slice reconstruction baselines.
    """
    def reconstruct(
        self,
        sparse_volume: Union[np.ndarray, torch.Tensor],
        acquired_indices: List[int],
        total_slices: int = 155,
    ) -> Union[np.ndarray, torch.Tensor]:
        raise NotImplementedError


class NearestSliceInterpolator(BaseVolumeInterpolator):
    """
    Nearest-neighbor slice interpolation.
    """
    def reconstruct(
        self,
        sparse_volume: Union[np.ndarray, torch.Tensor],
        acquired_indices: List[int],
        total_slices: int = 155,
    ) -> Union[np.ndarray, torch.Tensor]:
        is_tensor = isinstance(sparse_volume, torch.Tensor)
        device = sparse_volume.device if is_tensor else None
        vol = sparse_volume.detach().cpu().numpy() if is_tensor else np.copy(sparse_volume)

        # vol shape: (D, C, H, W) or (D, H, W)
        acquired = np.array(sorted(acquired_indices))
        dense_vol = np.zeros_like(vol)

        for z in range(total_slices):
            # Nearest acquired index
            nearest_idx = acquired[np.argmin(np.abs(acquired - z))]
            dense_vol[z] = vol[nearest_idx]

        if is_tensor:
            return torch.from_numpy(dense_vol).to(device)
        return dense_vol


class LinearSliceInterpolator(BaseVolumeInterpolator):
    """
    Piecewise linear inter-slice interpolation.
    Given slice z between prev_z and next_z:
    I(z) = (1 - alpha) * I(prev_z) + alpha * I(next_z)
    """
    def reconstruct(
        self,
        sparse_volume: Union[np.ndarray, torch.Tensor],
        acquired_indices: List[int],
        total_slices: int = 155,
    ) -> Union[np.ndarray, torch.Tensor]:
        is_tensor = isinstance(sparse_volume, torch.Tensor)
        device = sparse_volume.device if is_tensor else None
        vol = sparse_volume.detach().cpu().numpy() if is_tensor else np.copy(sparse_volume)

        acquired = sorted(acquired_indices)
        dense_vol = np.zeros_like(vol)

        # Retain acquired slices
        for z in acquired:
            dense_vol[z] = vol[z]

        # Interpolate missing slices
        for i in range(len(acquired) - 1):
            z_low = acquired[i]
            z_high = acquired[i + 1]
            gap = z_high - z_low
            if gap <= 1:
                continue

            v_low = vol[z_low]
            v_high = vol[z_high]

            for z in range(z_low + 1, z_high):
                alpha = float(z - z_low) / float(gap)
                dense_vol[z] = (1.0 - alpha) * v_low + alpha * v_high

        if is_tensor:
            return torch.from_numpy(dense_vol).to(device)
        return dense_vol


class CubicSplineSliceInterpolator(BaseVolumeInterpolator):
    """
    1D Cubic Spline interpolation along the z-axis (depth).
    """
    def reconstruct(
        self,
        sparse_volume: Union[np.ndarray, torch.Tensor],
        acquired_indices: List[int],
        total_slices: int = 155,
    ) -> Union[np.ndarray, torch.Tensor]:
        is_tensor = isinstance(sparse_volume, torch.Tensor)
        device = sparse_volume.device if is_tensor else None
        vol = sparse_volume.detach().cpu().numpy() if is_tensor else np.copy(sparse_volume)

        acquired = np.array(sorted(acquired_indices))
        # Shape: (D, ...)
        orig_shape = vol.shape
        D = orig_shape[0]
        spatial_dim = np.prod(orig_shape[1:])

        # Reshape to (D_acquired, spatial_dim)
        vol_acquired = vol[acquired].reshape(len(acquired), spatial_dim)

        # Fit cubic spline along z
        spline = CubicSpline(acquired, vol_acquired, axis=0, extrapolate=True)
        z_all = np.arange(total_slices)
        dense_flat = spline(z_all) # (total_slices, spatial_dim)

        dense_vol = dense_flat.reshape((total_slices,) + orig_shape[1:])

        # Retain exact acquired slices
        dense_vol[acquired] = vol[acquired]

        if is_tensor:
            return torch.from_numpy(dense_vol.astype(np.float32)).to(device)
        return dense_vol.astype(np.float32)


class BSpline3DVolumeInterpolator(BaseVolumeInterpolator):
    """
    Volumetric 3D B-spline / trilinear interpolation using scipy.ndimage map_coordinates.
    """
    def reconstruct(
        self,
        sparse_volume: Union[np.ndarray, torch.Tensor],
        acquired_indices: List[int],
        total_slices: int = 155,
    ) -> Union[np.ndarray, torch.Tensor]:
        is_tensor = isinstance(sparse_volume, torch.Tensor)
        device = sparse_volume.device if is_tensor else None
        vol = sparse_volume.detach().cpu().numpy() if is_tensor else np.copy(sparse_volume)

        acquired = np.array(sorted(acquired_indices))
        # Start with linear interpolation as robust initial grid
        linear_interp = LinearSliceInterpolator()
        init_dense = linear_interp.reconstruct(vol, acquired_indices, total_slices)

        # Apply mild 3D B-spline smoothing filter along z-axis only to missing slices
        # Gaussian blur along z-axis (axis 0)
        sigma = tuple([0.7] + [0.0] * (init_dense.ndim - 1))
        smoothed = scipy.ndimage.gaussian_filter(init_dense, sigma=sigma)
        # Keep exact acquired slices untouched
        smoothed[acquired] = vol[acquired]

        if is_tensor:
            return torch.from_numpy(smoothed.astype(np.float32)).to(device)
        return smoothed.astype(np.float32)


def get_baseline_interpolator(name: str) -> BaseVolumeInterpolator:
    name = name.lower()
    if name in ["nearest", "nn"]:
        return NearestSliceInterpolator()
    elif name in ["linear", "bilinear"]:
        return LinearSliceInterpolator()
    elif name in ["cubic", "spline", "cubic_spline"]:
        return CubicSplineSliceInterpolator()
    elif name in ["bspline_3d", "bspline", "3d_spline"]:
        return BSpline3DVolumeInterpolator()
    else:
        raise ValueError(f"Unknown baseline name: {name}")
