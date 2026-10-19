"""
Synthetic Sparsification & Gap Context Extraction Module for 3D Medical Volumes.
"""
from typing import List, Tuple, Dict, Any, Optional, Union
import numpy as np
import torch


class VolumeSparsifier:
    """
    Subsamples a 3D medical volume along the z-axis (depth/axial slices)
    to simulate sparse slice acquisition, while preserving ground truth for evaluation.
    """

    def __init__(self, stride: int = 4, total_slices: int = 155):
        if not isinstance(stride, int) or stride < 1 or not isinstance(total_slices, int) or total_slices < 1:
            raise ValueError("Stride and total_slices must be positive integers")
        self.stride = stride
        self.total_slices = total_slices

    def get_acquired_indices(self, total_slices: Optional[int] = None, stride: Optional[int] = None) -> List[int]:
        """
        Compute indices of acquired slices with boundary preservation (0 and D-1).
        """
        D = self.total_slices if total_slices is None else total_slices
        s = self.stride if stride is None else stride
        if not isinstance(D, int) or D < 1 or not isinstance(s, int) or s < 1:
            raise ValueError("Stride and depth must be positive integers")
        indices = list(range(0, D, s))
        if (D - 1) not in indices:
            indices.append(D - 1)
        indices = sorted(list(set(indices)))
        return indices

    def get_missing_indices(self, total_slices: Optional[int] = None, stride: Optional[int] = None) -> List[int]:
        """
        Compute indices of missing/unsampled slices.
        """
        D = self.total_slices if total_slices is None else total_slices
        acquired = set(self.get_acquired_indices(D, stride))
        return [i for i in range(D) if i not in acquired]

    def get_context_for_slice(
        self,
        target_z: int,
        acquired_indices: List[int],
        total_slices: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        For a given target missing slice index target_z, locate:
        - prev_z (nearest acquired slice below target_z)
        - next_z (nearest acquired slice above target_z)
        - relative_pos alpha = (target_z - prev_z) / (next_z - prev_z) in (0, 1)
        - normalized_z = target_z / (total_slices - 1) in [0, 1]
        - gap_distance = next_z - prev_z
        """
        D = self.total_slices if total_slices is None else total_slices
        if not isinstance(D, int) or D < 1 or not isinstance(target_z, int) or not 0 <= target_z < D:
            raise ValueError('Target depth must be an integer within the volume')
        if not acquired_indices:
            raise ValueError('Acquired indices must not be empty')
        lower_candidates = [idx for idx in acquired_indices if idx <= target_z]
        upper_candidates = [idx for idx in acquired_indices if idx >= target_z]

        prev_z = max(lower_candidates) if lower_candidates else acquired_indices[0]
        next_z = min(upper_candidates) if upper_candidates else acquired_indices[-1]

        if next_z == prev_z:
            alpha = 0.5
            gap_dist = 1
        else:
            alpha = float(target_z - prev_z) / float(next_z - prev_z)
            gap_dist = next_z - prev_z

        normalized_z = float(target_z) / max(1.0, float(D - 1))

        return {
            "target_z": target_z,
            "prev_z": prev_z,
            "next_z": next_z,
            "alpha": alpha,
            "normalized_z": normalized_z,
            "gap_distance": gap_dist,
        }

    def sparsify_volume(
        self,
        volume: Union[np.ndarray, torch.Tensor],
        stride: Optional[int] = None,
    ) -> Tuple[Union[np.ndarray, torch.Tensor], np.ndarray, List[int], List[int]]:
        """
        Given a depth-first volume of shape (D,H,W) or (D,C,H,W),
        returns:
        - sparse_volume: missing slices zeroed out
        - mask: boolean mask of shape (D,) where True indicates acquired slice
        - acquired_indices
        - missing_indices
        """
        s = self.stride if stride is None else stride
        is_tensor = isinstance(volume, torch.Tensor)
        # Explicit depth-first convention avoids ambiguous modality/depth axes.
        if is_tensor:
            shape = list(volume.shape)
        else:
            shape = list(volume.shape)

        if len(shape) not in (3, 4):
            raise ValueError("Expected depth-first (D,H,W) or (D,C,H,W) volume")
        D = shape[0]
        depth_axis = 0

        acquired_indices = self.get_acquired_indices(D, s)
        missing_indices = self.get_missing_indices(D, s)

        mask = np.zeros(D, dtype=bool)
        mask[acquired_indices] = True

        if is_tensor:
            sparse_vol = volume.clone()
            if depth_axis == 0:
                sparse_vol[missing_indices] = 0.0
            else:
                sparse_vol[:, missing_indices] = 0.0
        else:
            sparse_vol = np.copy(volume)
            if depth_axis == 0:
                sparse_vol[missing_indices] = 0.0
            else:
                sparse_vol[:, missing_indices] = 0.0

        return sparse_vol, mask, acquired_indices, missing_indices
