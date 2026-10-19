"""
PyTorch Dataset and DataModule for BraTS2020 Multi-Modal MRI Volumes and Slices.
Ensures strict patient/volume-level data separation with zero slice leakage.
"""
import os
import glob
from typing import List, Tuple, Dict, Any, Optional, Union
import numpy as np
import pandas as pd
import h5py
import torch
from torch.utils.data import Dataset, DataLoader

from src.data.transforms import MedicalIntensityNormalizer, MedicalSpatialAugmenter
from src.data.sparsifier import VolumeSparsifier
from src.data.preprocessing import normalize_acquired_volume


MODALITY_MAP = {
    "flair": 0,
    "t1": 1,
    "t1ce": 2,
    "t2": 3,
}


def configured_patient_splits(config, seed=None):
    """Apply the configured metadata path, split ratios and partition seed."""
    from pathlib import Path
    root = Path(__file__).resolve().parents[2]
    data = config['data']
    ratios = data.get('split_ratios', {'train': .7, 'val': .15, 'test': .15})
    metadata = data.get('metadata_file')
    return get_patient_splits(str(root / data['data_dir']),
        metadata_file=str(root / metadata) if metadata else None,
        train_ratio=ratios['train'], val_ratio=ratios['val'], test_ratio=ratios['test'],
        seed=config['project']['seed'] if seed is None else seed)


import functools


@functools.lru_cache(maxsize=128)
def _load_slice_from_h5(file_path: str, modality: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Cached reader for individual slice files to minimize disk I/O latency.
    """
    with h5py.File(file_path, "r") as f:
        img = f["image"][()]  # shape (240, 240, 4)
        mask = f["mask"][()] if "mask" in f else np.zeros((240, 240, 3), dtype=np.uint8)

    # Select modality
    if modality in MODALITY_MAP:
        ch = MODALITY_MAP[modality]
        img = img[:, :, ch : ch + 1]  # shape (240, 240, 1)
    elif modality == "all":
        pass  # Keep all 4 channels (240, 240, 4)
    else:
        raise ValueError(f"Unknown modality: {modality}")

    # Transpose to (C, H, W)
    img = np.transpose(img, (2, 0, 1)).astype(np.float32)
    mask = np.transpose(mask, (2, 0, 1)).astype(np.float32)
    return img, mask


def get_patient_splits(
    data_dir: str,
    metadata_file: Optional[str] = None,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
) -> Tuple[List[int], List[int], List[int]]:
    """
    Split volume IDs deterministically at patient level.
    """
    if metadata_file is None:
        auto_meta = os.path.join(data_dir, "meta_data.csv")
        if os.path.exists(auto_meta):
            metadata_file = auto_meta

    if metadata_file and os.path.exists(metadata_file):
        df = pd.read_csv(metadata_file)
        unique_vols = sorted(df["volume"].unique().tolist())
    else:
        # Discover from files
        h5_files = glob.glob(os.path.join(data_dir, "volume_*_slice_*.h5"))
        vol_ids = set()
        for f in h5_files:
            base = os.path.basename(f)
            # volume_100_slice_0.h5
            parts = base.split("_")
            if len(parts) >= 4 and parts[0] == "volume":
                vol_ids.add(int(parts[1]))
        unique_vols = sorted(list(vol_ids))

    if not np.isclose(train_ratio + val_ratio + test_ratio, 1.0) or min(train_ratio, val_ratio, test_ratio) < 0:
        raise ValueError("Split ratios must be nonnegative and sum to one")
    shuffled_vols = np.random.RandomState(seed).permutation(unique_vols).tolist()

    n_total = len(shuffled_vols)
    n_train = int(n_total * train_ratio)
    n_val = int(n_total * val_ratio)

    train_vols = sorted(shuffled_vols[:n_train])
    val_vols = sorted(shuffled_vols[n_train : n_train + n_val])
    test_vols = sorted(shuffled_vols[n_train + n_val :])

    return train_vols, val_vols, test_vols


@functools.lru_cache(maxsize=1024)
def acquired_slice_stats(data_dir, vol_id, modality, acquired):
    images = [_load_slice_from_h5(os.path.join(data_dir, f"volume_{vol_id}_slice_{z}.h5"), modality)[0] for z in acquired]
    _, stats = MedicalIntensityNormalizer().normalize(np.stack(images))
    return stats


class BraTSSliceDataset(Dataset):
    """
    Dataset for slice-level training of VAE and Conditional Diffusion.
    For each target slice, extracts:
    - target_slice: ground truth slice (C, H, W)
    - prev_slice, next_slice: neighboring acquired context slices
    - position_info: alpha (relative position in gap), normalized_z (global depth)
    - mask: tumor segmentation mask (3, H, W)
    """

    def __init__(
        self,
        data_dir: str,
        volume_ids: List[int],
        modality: str = "flair",
        stride: int = 4,
        total_slices_per_vol: int = 155,
        is_training: bool = True,
        augment: bool = False,
        min_brain_pixels: int = 100,
        return_context: bool = True,
    ):
        self.data_dir = data_dir
        self.volume_ids = set(volume_ids)
        self.modality = modality.lower()
        if self.modality not in {*MODALITY_MAP, "all"}:
            raise ValueError(f"Unknown modality: {modality}")
        self.stride = stride
        self.total_slices = total_slices_per_vol
        self.is_training = is_training
        self.augment = augment and is_training
        self.return_context = return_context

        self.normalizer = MedicalIntensityNormalizer()
        self.augmenter = MedicalSpatialAugmenter()
        self.sparsifier = VolumeSparsifier(stride=stride, total_slices=total_slices_per_vol)

        # Build index of valid slices
        self.samples = []
        self._build_index(min_brain_pixels)

    def _build_index(self, min_brain_pixels: int):
        acquired_indices = self.sparsifier.get_acquired_indices(self.total_slices, self.stride)

        for vol_id in sorted(list(self.volume_ids)):
            for slice_idx in range(self.total_slices):
                if self.return_context and slice_idx in acquired_indices:
                    continue
                slice_path = os.path.join(self.data_dir, f"volume_{vol_id}_slice_{slice_idx}.h5")
                if not os.path.exists(slice_path):
                    continue

                # Filter out pure empty air background slices during training if desired
                # (keep them during validation for unbiased evaluation)
                ctx = self.sparsifier.get_context_for_slice(slice_idx, acquired_indices, self.total_slices)
                self.samples.append({
                    "vol_id": vol_id,
                    "target_z": slice_idx,
                    "prev_z": ctx["prev_z"],
                    "next_z": ctx["next_z"],
                    "alpha": ctx["alpha"],
                    "normalized_z": ctx["normalized_z"],
                    "gap_dist": ctx["gap_distance"],
                    "path_target": slice_path,
                    "path_prev": os.path.join(self.data_dir, f"volume_{vol_id}_slice_{ctx['prev_z']}.h5"),
                    "path_next": os.path.join(self.data_dir, f"volume_{vol_id}_slice_{ctx['next_z']}.h5"),
                })

    def __len__(self) -> int:
        return len(self.samples)

    def _load_slice(self, file_path: str) -> Tuple[np.ndarray, np.ndarray]:
        return _load_slice_from_h5(file_path, self.modality)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        info = self.samples[idx]

        target_img, target_mask = self._load_slice(info["path_target"])
        acquired = tuple(self.sparsifier.get_acquired_indices())
        stats = acquired_slice_stats(self.data_dir, info["vol_id"], self.modality, acquired)
        target_norm, _ = self.normalizer.normalize(target_img, stats=stats)

        if not self.return_context:
            # Fast-path for VAE training: only load and process target slice
            if self.augment:
                # Augment only target and mask
                target_norm, _, _, target_mask = self.augmenter(
                    target_norm, target_norm, target_norm, target_mask
                )

            return {
                "target": torch.from_numpy(np.ascontiguousarray(target_norm)),
                "mask": torch.from_numpy(np.ascontiguousarray(target_mask)),
                "alpha": torch.tensor(info["alpha"], dtype=torch.float32),
                "normalized_z": torch.tensor(info["normalized_z"], dtype=torch.float32),
                "gap_dist": torch.tensor(info["gap_dist"], dtype=torch.float32),
                "target_z": torch.tensor(info["target_z"], dtype=torch.long),
                "vol_id": torch.tensor(info["vol_id"], dtype=torch.long),
                "stats_min": stats[0],
                "stats_max": stats[1],
            }

        prev_img, _ = self._load_slice(info["path_prev"])
        next_img, _ = self._load_slice(info["path_next"])

        prev_norm, _ = self.normalizer.normalize(prev_img, stats=stats)
        next_norm, _ = self.normalizer.normalize(next_img, stats=stats)

        # Spatial augmentation across all context slices
        if self.augment:
            target_norm, prev_norm, next_norm, target_mask = self.augmenter(
                target_norm, prev_norm, next_norm, target_mask
            )

        # np.flip() returns views with negative strides which torch cannot handle;
        # np.ascontiguousarray() always produces a C-contiguous copy.
        return {
            "target": torch.from_numpy(np.ascontiguousarray(target_norm)),   # (C, H, W)
            "prev":   torch.from_numpy(np.ascontiguousarray(prev_norm)),     # (C, H, W)
            "next":   torch.from_numpy(np.ascontiguousarray(next_norm)),     # (C, H, W)
            "mask":   torch.from_numpy(np.ascontiguousarray(target_mask)),   # (3, H, W)
            "alpha":        torch.tensor(info["alpha"],        dtype=torch.float32),
            "normalized_z": torch.tensor(info["normalized_z"], dtype=torch.float32),
            "gap_dist":     torch.tensor(info["gap_dist"],     dtype=torch.float32),
            "target_z":     torch.tensor(info["target_z"],     dtype=torch.long),
            "vol_id":       torch.tensor(info["vol_id"],       dtype=torch.long),
            "stats_min": stats[0],
            "stats_max": stats[1],
        }


class BraTSVolumeDataset(Dataset):
    """
    Dataset for full 3D volume reconstruction and evaluation.
    Loads all 155 slices of a volume as a full 3D tensor (D, C, H, W).
    """

    def __init__(
        self,
        data_dir: str,
        volume_ids: List[int],
        modality: str = "flair",
        total_slices: int = 155,
    ):
        self.data_dir = data_dir
        self.volume_ids = sorted(volume_ids)
        self.modality = modality.lower()
        if self.modality not in {*MODALITY_MAP, "all"}:
            raise ValueError(f"Unknown modality: {modality}")
        self.total_slices = total_slices
        self.normalizer = MedicalIntensityNormalizer()

    def __len__(self) -> int:
        return len(self.volume_ids)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        vol_id = self.volume_ids[idx]
        slices = []
        masks = []

        for z in range(self.total_slices):
            slice_path = os.path.join(self.data_dir, f"volume_{vol_id}_slice_{z}.h5")
            with h5py.File(slice_path, "r") as f:
                img = f["image"][()]
                mask = f["mask"][()] if "mask" in f else np.zeros((240, 240, 3), dtype=np.uint8)

            if self.modality in MODALITY_MAP:
                ch = MODALITY_MAP[self.modality]
                img = img[:, :, ch : ch + 1]
            # (H, W, C) -> (C, H, W)
            img = np.transpose(img, (2, 0, 1)).astype(np.float32)
            mask = np.transpose(mask, (2, 0, 1)).astype(np.float32)

            slices.append(img)
            masks.append(mask)

        # Full volume: shape (D, C, H, W)
        vol = np.stack(slices, axis=0)
        mask_vol = np.stack(masks, axis=0)

        # Volume-level normalization
        acquired = VolumeSparsifier(total_slices=self.total_slices).get_acquired_indices()
        vol_norm, stats = normalize_acquired_volume(vol, acquired)

        return {
            "volume": torch.from_numpy(vol_norm),          # (D, C, H, W)
            "volume_raw": torch.from_numpy(vol),           # (D, C, H, W)
            "mask_volume": torch.from_numpy(mask_vol),     # (D, 3, H, W)
            "vol_id": vol_id,
            "stats_min": stats[0],
            "stats_max": stats[1],
        }
