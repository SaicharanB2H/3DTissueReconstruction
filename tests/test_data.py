"""
Unit tests for Data Pipeline, Transforms, Sparsification and Patient Splits.
"""
import pytest
import numpy as np
import torch
import os

from src.data.transforms import MedicalIntensityNormalizer, MedicalSpatialAugmenter
from src.data.sparsifier import VolumeSparsifier
from src.data.dataset import get_patient_splits, BraTSSliceDataset, BraTSVolumeDataset


def test_normalizer():
    np.random.seed(42)
    normalizer = MedicalIntensityNormalizer()
    img = np.random.uniform(0, 500, size=(1, 240, 240)).astype(np.float32)
    img[:, :50, :50] = 0.0  # Simulate background

    norm_img, stats = normalizer.normalize(img)
    assert norm_img.shape == img.shape
    assert norm_img.min() >= -1.0
    assert norm_img.max() <= 1.0

    denorm = normalizer.denormalize(norm_img, stats)
    assert denorm.shape == img.shape
    # Non-zero foreground within percentile bounds should match original
    mask = (img > stats[0]) & (img < stats[1])
    np.testing.assert_allclose(denorm[mask], img[mask], rtol=2e-2, atol=6.0)


def test_sparsifier():
    sparsifier = VolumeSparsifier(stride=4, total_slices=155)
    acquired = sparsifier.get_acquired_indices()
    missing = sparsifier.get_missing_indices()

    assert 0 in acquired
    assert 154 in acquired # Boundary preservation
    assert len(acquired) + len(missing) == 155
    assert set(acquired).isdisjoint(set(missing))

    # Test context for slice 6
    ctx = sparsifier.get_context_for_slice(6, acquired)
    assert ctx["target_z"] == 6
    assert ctx["prev_z"] == 4
    assert ctx["next_z"] == 8
    assert np.isclose(ctx["alpha"], 0.5)
    assert ctx["gap_distance"] == 4


def test_patient_splits():
    data_dir = r"BraTS2020_training_data\content\data"
    if not os.path.exists(data_dir):
        pytest.skip("Dataset directory not found")

    train_vols, val_vols, test_vols = get_patient_splits(data_dir, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15)

    assert len(train_vols) > 0
    assert len(val_vols) > 0
    assert len(test_vols) > 0

    # Ensure zero data leakage
    train_set = set(train_vols)
    val_set = set(val_vols)
    test_set = set(test_vols)

    assert train_set.isdisjoint(val_set)
    assert train_set.isdisjoint(test_set)
    assert val_set.isdisjoint(test_set)


def test_dataset_sample():
    data_dir = r"BraTS2020_training_data\content\data"
    if not os.path.exists(data_dir):
        pytest.skip("Dataset directory not found")

    train_vols, _, _ = get_patient_splits(data_dir)
    dataset = BraTSSliceDataset(data_dir, volume_ids=train_vols[:2], modality="flair", stride=4)

    assert len(dataset) > 0
    sample = dataset[10]

    assert sample["target"].shape == (1, 240, 240)
    assert sample["prev"].shape == (1, 240, 240)
    assert sample["next"].shape == (1, 240, 240)
    assert sample["mask"].shape == (3, 240, 240)
    assert 0.0 <= sample["alpha"].item() <= 1.0
    assert 0.0 <= sample["normalized_z"].item() <= 1.0
