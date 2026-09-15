"""
Unit tests for Metrics and Baselines.
"""
import pytest
import torch
import numpy as np

from src.evaluation.metrics import (
    calculate_mse,
    calculate_mae,
    calculate_psnr,
    calculate_ssim,
    calculate_dice,
    evaluate_volume_reconstruction,
)
from src.baselines.interpolators import (
    NearestSliceInterpolator,
    LinearSliceInterpolator,
    CubicSplineSliceInterpolator,
    BSpline3DVolumeInterpolator,
)


def test_metrics_perfect_match():
    x = torch.randn(10, 1, 64, 64)
    assert calculate_mse(x, x) == 0.0
    assert calculate_mae(x, x) == 0.0
    assert calculate_psnr(x, x) >= 99.0
    assert np.isclose(calculate_ssim(x, x), 1.0, atol=1e-3)


def test_metrics_perturbed():
    x = torch.zeros(10, 1, 64, 64)
    y = torch.ones(10, 1, 64, 64) * 0.1

    mse = calculate_mse(y, x)
    assert np.isclose(mse, 0.01, atol=1e-4)

    mae = calculate_mae(y, x)
    assert np.isclose(mae, 0.1, atol=1e-4)

    # Data range 2.0 (for [-1, 1]): PSNR = 10 * log10(4 / 0.01) = 10 * log10(400) ~ 26.02 dB
    psnr = calculate_psnr(y, x, data_range=2.0)
    assert np.isclose(psnr, 26.02, atol=0.1)


def test_dice_score():
    m1 = torch.zeros(1, 64, 64)
    m1[:, 20:40, 20:40] = 1.0

    m2 = torch.zeros(1, 64, 64)
    m2[:, 20:40, 20:40] = 1.0

    # Perfect overlap -> Dice = 1.0
    assert calculate_dice(m1, m2) == 1.0

    # No overlap -> Dice = 0.0
    m3 = torch.zeros(1, 64, 64)
    m3[:, 0:10, 0:10] = 1.0
    assert np.isclose(calculate_dice(m1, m3), 0.0, atol=1e-3)


def test_baseline_interpolators():
    D, C, H, W = 16, 1, 32, 32
    # Create smooth synthetic volume along z
    vol = np.zeros((D, C, H, W), dtype=np.float32)
    for z in range(D):
        vol[z] = float(z) / float(D - 1)

    acquired = [0, 4, 8, 12, 15]

    for InterpolatorClass in [
        NearestSliceInterpolator,
        LinearSliceInterpolator,
        CubicSplineSliceInterpolator,
        BSpline3DVolumeInterpolator,
    ]:
        interp = InterpolatorClass()
        recon = interp.reconstruct(vol, acquired, total_slices=D)

        assert recon.shape == vol.shape
        # Acquired slices must match exactly
        np.testing.assert_allclose(recon[acquired], vol[acquired], atol=1e-5)
        # Missing slices should be in valid range [0, 1]
        assert recon.min() >= -0.1
        assert recon.max() <= 1.1
