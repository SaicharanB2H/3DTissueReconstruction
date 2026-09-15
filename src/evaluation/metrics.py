"""
Quantitative Evaluation Metrics for 3D Medical Volume Reconstruction.
Implements PSNR, SSIM, MAE, MSE, and BraTS Tumor Segmentation Dice Score.
"""
from typing import Dict, Optional, Union, Tuple, List
import numpy as np
import torch
import torch.nn.functional as F


def calculate_mse(pred: torch.Tensor, target: torch.Tensor, mask: Optional[torch.Tensor] = None) -> float:
    """Mean Squared Error."""
    if mask is not None:
        valid = mask > 0
        if valid.sum() == 0:
            return 0.0
        return float(F.mse_loss(pred[valid], target[valid]).item())
    return float(F.mse_loss(pred, target).item())


def calculate_mae(pred: torch.Tensor, target: torch.Tensor, mask: Optional[torch.Tensor] = None) -> float:
    """Mean Absolute Error."""
    if mask is not None:
        valid = mask > 0
        if valid.sum() == 0:
            return 0.0
        return float(F.l1_loss(pred[valid], target[valid]).item())
    return float(F.l1_loss(pred, target).item())


def calculate_psnr(
    pred: torch.Tensor,
    target: torch.Tensor,
    data_range: float = 2.0, # Default for [-1, 1] range
    mask: Optional[torch.Tensor] = None,
    eps: float = 1e-10,
) -> float:
    """
    Peak Signal-to-Noise Ratio (dB).
    """
    mse = calculate_mse(pred, target, mask=mask)
    if mse <= eps:
        return 100.0
    return float(10.0 * np.log10((data_range ** 2) / mse))


def _gaussian_window_2d(window_size: int = 11, sigma: float = 1.5) -> torch.Tensor:
    coords = torch.arange(window_size, dtype=torch.float32) - (window_size - 1) / 2.0
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g = g / g.sum()
    window = g.unsqueeze(1) @ g.unsqueeze(0)
    return window.unsqueeze(0).unsqueeze(0) # (1, 1, H, W)


def calculate_ssim(
    pred: torch.Tensor,
    target: torch.Tensor,
    data_range: float = 2.0,
    window_size: int = 11,
    k1: float = 0.01,
    k2: float = 0.03,
) -> float:
    """
    Structural Similarity Index (SSIM) computed across 2D slices and averaged across volume.
    Expects pred and target of shape (D, C, H, W) or (B, C, H, W) or (H, W).
    """
    if pred.ndim == 2:
        pred = pred.unsqueeze(0).unsqueeze(0)
        target = target.unsqueeze(0).unsqueeze(0)
    elif pred.ndim == 3: # (D, H, W) or (C, H, W)
        pred = pred.unsqueeze(1) if pred.shape[0] > 10 else pred.unsqueeze(0)
        target = target.unsqueeze(1) if target.shape[0] > 10 else target.unsqueeze(0)

    # Flatten outer batch/depth dims to (N, 1, H, W)
    if pred.ndim == 4:
        N, C, H, W = pred.shape
        pred_flat = pred.reshape(N * C, 1, H, W)
        target_flat = target.reshape(N * C, 1, H, W)
    else:
        pred_flat = pred
        target_flat = target

    device = pred.device
    dtype = pred.dtype
    window = _gaussian_window_2d(window_size).to(device=device, dtype=dtype)
    channel = 1

    c1 = (k1 * data_range) ** 2
    c2 = (k2 * data_range) ** 2

    # Convolutions
    mu1 = F.conv2d(pred_flat, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(target_flat, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(pred_flat * pred_flat, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(target_flat * target_flat, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(pred_flat * target_flat, window, padding=window_size // 2, groups=channel) - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + c1) * (2 * sigma12 + c2)) / (
        (mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2) + 1e-8
    )

    return float(ssim_map.mean().item())


def calculate_dice(
    pred_mask: torch.Tensor,
    target_mask: torch.Tensor,
    smooth: float = 1e-5,
) -> float:
    """
    Dice Similarity Coefficient for binary segmentation masks.
    """
    pred_bin = (pred_mask > 0.5).float()
    target_bin = (target_mask > 0.5).float()

    intersection = (pred_bin * target_bin).sum()
    union = pred_bin.sum() + target_bin.sum()

    if union == 0:
        return 1.0 # Both empty -> perfect agreement

    dice = (2.0 * intersection + smooth) / (union + smooth)
    return float(dice.item())


def evaluate_volume_reconstruction(
    pred_volume: torch.Tensor,
    gt_volume: torch.Tensor,
    missing_indices: Optional[List[int]] = None,
    mask_volume: Optional[torch.Tensor] = None,
    data_range: float = 2.0,
) -> Dict[str, float]:
    """
    Compute comprehensive metrics on full volume and specifically on missing slices.
    """
    pred_vol = pred_volume.detach().float()
    gt_vol = gt_volume.detach().float()

    # Metrics on full volume
    full_mse = calculate_mse(pred_vol, gt_vol)
    full_mae = calculate_mae(pred_vol, gt_vol)
    full_psnr = calculate_psnr(pred_vol, gt_vol, data_range=data_range)
    full_ssim = calculate_ssim(pred_vol, gt_vol, data_range=data_range)

    results = {
        "psnr_full": full_psnr,
        "ssim_full": full_ssim,
        "mae_full": full_mae,
        "mse_full": full_mse,
    }

    # Metrics specifically on synthesized / missing slices (crucial for evaluating interpolation quality)
    if missing_indices is not None and len(missing_indices) > 0:
        pred_missing = pred_vol[missing_indices]
        gt_missing = gt_vol[missing_indices]

        results["psnr_missing"] = calculate_psnr(pred_missing, gt_missing, data_range=data_range)
        results["ssim_missing"] = calculate_ssim(pred_missing, gt_missing, data_range=data_range)
        results["mae_missing"] = calculate_mae(pred_missing, gt_missing)
        results["mse_missing"] = calculate_mse(pred_missing, gt_missing)

    return results
