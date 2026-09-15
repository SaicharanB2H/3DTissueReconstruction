"""
Multi-Planar 2D Slice Visualizer & Error Difference Map Generator.
Renders Axial, Coronal, and Sagittal orthogonal cross-sections with PSNR/SSIM badges.
"""
from typing import Dict, Optional, List, Tuple, Union
import os
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

from src.evaluation.metrics import calculate_psnr, calculate_ssim


class SliceComparisonViewer:
    """
    Creates comprehensive visual comparisons between Ground Truth, Interpolation Baselines,
    and Conditional Diffusion Model reconstructions.
    """

    def __init__(self, output_dir: str = "./results/visualizations"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    @staticmethod
    def _to_numpy(x: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
        if isinstance(x, torch.Tensor):
            return x.detach().cpu().numpy()
        return np.copy(x)

    def plot_slice_grid(
        self,
        gt_slice: Union[np.ndarray, torch.Tensor],
        predictions: Dict[str, Union[np.ndarray, torch.Tensor]],
        slice_idx: int,
        volume_id: int,
        stride: int,
        save_name: Optional[str] = None,
        data_range: float = 2.0,
    ) -> str:
        """
        Plots a side-by-side row: Ground Truth, Model 1, Model 2, ..., Model N,
        followed by a bottom row of Absolute Difference Error Heatmaps.
        """
        gt = self._to_numpy(gt_slice)
        if gt.ndim == 3 and gt.shape[0] in [1, 4]: # (C, H, W)
            gt = gt[0] # Take first channel for display

        methods = list(predictions.keys())
        n_cols = len(methods) + 1

        fig, axes = plt.subplots(2, n_cols, figsize=(4.0 * n_cols, 8.0))
        plt.subplots_adjust(wspace=0.05, hspace=0.15)

        # Top-Left: Ground Truth
        axes[0, 0].imshow(gt, cmap="gray", vmin=-1.0, vmax=1.0)
        axes[0, 0].set_title(f"Ground Truth\n(Slice {slice_idx}, Vol {volume_id})", fontsize=13, fontweight="bold")
        axes[0, 0].axis("off")

        axes[1, 0].text(
            0.5, 0.5, "Error Heatmaps\n(|Pred - GT|)",
            ha="center", va="center", fontsize=14, fontweight="bold", color="darkred"
        )
        axes[1, 0].axis("off")

        # Plot predictions and difference maps
        for col_idx, method in enumerate(methods, start=1):
            pred = self._to_numpy(predictions[method])
            if pred.ndim == 3 and pred.shape[0] in [1, 4]:
                pred = pred[0]

            # Compute slice metrics
            psnr_val = calculate_psnr(
                torch.from_numpy(pred), torch.from_numpy(gt), data_range=data_range
            )
            ssim_val = calculate_ssim(
                torch.from_numpy(pred), torch.from_numpy(gt), data_range=data_range
            )

            # Top row: Predicted slice
            axes[0, col_idx].imshow(pred, cmap="gray", vmin=-1.0, vmax=1.0)
            axes[0, col_idx].set_title(f"{method}\nPSNR: {psnr_val:.2f}dB | SSIM: {ssim_val:.3f}", fontsize=12)
            axes[0, col_idx].axis("off")

            # Bottom row: Error Heatmap
            diff = np.abs(pred - gt)
            im_diff = axes[1, col_idx].imshow(diff, cmap="inferno", vmin=0.0, vmax=1.0)
            axes[1, col_idx].set_title(f"{method} Error\nMAE: {diff.mean():.4f}", fontsize=11)
            axes[1, col_idx].axis("off")

        # Colorbar for error heatmap
        cbar_ax = fig.add_axes([0.92, 0.12, 0.015, 0.35])
        fig.colorbar(im_diff, cax=cbar_ax)

        fig.suptitle(f"Reconstruction Comparison at Stride {stride}x — Volume {volume_id} Slice {slice_idx}", fontsize=15, y=0.98)

        if save_name is None:
            save_name = f"vol_{volume_id}_slice_{slice_idx}_stride_{stride}x.png"
        save_path = os.path.join(self.output_dir, save_name)
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        return save_path

    def plot_orthogonal_views(
        self,
        gt_volume: Union[np.ndarray, torch.Tensor],
        recon_volume: Union[np.ndarray, torch.Tensor],
        method_name: str,
        volume_id: int,
        stride: int,
        save_name: Optional[str] = None,
    ) -> str:
        """
        Plots Axial, Coronal, and Sagittal orthogonal cross-sections comparing GT vs Reconstructed.
        """
        gt_vol = self._to_numpy(gt_volume)
        rec_vol = self._to_numpy(recon_volume)

        if gt_vol.ndim == 4: # (D, C, H, W)
            gt_vol = gt_vol[:, 0, :, :]
            rec_vol = rec_vol[:, 0, :, :]

        D, H, W = gt_vol.shape
        z_mid, y_mid, x_mid = D // 2, H // 2, W // 2

        fig, axes = plt.subplots(3, 3, figsize=(12, 12))

        # Axial view (z-slice)
        axes[0, 0].imshow(gt_vol[z_mid], cmap="gray", vmin=-1.0, vmax=1.0)
        axes[0, 0].set_title(f"GT Axial (z={z_mid})", fontsize=12)
        axes[0, 1].imshow(rec_vol[z_mid], cmap="gray", vmin=-1.0, vmax=1.0)
        axes[0, 1].set_title(f"{method_name} Axial", fontsize=12)
        axes[0, 2].imshow(np.abs(gt_vol[z_mid] - rec_vol[z_mid]), cmap="inferno", vmin=0, vmax=0.8)
        axes[0, 2].set_title("Axial Error", fontsize=12)

        # Coronal view (y-slice)
        axes[1, 0].imshow(gt_vol[:, y_mid, :], cmap="gray", vmin=-1.0, vmax=1.0, aspect=H/D)
        axes[1, 0].set_title(f"GT Coronal (y={y_mid})", fontsize=12)
        axes[1, 1].imshow(rec_vol[:, y_mid, :], cmap="gray", vmin=-1.0, vmax=1.0, aspect=H/D)
        axes[1, 1].set_title(f"{method_name} Coronal", fontsize=12)
        axes[1, 2].imshow(np.abs(gt_vol[:, y_mid, :] - rec_vol[:, y_mid, :]), cmap="inferno", vmin=0, vmax=0.8, aspect=H/D)
        axes[1, 2].set_title("Coronal Error", fontsize=12)

        # Sagittal view (x-slice)
        axes[2, 0].imshow(gt_vol[:, :, x_mid], cmap="gray", vmin=-1.0, vmax=1.0, aspect=W/D)
        axes[2, 0].set_title(f"GT Sagittal (x={x_mid})", fontsize=12)
        axes[2, 1].imshow(rec_vol[:, :, x_mid], cmap="gray", vmin=-1.0, vmax=1.0, aspect=W/D)
        axes[2, 1].set_title(f"{method_name} Sagittal", fontsize=12)
        axes[2, 2].imshow(np.abs(gt_vol[:, :, x_mid] - rec_vol[:, :, x_mid]), cmap="inferno", vmin=0, vmax=0.8, aspect=W/D)
        axes[2, 2].set_title("Sagittal Error", fontsize=12)

        for ax in axes.flat:
            ax.axis("off")

        fig.suptitle(f"Orthogonal Views: GT vs {method_name} (Stride {stride}x, Vol {volume_id})", fontsize=15)

        if save_name is None:
            save_name = f"orthogonal_vol_{volume_id}_stride_{stride}x_{method_name}.png"
        save_path = os.path.join(self.output_dir, save_name)
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        return save_path
