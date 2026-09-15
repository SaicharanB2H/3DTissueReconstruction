"""
Benchmark Evaluation Engine comparing Baselines vs Conditional Diffusion across Strides.
"""
from typing import List, Dict, Any, Optional
import os
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from src.data.dataset import BraTSVolumeDataset
from src.data.sparsifier import VolumeSparsifier
from src.baselines.interpolators import get_baseline_interpolator
from src.evaluation.metrics import evaluate_volume_reconstruction


class VolumeBenchmarkEvaluator:
    """
    Runs automated benchmark across test volumes and sparsification strides.
    """

    def __init__(
        self,
        dataset: BraTSVolumeDataset,
        strides: List[int] = [2, 4, 8, 16],
        baselines: List[str] = ["nearest", "linear", "cubic", "bspline_3d"],
        diffusion_pipeline: Optional[Any] = None,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        self.dataset = dataset
        self.strides = strides
        self.baselines = baselines
        self.diffusion_pipeline = diffusion_pipeline
        self.device = device

    def evaluate_all(self, max_volumes: Optional[int] = None) -> pd.DataFrame:
        """
        Evaluate all methods across all strides on the test dataset.
        """
        records = []
        n_vols = len(self.dataset) if max_volumes is None else min(max_volumes, len(self.dataset))

        for stride in self.strides:
            sparsifier = VolumeSparsifier(stride=stride, total_slices=self.dataset.total_slices)
            acquired_indices = sparsifier.get_acquired_indices()
            missing_indices = sparsifier.get_missing_indices()

            print(f"\n--- Benchmarking Stride {stride}x ({len(acquired_indices)} acquired, {len(missing_indices)} missing) ---")

            for vol_idx in tqdm(range(n_vols), desc=f"Stride {stride}x Evaluation"):
                item = self.dataset[vol_idx]
                gt_vol = item["volume"].to(self.device) # (D, C, H, W)
                vol_id = item["vol_id"]

                # 1. Sparsify volume
                sparse_vol, mask, _, _ = sparsifier.sparsify_volume(gt_vol, stride=stride)

                # 2. Evaluate classical baselines
                for baseline_name in self.baselines:
                    interpolator = get_baseline_interpolator(baseline_name)
                    recon_vol = interpolator.reconstruct(sparse_vol, acquired_indices, self.dataset.total_slices)

                    metrics = evaluate_volume_reconstruction(
                        recon_vol,
                        gt_vol,
                        missing_indices=missing_indices,
                    )

                    records.append({
                        "vol_id": vol_id,
                        "stride": stride,
                        "method": baseline_name.capitalize(),
                        "psnr_full": metrics["psnr_full"],
                        "ssim_full": metrics["ssim_full"],
                        "mae_full": metrics["mae_full"],
                        "mse_full": metrics["mse_full"],
                        "psnr_missing": metrics.get("psnr_missing", np.nan),
                        "ssim_missing": metrics.get("ssim_missing", np.nan),
                        "mae_missing": metrics.get("mae_missing", np.nan),
                        "mse_missing": metrics.get("mse_missing", np.nan),
                    })

                # 3. Evaluate Conditional Diffusion Model if provided
                if self.diffusion_pipeline is not None:
                    with torch.no_grad():
                        diff_recon = self.diffusion_pipeline.reconstruct_volume(
                            sparse_vol,
                            acquired_indices=acquired_indices,
                            stride=stride,
                        )

                    diff_metrics = evaluate_volume_reconstruction(
                        diff_recon,
                        gt_vol,
                        missing_indices=missing_indices,
                    )

                    records.append({
                        "vol_id": vol_id,
                        "stride": stride,
                        "method": "Conditional Diffusion",
                        "psnr_full": diff_metrics["psnr_full"],
                        "ssim_full": diff_metrics["ssim_full"],
                        "mae_full": diff_metrics["mae_full"],
                        "mse_full": diff_metrics["mse_full"],
                        "psnr_missing": diff_metrics.get("psnr_missing", np.nan),
                        "ssim_missing": diff_metrics.get("ssim_missing", np.nan),
                        "mae_missing": diff_metrics.get("mae_missing", np.nan),
                        "mse_missing": diff_metrics.get("mse_missing", np.nan),
                    })

        df = pd.DataFrame(records)
        return df

    @staticmethod
    def generate_summary_table(df: pd.DataFrame) -> pd.DataFrame:
        """
        Aggregate results (mean +/- std) grouped by Stride and Method.
        """
        summary = df.groupby(["stride", "method"]).agg({
            "psnr_full": ["mean", "std"],
            "ssim_full": ["mean", "std"],
            "mae_full": ["mean", "std"],
            "psnr_missing": ["mean", "std"],
            "ssim_missing": ["mean", "std"],
        }).reset_index()
        return summary
