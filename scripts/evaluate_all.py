"""
Comprehensive Benchmark Script: Evaluates Baselines vs Conditional Diffusion across Strides 2x, 4x, 8x, 16x.
"""
import os
import sys
import argparse
import yaml
import torch
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data.dataset import BraTSVolumeDataset, get_patient_splits
from src.models.slice_encoder import SharedSliceEncoder
from src.models.vae import MedicalAutoencoderKL
from src.models.unet_diffusion import ConditionalUNet2D
from src.models.scheduler import DiffusionNoiseScheduler
from src.models.pipeline import SparseToDenseDiffusionPipeline
from src.evaluation.evaluator import VolumeBenchmarkEvaluator
from src.utils.logger import setup_logger


def run_benchmark(
    max_test_vols: int = 5,
    strides: list = [2, 4, 8, 16],
    use_diffusion: bool = True,
    diffusion_ckpt: str = "./experiments/checkpoints/diffusion/best_model.pt",
    vae_ckpt: str = "./experiments/checkpoints/vae/best_model.pt",
):
    with open("configs/default_config.yaml", "r") as f:
        default_cfg = yaml.safe_load(f)
    with open("configs/diffusion_config.yaml", "r") as f:
        diff_cfg = yaml.safe_load(f)
    with open("configs/vae_config.yaml", "r") as f:
        vae_cfg = yaml.safe_load(f)

    device = default_cfg["project"]["device"] if torch.cuda.is_available() else "cpu"
    logger = setup_logger("BenchmarkEvaluator")
    logger.info(f"Running benchmark on device: {device}")

    # 1. Load Test Dataset
    data_dir = default_cfg["data"]["data_dir"]
    modality = default_cfg["data"]["modality"]
    _, _, test_vols = get_patient_splits(
        data_dir,
        train_ratio=default_cfg["data"]["split_ratios"]["train"],
        val_ratio=default_cfg["data"]["split_ratios"]["val"],
        test_ratio=default_cfg["data"]["split_ratios"]["test"],
        seed=default_cfg["project"]["seed"],
    )

    logger.info(f"Found {len(test_vols)} held-out test volumes. Evaluating on first {max_test_vols} volumes.")
    test_dataset = BraTSVolumeDataset(
        data_dir=data_dir,
        volume_ids=test_vols[:max_test_vols],
        modality=modality,
    )

    # 2. Build Pipeline
    pipeline = None
    if use_diffusion:
        in_channels = default_cfg["data"]["in_channels"]
        vae = MedicalAutoencoderKL(
            in_channels=in_channels,
            out_channels=in_channels,
            latent_channels=vae_cfg["model"]["latent_channels"],
            base_channels=vae_cfg["model"]["base_channels"],
            channel_multipliers=vae_cfg["model"]["channel_multipliers"],
            num_res_blocks=vae_cfg["model"]["num_res_blocks"],
        ).to(device)

        slice_encoder = SharedSliceEncoder(
            in_channels=in_channels,
            base_channels=32,
            context_dim=diff_cfg["unet"]["context_dim"],
            out_spatial_channels=8,
        ).to(device)

        unet = ConditionalUNet2D(
            in_channels=diff_cfg["unet"]["in_channels"],
            spatial_context_channels=8,
            out_channels=diff_cfg["unet"]["out_channels"],
            base_channels=diff_cfg["unet"]["base_channels"],
            channel_multipliers=diff_cfg["unet"]["channel_multipliers"],
            num_res_blocks=diff_cfg["unet"]["num_res_blocks"],
            context_dim=diff_cfg["unet"]["context_dim"],
            num_heads=diff_cfg["unet"]["num_heads"],
        ).to(device)

        scheduler = DiffusionNoiseScheduler(
            num_train_timesteps=diff_cfg["diffusion"]["num_train_timesteps"],
            beta_schedule=diff_cfg["diffusion"]["beta_schedule"],
        ).to(device)

        # Load weights if checkpoints exist
        if os.path.exists(diffusion_ckpt):
            logger.info(f"Loading diffusion checkpoint from {diffusion_ckpt}")
            ckpt = torch.load(diffusion_ckpt, map_location=device)
            if "slice_encoder" in ckpt["models"]:
                slice_encoder.load_state_dict(ckpt["models"]["slice_encoder"])
            if "unet" in ckpt["models"]:
                unet.load_state_dict(ckpt["models"]["unet"])

        if os.path.exists(vae_ckpt):
            logger.info(f"Loading VAE checkpoint from {vae_ckpt}")
            vae_state = torch.load(vae_ckpt, map_location=device)
            vae.load_state_dict(vae_state["models"]["vae"] if "models" in vae_state else vae_state)

        vae.eval()
        slice_encoder.eval()
        unet.eval()

        pipeline = SparseToDenseDiffusionPipeline(slice_encoder, vae, unet, scheduler)

    # 3. Run Benchmark Evaluator
    evaluator = VolumeBenchmarkEvaluator(
        dataset=test_dataset,
        strides=strides,
        baselines=["nearest", "linear", "cubic", "bspline_3d"],
        diffusion_pipeline=pipeline,
        device=device,
    )

    df_results = evaluator.evaluate_all(max_volumes=max_test_vols)

    # 4. Save and format results
    os.makedirs("./results", exist_ok=True)
    csv_path = "./results/benchmark_metrics.csv"
    df_results.to_csv(csv_path, index=False)
    logger.info(f"Saved detailed metrics to {csv_path}")

    # Summary table
    summary = df_results.groupby(["stride", "method"])[["psnr_full", "ssim_full", "mae_full", "psnr_missing", "ssim_missing"]].mean().reset_index()
    print("\n" + "="*80)
    print("BENCHMARK SUMMARY (Mean Metrics across Test Volumes)")
    print("="*80)
    print(summary.to_string(index=False))

    md_summary = summary.to_markdown(index=False)
    with open("./results/benchmark_summary.md", "w") as f:
        f.write("# Sparse-to-Dense 3D Medical Volume Reconstruction Benchmark\n\n")
        f.write(f"Primary Modality: **{modality.upper()}**\n\n")
        f.write(md_summary)

    logger.info("Saved summary Markdown table to ./results/benchmark_summary.md")
    return df_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--vols", type=int, default=3, help="Number of test volumes to evaluate")
    parser.add_argument("--no_diff", action="store_true", help="Skip diffusion and evaluate baselines only")
    args = parser.parse_args()

    run_benchmark(max_test_vols=args.vols, use_diffusion=not args.no_diff)
