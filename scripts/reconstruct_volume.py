"""
Reconstruction CLI tool for single 3D medical volume.
"""
import os
import sys
import argparse
import yaml
import torch
import numpy as np
import h5py
import nibabel as nib

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data.dataset import BraTSVolumeDataset, get_patient_splits
from src.data.sparsifier import VolumeSparsifier
from src.models.slice_encoder import SharedSliceEncoder
from src.models.vae import MedicalAutoencoderKL
from src.models.unet_diffusion import ConditionalUNet2D
from src.models.scheduler import DiffusionNoiseScheduler
from src.models.pipeline import SparseToDenseDiffusionPipeline
from src.evaluation.metrics import evaluate_volume_reconstruction
from src.utils.logger import setup_logger


def reconstruct_single_volume(
    volume_id: int,
    stride: int = 4,
    diffusion_ckpt: str = "./experiments/checkpoints/diffusion/best_model.pt",
    vae_ckpt: str = "./experiments/checkpoints/vae/best_model.pt",
    output_dir: str = "./results/reconstructed_volumes",
    save_format: str = "nifti", # "nifti" or "h5"
):
    with open("configs/default_config.yaml", "r") as f:
        default_cfg = yaml.safe_load(f)
    with open("configs/diffusion_config.yaml", "r") as f:
        diff_cfg = yaml.safe_load(f)
    with open("configs/vae_config.yaml", "r") as f:
        vae_cfg = yaml.safe_load(f)

    device = default_cfg["project"]["device"] if torch.cuda.is_available() else "cpu"
    logger = setup_logger("ReconstructCLI")
    os.makedirs(output_dir, exist_ok=True)

    data_dir = default_cfg["data"]["data_dir"]
    modality = default_cfg["data"]["modality"]
    in_channels = default_cfg["data"]["in_channels"]

    # 1. Load volume
    dataset = BraTSVolumeDataset(data_dir=data_dir, volume_ids=[volume_id], modality=modality)
    item = dataset[0]
    gt_vol = item["volume"].to(device) # (D, C, H, W)
    total_slices = dataset.total_slices

    # 2. Sparsify
    sparsifier = VolumeSparsifier(stride=stride, total_slices=total_slices)
    sparse_vol, mask, acquired, missing = sparsifier.sparsify_volume(gt_vol, stride=stride)
    logger.info(f"Volume {volume_id} sparsified with stride {stride}x: {len(acquired)} acquired, {len(missing)} missing slices.")

    # 3. Build Pipeline
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

    if os.path.exists(diffusion_ckpt):
        ckpt = torch.load(diffusion_ckpt, map_location=device)
        if "slice_encoder" in ckpt["models"]:
            slice_encoder.load_state_dict(ckpt["models"]["slice_encoder"])
        if "unet" in ckpt["models"]:
            unet.load_state_dict(ckpt["models"]["unet"])

    if os.path.exists(vae_ckpt):
        vae_state = torch.load(vae_ckpt, map_location=device)
        vae.load_state_dict(vae_state["models"]["vae"] if "models" in vae_state else vae_state)

    pipeline = SparseToDenseDiffusionPipeline(slice_encoder, vae, unet, scheduler)
    pipeline.eval()

    # 4. Reconstruct
    logger.info("Running reverse latent diffusion reconstruction...")
    with torch.no_grad():
        recon_vol = pipeline.reconstruct_volume(
            sparse_vol,
            acquired_indices=acquired,
            stride=stride,
            num_inference_steps=diff_cfg["diffusion"]["ddim_sampling_steps"],
        )

    # 5. Evaluate
    metrics = evaluate_volume_reconstruction(recon_vol, gt_vol, missing_indices=missing)
    logger.info(f"Reconstruction Quality: PSNR={metrics['psnr_full']:.2f}dB | SSIM={metrics['ssim_full']:.4f} | Missing PSNR={metrics.get('psnr_missing', 0):.2f}dB")

    # 6. Save volume
    recon_np = recon_vol.cpu().numpy() # (D, C, H, W)
    if recon_np.shape[1] == 1:
        recon_np = recon_np[:, 0, :, :] # (D, H, W)

    if save_format == "nifti":
        # Save as NIfTI volume (.nii.gz)
        # Transpose (D, H, W) -> (H, W, D) for standard medical viewer compatibility
        nii_arr = np.transpose(recon_np, (1, 2, 0))
        nii_img = nib.Nifti1Image(nii_arr, affine=np.eye(4))
        save_path = os.path.join(output_dir, f"recon_vol_{volume_id}_stride_{stride}x.nii.gz")
        nib.save(nii_img, save_path)
    else:
        save_path = os.path.join(output_dir, f"recon_vol_{volume_id}_stride_{stride}x.h5")
        with h5py.File(save_path, "w") as f:
            f.create_dataset("reconstructed", data=recon_np)

    logger.info(f"Saved reconstructed 3D volume to {save_path}")
    return save_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--vol_id", type=int, default=100, help="BraTS Volume ID to reconstruct")
    parser.add_argument("--stride", type=int, default=4, help="Sparsification stride (e.g. 2, 4, 8, 16)")
    parser.add_argument("--format", type=str, default="nifti", choices=["nifti", "h5"])
    args = parser.parse_args()

    reconstruct_single_volume(volume_id=args.vol_id, stride=args.stride, save_format=args.format)
