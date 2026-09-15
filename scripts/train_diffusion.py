"""
Training Script for Conditional Latent Diffusion Model (Sparse-to-Dense 3D Reconstruction).
"""
import os
import sys
import argparse
import yaml
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.amp import GradScaler, autocast
from tqdm import tqdm

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data.dataset import BraTSSliceDataset, get_patient_splits
from src.models.slice_encoder import SharedSliceEncoder
from src.models.vae import MedicalAutoencoderKL
from src.models.unet_diffusion import ConditionalUNet2D
from src.models.scheduler import DiffusionNoiseScheduler
from src.models.pipeline import SparseToDenseDiffusionPipeline
from src.training.losses import DiffusionNoiseLoss
from src.utils.logger import setup_logger, MetricTracker
from src.utils.checkpoint import CheckpointManager, EMAModel
from src.evaluation.metrics import calculate_psnr, calculate_ssim


def train_diffusion(
    config_path: str = "configs/diffusion_config.yaml",
    default_config_path: str = "configs/default_config.yaml",
    vae_config_path: str = "configs/vae_config.yaml",
):
    with open(config_path, "r") as f:
        diff_cfg = yaml.safe_load(f)
    with open(default_config_path, "r") as f:
        default_cfg = yaml.safe_load(f)
    with open(vae_config_path, "r") as f:
        vae_cfg = yaml.safe_load(f)

    device = default_cfg["project"]["device"] if torch.cuda.is_available() else "cpu"
    logger = setup_logger("TrainDiffusion", default_cfg["logging"]["tensorboard_dir"])
    logger.info(f"Training on device: {device} ({torch.cuda.get_device_name(0) if device == 'cuda' else 'CPU'})")

    # 1. Dataset & DataLoaders
    data_dir = default_cfg["data"]["data_dir"]
    modality = default_cfg["data"]["modality"]
    train_vols, val_vols, _ = get_patient_splits(
        data_dir,
        metadata_file=default_cfg["data"].get("metadata_file"),
        train_ratio=default_cfg["data"]["split_ratios"]["train"],
        val_ratio=default_cfg["data"]["split_ratios"]["val"],
        test_ratio=default_cfg["data"]["split_ratios"]["test"],
        seed=default_cfg["project"]["seed"],
    )

    logger.info(f"Patient splits: {len(train_vols)} train vols, {len(val_vols)} val vols. Primary modality: {modality.upper()}")

    train_dataset = BraTSSliceDataset(
        data_dir=data_dir,
        volume_ids=train_vols,
        modality=modality,
        stride=default_cfg["sparsification"]["default_stride"],
        is_training=True,
        augment=True,
    )
    val_dataset = BraTSSliceDataset(
        data_dir=data_dir,
        volume_ids=val_vols[:10], # Fast validation subset
        modality=modality,
        stride=default_cfg["sparsification"]["default_stride"],
        is_training=False,
        augment=False,
    )

    batch_size = diff_cfg["training"]["batch_size"]
    num_workers = default_cfg["data"].get("num_workers", 2)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=(num_workers > 0),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=(num_workers > 0),
    )

    # 2. Models
    in_channels = default_cfg["data"]["in_channels"]

    # VAE (Frozen or Pre-trained)
    vae = MedicalAutoencoderKL(
        in_channels=in_channels,
        out_channels=in_channels,
        latent_channels=vae_cfg["model"]["latent_channels"],
        base_channels=vae_cfg["model"]["base_channels"],
        channel_multipliers=vae_cfg["model"]["channel_multipliers"],
        num_res_blocks=vae_cfg["model"]["num_res_blocks"],
    ).to(device)

    # Check if pre-trained VAE exists, otherwise train jointly or use fresh weights
    vae_ckpt_path = diff_cfg["training"]["vae_checkpoint"]
    if os.path.exists(vae_ckpt_path):
        logger.info(f"Loading pre-trained VAE from {vae_ckpt_path}")
        vae_state = torch.load(vae_ckpt_path, map_location=device)
        vae.load_state_dict(vae_state["models"]["vae"] if "models" in vae_state else vae_state)
    else:
        logger.info("No pre-trained VAE found. Using initialized VAE.")

    vae.eval()
    for p in vae.parameters():
        p.requires_grad = False

    # Shared Contextual Slice Encoder
    slice_encoder = SharedSliceEncoder(
        in_channels=in_channels,
        base_channels=32,
        context_dim=diff_cfg["unet"]["context_dim"],
        out_spatial_channels=8,
    ).to(device)

    # Conditional Denoising U-Net
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

    # 3. Optimizer & EMA
    trainable_params = list(slice_encoder.parameters()) + list(unet.parameters())
    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=diff_cfg["training"]["lr"],
        weight_decay=diff_cfg["training"]["weight_decay"],
    )

    epochs = diff_cfg["training"]["epochs"]
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=diff_cfg["training"]["min_lr"]
    )

    ema_unet = EMAModel(unet, decay=diff_cfg["training"]["ema_decay"])
    scaler = GradScaler('cuda', enabled=diff_cfg["training"]["mixed_precision"])
    criterion = DiffusionNoiseLoss(loss_type="mse")
    checkpoint_mgr = CheckpointManager(diff_cfg["training"]["checkpoint_dir"])
    tracker = MetricTracker(default_cfg["logging"]["tensorboard_dir"])

    best_val_loss = float("inf")
    start_epoch = 1

    # Automatically resume from latest checkpoint if available
    ckpt = checkpoint_mgr.load_latest_or_none(
        models={"slice_encoder": slice_encoder, "unet": unet},
        optimizers={"optimizer": optimizer},
        schedulers={"lr_scheduler": lr_scheduler},
        scalers={"scaler": scaler},
        device=device,
    )
    if ckpt is not None:
        start_epoch = ckpt["epoch"] + 1
        best_val_loss = ckpt.get("metric_value", float("inf"))
        if "ema" in ckpt:
            ema_unet.load_state_dict(ckpt["ema"])
        logger.info(
            f"--> Resumed diffusion training from epoch {ckpt['epoch']}. "
            f"Continuing from epoch {start_epoch}/{epochs} (Best Val Loss: {best_val_loss:.4f})"
        )

    if start_epoch > epochs:
        logger.info(f"Diffusion training already completed ({start_epoch - 1}/{epochs} epochs).")
        return

    logger.info(f"Starting Conditional Latent Diffusion Training from epoch {start_epoch} to {epochs}...")
    for epoch in range(start_epoch, epochs + 1):
        slice_encoder.train()
        unet.train()
        tracker.reset()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs} [Train]")

        for step, batch in enumerate(pbar):
            target_slice = batch["target"].to(device) # (B, C, H, W)
            prev_slice = batch["prev"].to(device)
            next_slice = batch["next"].to(device)
            norm_z = batch["normalized_z"].to(device)
            alpha = batch["alpha"].to(device)
            gap_dist = batch["gap_dist"].to(device)
            B = target_slice.size(0)

            optimizer.zero_grad()

            with autocast('cuda', enabled=diff_cfg["training"]["mixed_precision"]):
                # 1. Encode clean target slice to latent space z_0
                with torch.no_grad():
                    dist = vae.encode(target_slice)
                    z_0 = dist.sample() * vae.scale_factor # (B, 4, 30, 30)

                # 2. Sample random timesteps t and noise epsilon
                timesteps = torch.randint(0, scheduler.num_train_timesteps, (B,), device=device).long()
                noise = torch.randn_like(z_0)

                # 3. Add noise: z_t = sqrt(alpha_bar_t) * z_0 + sqrt(1 - alpha_bar_t) * noise
                z_t = scheduler.add_noise(z_0, noise, timesteps)

                # 4. Extract conditioning features from Context Slice Encoder
                context_tokens, spatial_context = slice_encoder(
                    prev_slice, next_slice, norm_z, alpha, gap_dist
                )

                # 5. Predict noise with Conditional Denoising U-Net
                noise_pred = unet(z_t, timesteps, context_tokens, spatial_context)
                loss, metrics = criterion(noise_pred, noise)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(trainable_params, diff_cfg["training"]["grad_clip_norm"])
            scaler.step(optimizer)
            scaler.update()

            ema_unet.update(unet)

            for k, v in metrics.items():
                tracker.update(k, v, n=B)

            pbar.set_postfix({"Noise Loss": f"{tracker.get_avg('diffusion_loss'):.4f}"})

        lr_scheduler.step()
        train_loss = tracker.get_avg("diffusion_loss")
        tracker.log_to_tensorboard(epoch, prefix="diff_train")

        # --- Validation ---
        slice_encoder.eval()
        unet.eval()
        val_tracker = MetricTracker()

        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Epoch {epoch}/{epochs} [Val]"):
                target_slice = batch["target"].to(device)
                prev_slice = batch["prev"].to(device)
                next_slice = batch["next"].to(device)
                norm_z = batch["normalized_z"].to(device)
                alpha = batch["alpha"].to(device)
                gap_dist = batch["gap_dist"].to(device)
                B = target_slice.size(0)

                with autocast('cuda', enabled=diff_cfg["training"]["mixed_precision"]):
                    dist = vae.encode(target_slice)
                    z_0 = dist.sample() * vae.scale_factor
                    timesteps = torch.randint(0, scheduler.num_train_timesteps, (B,), device=device).long()
                    noise = torch.randn_like(z_0)
                    z_t = scheduler.add_noise(z_0, noise, timesteps)

                    context_tokens, spatial_context = slice_encoder(
                        prev_slice, next_slice, norm_z, alpha, gap_dist
                    )
                    noise_pred = unet(z_t, timesteps, context_tokens, spatial_context)
                    loss, metrics = criterion(noise_pred, noise)

                for k, v in metrics.items():
                    val_tracker.update(k, v, n=B)

        val_loss = val_tracker.get_avg("diffusion_loss")
        val_tracker.log_to_tensorboard(epoch, prefix="diff_val")

        logger.info(f"Epoch {epoch:03d}/{epochs:03d} | Train Noise Loss: {train_loss:.4f} | Val Noise Loss: {val_loss:.4f}")

        is_best = val_loss < best_val_loss
        if is_best:
            best_val_loss = val_loss
            logger.info(f"--> Saved new best diffusion model (Val Loss: {best_val_loss:.4f})")

        checkpoint_mgr.save(
            epoch=epoch,
            models={
                "slice_encoder": slice_encoder,
                "unet": unet,
            },
            optimizers={"optimizer": optimizer},
            schedulers={"lr_scheduler": lr_scheduler},
            scalers={"scaler": scaler},
            extra_state={"ema": ema_unet.state_dict()},
            metric_value=val_loss,
            is_best=is_best,
        )
        logger.info(f"--> Checkpoint saved for epoch {epoch} (checkpoint_epoch_{epoch:03d}.pt & latest.pt)")

    logger.info("Conditional Latent Diffusion Training Completed.")


if __name__ == "__main__":
    train_diffusion()
