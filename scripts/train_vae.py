"""
Training Script for Medical AutoencoderKL (VAE).
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

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data.dataset import BraTSSliceDataset, get_patient_splits
from src.models.vae import MedicalAutoencoderKL
from src.training.losses import VAELoss
from src.utils.logger import setup_logger, MetricTracker
from src.utils.checkpoint import CheckpointManager


def train_vae(config_path: str = "configs/vae_config.yaml", default_config_path: str = "configs/default_config.yaml"):
    with open(config_path, "r") as f:
        vae_cfg = yaml.safe_load(f)
    with open(default_config_path, "r") as f:
        default_cfg = yaml.safe_load(f)

    device = default_cfg["project"]["device"] if torch.cuda.is_available() else "cpu"
    logger = setup_logger("TrainVAE", default_cfg["logging"]["tensorboard_dir"])
    logger.info(f"Using device: {device} ({torch.cuda.get_device_name(0) if device == 'cuda' else 'CPU'})")

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

    logger.info(f"Loaded patient splits: {len(train_vols)} train volumes, {len(val_vols)} val volumes.")

    # Fast-path for VAE: only load target slice (return_context=False), disable heavy CPU augmentations
    train_dataset = BraTSSliceDataset(
        data_dir=data_dir,
        volume_ids=train_vols,
        modality=modality,
        stride=default_cfg["sparsification"]["default_stride"],
        is_training=True,
        augment=False,
        return_context=False,
    )
    val_dataset = BraTSSliceDataset(
        data_dir=data_dir,
        volume_ids=val_vols,
        modality=modality,
        stride=default_cfg["sparsification"]["default_stride"],
        is_training=False,
        augment=False,
        return_context=False,
    )

    batch_size = vae_cfg["training"]["batch_size"]
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

    # 2. Model, Loss, Optimizer
    in_channels = default_cfg["data"]["in_channels"]
    model = MedicalAutoencoderKL(
        in_channels=in_channels,
        out_channels=in_channels,
        latent_channels=vae_cfg["model"]["latent_channels"],
        base_channels=vae_cfg["model"]["base_channels"],
        channel_multipliers=vae_cfg["model"]["channel_multipliers"],
        num_res_blocks=vae_cfg["model"]["num_res_blocks"],
    ).to(device)

    criterion = VAELoss(
        l1_weight=vae_cfg["training"]["l1_weight"],
        kl_weight=vae_cfg["training"]["kl_weight"],
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=vae_cfg["training"]["lr"],
        weight_decay=vae_cfg["training"]["weight_decay"],
    )

    epochs = vae_cfg["training"]["epochs"]
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=vae_cfg["training"]["min_lr"]
    )

    scaler = GradScaler('cuda', enabled=vae_cfg["training"]["mixed_precision"])
    checkpoint_mgr = CheckpointManager(vae_cfg["training"]["checkpoint_dir"])
    tracker = MetricTracker(default_cfg["logging"]["tensorboard_dir"])

    best_val_loss = float("inf")
    start_epoch = 1

    # Automatically resume from latest checkpoint if available
    ckpt = checkpoint_mgr.load_latest_or_none(
        models={"vae": model},
        optimizers={"optimizer": optimizer},
        schedulers={"lr_scheduler": scheduler},
        scalers={"scaler": scaler},
        device=device,
    )
    if ckpt is not None:
        start_epoch = ckpt["epoch"] + 1
        best_val_loss = ckpt.get("metric_value", float("inf"))
        logger.info(
            f"--> Resumed from latest checkpoint: completed epoch {ckpt['epoch']}. "
            f"Continuing from epoch {start_epoch}/{epochs} (Best Val Loss: {best_val_loss:.4f})"
        )

    if start_epoch > epochs:
        logger.info(f"VAE Training already completed ({start_epoch - 1}/{epochs} epochs).")
        return

    logger.info(f"Starting VAE Training from epoch {start_epoch} to {epochs}...")
    for epoch in range(start_epoch, epochs + 1):
        # --- Training ---
        model.train()
        tracker.reset()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs} [Train]")

        for step, batch in enumerate(pbar):
            x = batch["target"].to(device) # (B, C, H, W)
            optimizer.zero_grad()

            with autocast('cuda', enabled=vae_cfg["training"]["mixed_precision"]):
                recon, posterior = model(x, sample_posterior=True)
                loss, metrics = criterion(recon, x, posterior)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), vae_cfg["training"]["grad_clip_norm"])
            scaler.step(optimizer)
            scaler.update()

            for k, v in metrics.items():
                tracker.update(k, v, n=x.size(0))

            pbar.set_postfix({"Loss": f"{tracker.get_avg('loss_total'):.4f}", "L1": f"{tracker.get_avg('loss_l1'):.4f}"})

        scheduler.step()
        train_loss = tracker.get_avg("loss_total")
        tracker.log_to_tensorboard(epoch, prefix="vae_train")

        # --- Validation ---
        model.eval()
        val_tracker = MetricTracker()
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Epoch {epoch}/{epochs} [Val]"):
                x = batch["target"].to(device)
                with autocast('cuda', enabled=vae_cfg["training"]["mixed_precision"]):
                    recon, posterior = model(x, sample_posterior=False)
                    loss, metrics = criterion(recon, x, posterior)
                for k, v in metrics.items():
                    val_tracker.update(k, v, n=x.size(0))

        val_loss = val_tracker.get_avg("loss_total")
        val_l1 = val_tracker.get_avg("loss_l1")
        val_tracker.log_to_tensorboard(epoch, prefix="vae_val")

        logger.info(f"Epoch {epoch:03d}/{epochs:03d} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val L1: {val_l1:.4f}")

        # Checkpoint saving after every epoch
        is_best = val_loss < best_val_loss
        if is_best:
            best_val_loss = val_loss
            logger.info(f"--> Saved new best VAE model with Val Loss: {best_val_loss:.4f}")

        checkpoint_mgr.save(
            epoch=epoch,
            models={"vae": model},
            optimizers={"optimizer": optimizer},
            schedulers={"lr_scheduler": scheduler},
            scalers={"scaler": scaler},
            metric_value=val_loss,
            is_best=is_best,
        )
        logger.info(f"--> Checkpoint saved for epoch {epoch} (checkpoint_epoch_{epoch:03d}.pt & latest.pt)")

    logger.info("VAE Training Completed.")


if __name__ == "__main__":
    train_vae()
