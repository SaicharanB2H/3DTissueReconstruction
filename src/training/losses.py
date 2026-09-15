"""
Loss Functions for VAE Pretraining and Conditional Latent Diffusion Training.
"""
from typing import Dict, Tuple, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.vae import DiagonalGaussianDistribution


class VAELoss(nn.Module):
    """
    Combined Reconstruction and KL Regularization Loss for Medical AutoencoderKL.
    L = L1 + alpha * L2 + beta_kl * KL + gamma_grad * L_grad
    """

    def __init__(
        self,
        l1_weight: float = 1.0,
        l2_weight: float = 0.5,
        kl_weight: float = 1.0e-5,
        gradient_weight: float = 0.1,
    ):
        super().__init__()
        self.l1_weight = l1_weight
        self.l2_weight = l2_weight
        self.kl_weight = kl_weight
        self.gradient_weight = gradient_weight

    def _gradient_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Computes spatial gradient difference to encourage sharp structural edges."""
        dx_pred = pred[:, :, :, 1:] - pred[:, :, :, :-1]
        dx_gt = target[:, :, :, 1:] - target[:, :, :, :-1]
        dy_pred = pred[:, :, 1:, :] - pred[:, :, :-1, :]
        dy_gt = target[:, :, 1:, :] - target[:, :, :-1, :]

        loss_dx = F.l1_loss(dx_pred, dx_gt)
        loss_dy = F.l1_loss(dy_pred, dy_gt)
        return loss_dx + loss_dy

    def forward(
        self,
        recon: torch.Tensor,
        target: torch.Tensor,
        posterior: DiagonalGaussianDistribution,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        l1_loss = F.l1_loss(recon, target)
        l2_loss = F.mse_loss(recon, target)
        kl_loss = posterior.kl().mean()
        grad_loss = self._gradient_loss(recon, target)

        total_loss = (
            self.l1_weight * l1_loss
            + self.l2_weight * l2_loss
            + self.kl_weight * kl_loss
            + self.gradient_weight * grad_loss
        )

        metrics = {
            "loss_total": float(total_loss.item()),
            "loss_l1": float(l1_loss.item()),
            "loss_l2": float(l2_loss.item()),
            "loss_kl": float(kl_loss.item()),
            "loss_grad": float(grad_loss.item()),
        }

        return total_loss, metrics


class DiffusionNoiseLoss(nn.Module):
    """
    Noise Prediction Loss for Conditional Latent Diffusion Models.
    Computes MSE between true Gaussian noise \epsilon and predicted \hat{\epsilon}_\theta.
    """

    def __init__(self, loss_type: str = "mse"):
        super().__init__()
        self.loss_type = loss_type

    def forward(
        self,
        noise_pred: torch.Tensor,
        noise_target: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        if self.loss_type == "l1":
            loss = F.l1_loss(noise_pred, noise_target)
        elif self.loss_type == "huber":
            loss = F.smooth_l1_loss(noise_pred, noise_target)
        else:
            loss = F.mse_loss(noise_pred, noise_target)

        metrics = {
            "diffusion_loss": float(loss.item()),
        }
        return loss, metrics
