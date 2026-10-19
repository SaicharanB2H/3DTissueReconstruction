"""
Diffusion Noise Scheduler & DDIM / DDPM Sampler.
Supports Cosine and Linear beta schedules with fast accelerated DDIM sampling.
"""
import math
from typing import Tuple, Optional, List
import torch
import torch.nn as nn


def cosine_beta_schedule(timesteps: int, s: float = 0.008) -> torch.Tensor:
    """
    Cosine schedule as proposed in Nichol & Dhariwal 2021.
    """
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps, dtype=torch.float32)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0.0001, 0.9999)


def linear_beta_schedule(timesteps: int, beta_start: float = 0.0001, beta_end: float = 0.02) -> torch.Tensor:
    """
    Standard linear schedule from Ho et al. 2020.
    """
    return torch.linspace(beta_start, beta_end, timesteps, dtype=torch.float32)


class DiffusionNoiseScheduler(nn.Module):
    """
    Manages forward noise addition q(z_t | z_0) and reverse DDIM/DDPM sampling.
    """

    def __init__(
        self,
        num_train_timesteps: int = 1000,
        beta_schedule: str = "cosine",
        beta_start: float = 0.0001,
        beta_end: float = 0.02,
    ):
        super().__init__()
        if not isinstance(num_train_timesteps, int) or num_train_timesteps < 2:
            raise ValueError("At least two training timesteps are required")
        if beta_schedule not in ('cosine', 'linear') or not 0 < beta_start < beta_end < 1:
            raise ValueError("Invalid noise schedule")
        self.num_train_timesteps = num_train_timesteps

        if beta_schedule == "cosine":
            betas = cosine_beta_schedule(num_train_timesteps)
        else:
            betas = linear_beta_schedule(num_train_timesteps, beta_start, beta_end)

        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = torch.cat([torch.tensor([1.0]), alphas_cumprod[:-1]])

        # Register buffers
        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        self.register_buffer("alphas_cumprod_prev", alphas_cumprod_prev)

        # Calculations for diffusion q(x_t | x_0) and posterior
        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        self.register_buffer("sqrt_one_minus_alphas_cumprod", torch.sqrt(1.0 - alphas_cumprod))
        self.register_buffer("log_one_minus_alphas_cumprod", torch.log(1.0 - alphas_cumprod))
        self.register_buffer("sqrt_recip_alphas_cumprod", torch.sqrt(1.0 / alphas_cumprod))
        self.register_buffer("sqrt_recipm1_alphas_cumprod", torch.sqrt(1.0 / alphas_cumprod - 1))

        # Posterior variance
        posterior_variance = betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        self.register_buffer("posterior_variance", posterior_variance)

    def add_noise(
        self,
        original_samples: torch.Tensor,
        noise: torch.Tensor,
        timesteps: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward diffusion process: q(x_t | x_0) = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * noise
        """
        # Ensure timesteps is on the same device and buffer is indexed properly
        sqrt_alpha_prod = self.sqrt_alphas_cumprod[timesteps].view(-1, 1, 1, 1)
        sqrt_one_minus_alpha_prod = self.sqrt_one_minus_alphas_cumprod[timesteps].view(-1, 1, 1, 1)

        return sqrt_alpha_prod * original_samples + sqrt_one_minus_alpha_prod * noise

    def ddim_step(
        self,
        model_output: torch.Tensor,
        timestep: int,
        prev_timestep: int,
        sample: torch.Tensor,
        eta: float = 0.0,
        generator: Optional[torch.Generator] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Computes single reverse DDIM step from t to prev_t.
        Returns: (prev_sample, pred_original_sample)
        """
        alpha_prod_t = self.alphas_cumprod[timestep]
        alpha_prod_t_prev = self.alphas_cumprod[prev_timestep] if prev_timestep >= 0 else torch.tensor(1.0, device=sample.device)

        beta_prod_t = 1.0 - alpha_prod_t
        beta_prod_t_prev = 1.0 - alpha_prod_t_prev

        # 1. Predict original sample x_0 from noise prediction epsilon
        pred_original_sample = (sample - torch.sqrt(beta_prod_t) * model_output) / torch.sqrt(alpha_prod_t)

        # 2. Compute variance sigma_t
        variance = (beta_prod_t_prev / beta_prod_t) * (1.0 - alpha_prod_t / alpha_prod_t_prev)
        std_dev_t = eta * torch.sqrt(torch.clamp(variance, min=0.0))

        # 3. Compute direction pointing to x_t
        pred_sample_direction = torch.sqrt(torch.clamp(1.0 - alpha_prod_t_prev - std_dev_t ** 2, min=0.0)) * model_output

        # 4. Compute x_{t-1}
        prev_sample = torch.sqrt(alpha_prod_t_prev) * pred_original_sample + pred_sample_direction

        if eta > 0.0:
            noise = torch.randn(sample.shape, device=sample.device, dtype=sample.dtype, generator=generator)
            prev_sample = prev_sample + std_dev_t * noise

        return prev_sample, pred_original_sample

    def get_ddim_timesteps(self, num_inference_steps: int = 25) -> List[int]:
        """
        Generates evenly spaced timesteps for fast DDIM sampling (e.g. 25 steps).
        """
        if not isinstance(num_inference_steps, int) or not 1 <= num_inference_steps <= self.num_train_timesteps:
            raise ValueError("Inference steps must be between 1 and training timesteps")
        step_ratio = self.num_train_timesteps // num_inference_steps
        timesteps = (torch.arange(0, num_inference_steps) * step_ratio).round().long().flip(0)
        return timesteps.tolist()
