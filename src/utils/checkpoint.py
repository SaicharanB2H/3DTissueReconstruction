"""
Model Checkpointing & Exponential Moving Average (EMA) Utility.
"""
import os
from typing import Dict, Any, Optional
import torch
import torch.nn as nn


class EMAModel:
    """
    Maintains exponential moving average of model weights for smoother diffusion generation.
    """

    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.decay = decay
        self.shadow: Dict[str, torch.Tensor] = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self, model: nn.Module):
        for name, param in model.named_parameters():
            if param.requires_grad and name in self.shadow:
                self.shadow[name].mul_(self.decay).add_(param.data, alpha=1.0 - self.decay)

    def copy_to(self, model: nn.Module):
        for name, param in model.named_parameters():
            if param.requires_grad and name in self.shadow:
                param.data.copy_(self.shadow[name])

    def state_dict(self) -> Dict[str, torch.Tensor]:
        return self.shadow

    def load_state_dict(self, state_dict: Dict[str, torch.Tensor]):
        self.shadow = {k: v.clone() for k, v in state_dict.items()}


class CheckpointManager:
    """
    Saves and loads model weights with best-metric tracking and automatic latest-checkpoint resumption.
    """

    def __init__(self, checkpoint_dir: str, max_to_keep: int = 3):
        self.checkpoint_dir = checkpoint_dir
        self.max_to_keep = max_to_keep
        os.makedirs(checkpoint_dir, exist_ok=True)
        self.best_metric = float("inf")

    def save(
        self,
        epoch: int,
        models: Dict[str, nn.Module],
        optimizers: Dict[str, Any],
        metric_value: float,
        is_best: bool = False,
        schedulers: Optional[Dict[str, Any]] = None,
        scalers: Optional[Dict[str, Any]] = None,
        extra_state: Optional[Dict[str, Any]] = None,
        filename: Optional[str] = None,
    ):
        state = {
            "epoch": epoch,
            "metric_value": metric_value,
            "models": {k: m.state_dict() for k, m in models.items()},
            "optimizers": {k: opt.state_dict() for k, opt in optimizers.items()},
        }
        if is_best:
            self.best_metric = min(self.best_metric, metric_value)
        state["best_metric"] = self.best_metric
        if schedulers:
            state["schedulers"] = {k: s.state_dict() for k, s in schedulers.items()}
        if scalers:
            state["scalers"] = {k: sc.state_dict() for k, sc in scalers.items()}
        if extra_state:
            state.update(extra_state)

        if filename is None:
            filename = f"checkpoint_epoch_{epoch:03d}.pt"

        save_path = os.path.join(self.checkpoint_dir, filename)
        torch.save(state, save_path)

        # Always save as latest.pt so training can resume seamlessly if interrupted
        latest_path = os.path.join(self.checkpoint_dir, "latest.pt")
        torch.save(state, latest_path)

        if is_best:
            best_path = os.path.join(self.checkpoint_dir, "best_model.pt")
            torch.save(state, best_path)

    def load(
        self,
        checkpoint_path: str,
        models: Dict[str, nn.Module],
        optimizers: Optional[Dict[str, Any]] = None,
        schedulers: Optional[Dict[str, Any]] = None,
        scalers: Optional[Dict[str, Any]] = None,
        device: str = "cpu",
    ) -> Dict[str, Any]:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
        self.best_metric = checkpoint.get("best_metric", float("inf"))
        if self.best_metric == float("inf"):
            best_path = os.path.join(self.checkpoint_dir, "best_model.pt")
            if os.path.isfile(best_path):
                self.best_metric = torch.load(best_path, map_location="cpu", weights_only=True).get("metric_value", float("inf"))

        for name, model in models.items():
            if name in checkpoint["models"]:
                model.load_state_dict(checkpoint["models"][name])

        if optimizers is not None and "optimizers" in checkpoint:
            for name, opt in optimizers.items():
                if name in checkpoint["optimizers"]:
                    opt.load_state_dict(checkpoint["optimizers"][name])

        if schedulers is not None and "schedulers" in checkpoint:
            for name, sch in schedulers.items():
                if name in checkpoint["schedulers"]:
                    sch.load_state_dict(checkpoint["schedulers"][name])

        if scalers is not None and "scalers" in checkpoint:
            for name, sc in scalers.items():
                if name in checkpoint["scalers"]:
                    sc.load_state_dict(checkpoint["scalers"][name])

        return checkpoint

    def load_latest_or_none(
        self,
        models: Dict[str, nn.Module],
        optimizers: Optional[Dict[str, Any]] = None,
        schedulers: Optional[Dict[str, Any]] = None,
        scalers: Optional[Dict[str, Any]] = None,
        device: str = "cpu",
    ) -> Optional[Dict[str, Any]]:
        latest_path = os.path.join(self.checkpoint_dir, "latest.pt")
        if not os.path.exists(latest_path):
            return None
        return self.load(
            latest_path,
            models=models,
            optimizers=optimizers,
            schedulers=schedulers,
            scalers=scalers,
            device=device,
        )
