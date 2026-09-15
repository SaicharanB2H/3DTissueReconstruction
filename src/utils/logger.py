"""
Structured Logging & TensorBoard Tracking Utility.
"""
import os
import sys
import logging
from typing import Dict, Any, Optional
try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    SummaryWriter = None


def setup_logger(name: str, log_dir: Optional[str] = None) -> logging.Logger:
    """Configures a clean stream and file logger."""
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.propagate = False

    if not logger.handlers:
        formatter = logging.Formatter(
            "[%(asctime)s] [%(name)s] [%(levelname)s]: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

        # File handler
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
            file_handler = logging.FileHandler(os.path.join(log_dir, "run.log"))
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)

    return logger


class MetricTracker:
    """Tracks running averages and logs scalars to TensorBoard."""

    def __init__(self, log_dir: Optional[str] = None):
        self.writer = SummaryWriter(log_dir) if log_dir else None
        self.metrics: Dict[str, float] = {}
        self.counts: Dict[str, int] = {}

    def update(self, name: str, value: float, n: int = 1):
        if name not in self.metrics:
            self.metrics[name] = 0.0
            self.counts[name] = 0
        self.metrics[name] += value * n
        self.counts[name] += n

    def get_avg(self, name: str) -> float:
        if name not in self.metrics or self.counts[name] == 0:
            return 0.0
        return self.metrics[name] / self.counts[name]

    def reset(self):
        self.metrics.clear()
        self.counts.clear()

    def log_to_tensorboard(self, step: int, prefix: str = ""):
        if self.writer is None:
            return
        for name in self.metrics:
            tag = f"{prefix}/{name}" if prefix else name
            self.writer.add_scalar(tag, self.get_avg(name), step)

    def close(self):
        if self.writer:
            self.writer.close()
