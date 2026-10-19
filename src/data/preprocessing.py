"""Shared preprocessing: statistics depend only on acquired intensities."""
import numpy as np
from src.data.transforms import MedicalIntensityNormalizer


def validate_acquired(indices, depth):
    values = list(indices)
    if not values or any(isinstance(i, bool) or not isinstance(i, (int, np.integer)) for i in values):
        raise ValueError("Acquired indices must be a nonempty list of integers")
    if len(set(values)) != len(values) or min(values) < 0 or max(values) >= depth:
        raise ValueError("Acquired indices must be unique and within volume depth")
    return sorted(values)


def normalize_acquired_volume(volume, acquired_indices):
    acquired = validate_acquired(acquired_indices, volume.shape[0])
    normalizer = MedicalIntensityNormalizer()
    _, stats = normalizer.normalize(volume[acquired])
    normalized, _ = normalizer.normalize(volume, stats=stats)
    return normalized, stats
