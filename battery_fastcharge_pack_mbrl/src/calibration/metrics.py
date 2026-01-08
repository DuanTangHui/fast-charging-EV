"""Calibration metrics utilities."""
from __future__ import annotations

import numpy as np


def mse(pred: np.ndarray, target: np.ndarray) -> float:
    """Mean squared error."""

    return float(np.mean((pred - target) ** 2))
