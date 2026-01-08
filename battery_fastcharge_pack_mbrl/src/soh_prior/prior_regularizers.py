"""Prior regularization utilities."""
from __future__ import annotations

import numpy as np


def l2_prior(theta: np.ndarray, theta_prior: np.ndarray, lam: float) -> float:
    """Compute L2 prior penalty."""

    return float(lam * np.sum((theta - theta_prior) ** 2))
