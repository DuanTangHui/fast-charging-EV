"""Fast calibration with SOH prior regularization."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from .metrics import mse
from ..soh_prior.prior_regularizers import l2_prior


@dataclass
class CalibrationResult:
    """Outputs of fast calibration."""

    theta_hat: np.ndarray
    loss: float


def fast_calibrate(
    segments: np.ndarray,
    simulator: Callable[[np.ndarray], np.ndarray],
    theta_prior: np.ndarray,
    lam: float,
    steps: int = 50,
    lr: float = 0.1,
) -> CalibrationResult:
    """Fit theta with simple gradient-free optimization.

    Args:
        segments: Target observations (vector).
        simulator: Function mapping theta -> predicted vector.
        theta_prior: Prior parameter vector.
        lam: Prior penalty weight.
        steps: Number of optimization steps.
        lr: Learning rate for finite-difference updates.

    Returns:
        CalibrationResult containing theta_hat and loss.
    """

    theta = theta_prior.copy()
    eps = 1e-3
    for _ in range(steps):
        pred = simulator(theta)
        loss_val = mse(pred, segments) + l2_prior(theta, theta_prior, lam)
        grad = np.zeros_like(theta)
        for i in range(theta.size):
            theta_eps = theta.copy()
            theta_eps[i] += eps
            pred_eps = simulator(theta_eps)
            loss_eps = mse(pred_eps, segments) + l2_prior(theta_eps, theta_prior, lam)
            grad[i] = (loss_eps - loss_val) / eps
        theta -= lr * grad
    final_pred = simulator(theta)
    final_loss = mse(final_pred, segments) + l2_prior(theta, theta_prior, lam)
    return CalibrationResult(theta_hat=theta, loss=float(final_loss))
