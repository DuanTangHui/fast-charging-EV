"""Combined surrogate that sums static and differential predictions."""
from __future__ import annotations

from typing import Callable, Tuple

import numpy as np

from .gp_static import StaticSurrogate
from .gp_differential import DifferentialSurrogate


class CombinedSurrogate:
    """Combined surrogate with mu = mu_G0 + mu_G^."""

    def __init__(self, static: StaticSurrogate, differential: DifferentialSurrogate | None = None) -> None:
        self.static = static
        self.differential = differential

    def predict(self, state: np.ndarray, action: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Predict combined delta mean and std."""

        delta_static, std_static = self.static.predict(state, action)
        if self.differential is None:
            return delta_static, std_static
        delta_diff, std_diff = self.differential.predict(state, action)
        return delta_static + delta_diff, std_static + std_diff

    def rollout(self, state: np.ndarray, policy: Callable[[np.ndarray], np.ndarray], horizon: int) -> np.ndarray:
        """Rollout the combined surrogate."""

        traj = [state]
        current = state
        for _ in range(horizon):
            action = policy(current)
            delta, _ = self.predict(current, action)
            current = current + delta
            traj.append(current)
        return np.stack(traj)
