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
    # 预测
    def predict(self, state: np.ndarray, action: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Predict combined delta mean and std."""
        # 1. 获取基础模型的预测
        delta_static, std_static = self.static.predict(state, action)
        # 2. 如果没有差分模型（比如刚开始训练），直接返回基础预测
        if self.differential is None:
            return delta_static, std_static
        # 3. 获取差分模型的预测（即：修正量）
        delta_diff, std_diff = self.differential.predict(state, action)
        return delta_static + delta_diff, std_static + std_diff
    # 虚拟推演
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
