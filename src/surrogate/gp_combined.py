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
        # 统一口径：所有 surrogate 都输出 6 维 delta（不含 I_prev）
        delta = delta_static + delta_diff
        std = np.sqrt(std_static**2 + std_diff**2)
        return delta, std
    # 虚拟推演
    def rollout(self, state: np.ndarray, policy: Callable[[np.ndarray], np.ndarray], horizon: int) -> np.ndarray:
        """Rollout the combined surrogate with project-consistent state update.
        State is 7D; delta is 6D (no I_prev). next_state[:6]+=delta; next_state[-1]=action.
        """
        traj = [state.copy()]
        current = state.copy()
        for _ in range(horizon):
            action = policy(current)
            delta6, _ = self.predict(current, action)

            next_state = current.copy()
            next_state[:6] = current[:6] + delta6
            next_state[-1] = float(np.asarray(action).reshape(-1)[0])  # I_prev = action
            current = next_state
            traj.append(current.copy())
        return np.stack(traj)

