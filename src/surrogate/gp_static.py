"""Static surrogate model (G0) implemented with NN ensemble placeholder."""
from __future__ import annotations

from typing import Callable, List, Tuple

import numpy as np

from .dataset import TransitionDataset, fast_idxs, SOH_INDEX
from .nn_delta_model import EnsembleConfig, EnsembleDeltaModel


class StaticSurrogate:
    """Static surrogate that predicts delta state."""

    def __init__(self, input_dim: int, output_dim: int, hidden_sizes: List[int], ensemble_size: int, lr: float) -> None:
        self.model = EnsembleDeltaModel(
            input_dim=input_dim,
            output_dim=output_dim,
            config=EnsembleConfig(hidden_sizes, ensemble_size, lr),
        )
        self.dataset: TransitionDataset | None = None

    def fit(self, dataset: TransitionDataset, epochs: int = 10) -> None:
        """Fit the static surrogate."""

        self.dataset = dataset
        self.model.fit(dataset, epochs=epochs)

    def predict(self, state: np.ndarray, action: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Predict delta state mean and std."""

        if self.dataset is None:
            raise RuntimeError("Static surrogate has not been fit.")
        # 1. 模型输出的是归一化的值（比如 0 到 1 之间）
        # mean, std = self.model.predict(self.dataset, state, action)
        # # 2. 反归一化：将神经网络的输出转换回物理单位（如 V, K, SOC）
        # delta = self.dataset.denormalize_delta(mean)
        # # 3. 标准差也需要反归一化，注意这里的 scale 操作
        # delta_std = std * (self.dataset.d_std + 1e-6)
        # return delta, delta_std
        mean_fast, std_fast = self.model.predict(self.dataset, state, action)

        delta_fast = self.dataset.denormalize_delta_fast(mean_fast)
        delta_fast_std = std_fast * (self.dataset.d_std_fast + 1e-6)

        obs_dim = state.shape[0]
        idxs = fast_idxs(obs_dim)

        delta = np.zeros(obs_dim, dtype=float)
        delta_std = np.zeros(obs_dim, dtype=float)

        delta[idxs] = delta_fast
        delta_std[idxs] = delta_fast_std

        # SOH delta = 0（不变）
        delta[SOH_INDEX] = 0.0
        delta_std[SOH_INDEX] = 1e-6

        return delta, delta_std


    def rollout(self, state: np.ndarray, policy: Callable[[np.ndarray], np.ndarray], horizon: int) -> np.ndarray:
        """Rollout the surrogate model for a horizon."""
        traj = [state]
        s = state # 当前状态
        for _ in range(horizon):
            action = policy(s)
            delta, _ = self.predict(s, action)
            s = s + delta # 下一个状态
            traj.append(s)
        return np.stack(traj)
