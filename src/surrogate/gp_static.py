"""Static surrogate model (G0) implemented with NN ensemble placeholder."""
from __future__ import annotations

from typing import Callable, List, Tuple

import numpy as np

from .dataset import TransitionDataset
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
       
        # 1. 获取归一化的预测值
        # 注意：底层模型返回的是 (D,) 的一维数组
        norm_mean, norm_std = self.model.predict(self.dataset, state, action)

        # 2. 反归一化： 将神经网络的输出转换回物理单位 (如 V, K, SOC)
        # 公式: Real = Norm * Std + Mean
        delta_phys = norm_mean * (self.dataset.d_std + 1e-6) + self.dataset.d_mean
        delta_std_phys = norm_std * (self.dataset.d_std + 1e-6)

        # 3. 【核心物理约束】: 强制温度不下降
        # 如果预测值 < 0，强制设为 0。
        # 这样 DDPG 就不会因为害怕降温（实际上不可能）而乱来了。
        t_idx = -2  # 温度是倒数第2个
        # 检查维度并应用约束
        if delta_phys.ndim == 1:
            # 单样本预测
            delta_phys[t_idx] = max(0.0, delta_phys[t_idx])
        else:
            # Batch 预测
            delta_phys[:, t_idx] = np.maximum(delta_phys[:, t_idx], 0.0)

        return delta_phys, delta_std_phys


    def rollout(self, state: np.ndarray, policy: Callable[[np.ndarray], np.ndarray], horizon: int) -> np.ndarray:
        """Rollout the surrogate model for a horizon."""
        traj = [state.astype(np.float32)]
        s = traj[0].copy()

        for _ in range(horizon):
            action = policy(s) # (1,)
            delta, _ = self.predict(s, action) # (6,)
            s_next = s.copy()
            s_next[:6] = s[:6] + delta                   # 只更新前 6 维
            s_next[6] = float(action[0])                   # Iprev_{t+1} = action（近似 I_true）
            traj.append(s_next)
            s = s_next  
        return np.stack(traj)
