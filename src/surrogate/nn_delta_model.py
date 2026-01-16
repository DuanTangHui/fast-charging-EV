"""Neural network delta model with ensemble uncertainty."""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import torch
from torch import nn

from .dataset import TransitionDataset
from src.surrogate import dataset
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class MLP(nn.Module):
    """Simple MLP for delta prediction."""

    def __init__(self, input_dim: int, output_dim: int, hidden_sizes: List[int]) -> None:
        super().__init__()
        layers: List[nn.Module] = []
        last_dim = input_dim
        for size in hidden_sizes:
            layers.append(nn.Linear(last_dim, size))
            layers.append(nn.ReLU())
            last_dim = size
        layers.append(nn.Linear(last_dim, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


@dataclass
class EnsembleConfig:
    """Configuration for ensemble model."""

    hidden_sizes: List[int]
    ensemble_size: int
    learning_rate: float

def print_array_stats(name: str, arr: np.ndarray) -> None:
    total = arr.size
    nan_count = np.isnan(arr).sum()
    posinf_count = np.isposinf(arr).sum()
    neginf_count = np.isneginf(arr).sum()

    print(f"\n===== Stats for {name} =====")
    print(f"Shape: {arr.shape}")
    print(f"NaN ratio: {nan_count / total:.6f}  ({nan_count}/{total})")
    print(f"+Inf ratio: {posinf_count / total:.6f}")
    print(f"-Inf ratio: {neginf_count / total:.6f}")

    finite = arr[np.isfinite(arr)]
    if finite.size > 0:
        print(f"Finite min/max: {finite.min():.6f} / {finite.max():.6f}")
        print(f"Finite mean/std: {finite.mean():.6f} / {finite.std():.6f}")
    else:
        print("No finite values!")
class EnsembleDeltaModel:
    """Ensemble MLP for delta prediction with uncertainty."""

    def __init__(self, input_dim: int, output_dim: int, config: EnsembleConfig) -> None:
        self.models = [MLP(input_dim, output_dim, config.hidden_sizes).to(DEVICE) for _ in range(config.ensemble_size)]
        self.optimizers = [torch.optim.Adam(model.parameters(), lr=config.learning_rate) for model in self.models]
        self.loss_fn = nn.MSELoss()

    def fit(self, dataset: TransitionDataset, epochs: int = 10) -> None:
        """Fit the ensemble on normalized data."""
        print("Using device:", DEVICE)

        s_norm, a_norm = dataset.normalize_sa(dataset.states, dataset.actions)
        
        x = np.concatenate([s_norm, a_norm], axis=-1)
       
        # y = (dataset.deltas - dataset.d_mean) / (dataset.d_std + 1e-6)
        y = (dataset.deltas_fast - dataset.d_mean_fast) / (dataset.d_std_fast + 1e-6)

        # # ========== 新增：数据质量诊断 ==========
        # print("deltas_fast shape:", dataset.deltas_fast.shape)   # (N, 11)
        # print("d_std_fast min:", dataset.d_std_fast.min())

        # print_array_stats("x (normalized s+a)", x)
        # print_array_stats("y (normalized delta)", y)

        # # 额外：打印每个维度的统计信息
        # print("\nPer-dimension stats for y:")
        # for i in range(y.shape[1]):
        #     col = y[:, i]
        #     finite = col[np.isfinite(col)]
        #     if finite.size > 0:
        #         print(
        #             f"dim {i:02d}: "
        #             f"mean={finite.mean():.4f}, "
        #             f"std={finite.std():.4f}, "
        #             f"min={finite.min():.4f}, "
        #             f"max={finite.max():.4f}"
        #         )
        # 如果数据里出现 NaN/Inf，直接替换掉，避免训练炸掉
        x = np.nan_to_num(x, nan=0.0, posinf=1e6, neginf=-1e6)
        y = np.nan_to_num(y, nan=0.0, posinf=1e6, neginf=-1e6)
        x_tensor = torch.tensor(x, dtype=torch.float32, device=DEVICE)
        y_tensor = torch.tensor(y, dtype=torch.float32, device=DEVICE)
        for _ in range(epochs):
            for model, opt in zip(self.models, self.optimizers):
                opt.zero_grad()
                pred = model(x_tensor)
                loss = self.loss_fn(pred, y_tensor)
                loss.backward()
                opt.step()

    def predict(self, dataset: TransitionDataset, state: np.ndarray, action: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Predict delta mean and std using the ensemble."""

        s_norm, a_norm = dataset.normalize_sa(state[None, :], action[None, :])
        x = np.concatenate([s_norm, a_norm], axis=-1)
        x_tensor = torch.tensor(x, dtype=torch.float32, device=DEVICE)
        preds = []
        for model in self.models:
            with torch.no_grad():
                preds.append(model(x_tensor).detach().cpu().numpy()[0])
        preds_np = np.stack(preds)
        mean = preds_np.mean(axis=0)
        std = preds_np.std(axis=0)
        # Sanitize outputs to avoid NaN/Inf getting propagated into the simulator.
        mean = np.nan_to_num(mean, nan=0.0, posinf=1e6, neginf=-1e6)
        std = np.nan_to_num(std, nan=1e-6, posinf=1e6, neginf=-1e6)
        std[std < 1e-6] = 1e-6
        return mean, std
