"""Neural network delta model with ensemble uncertainty."""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import torch
from torch import nn

from .dataset import TransitionDataset


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


class EnsembleDeltaModel:
    """Ensemble MLP for delta prediction with uncertainty."""

    def __init__(self, input_dim: int, output_dim: int, config: EnsembleConfig) -> None:
        self.models = [MLP(input_dim, output_dim, config.hidden_sizes) for _ in range(config.ensemble_size)]
        self.optimizers = [torch.optim.Adam(model.parameters(), lr=config.learning_rate) for model in self.models]
        self.loss_fn = nn.MSELoss()
        self.config = config

    def fit(self, dataset: TransitionDataset, epochs: int = 10) -> None:
        """Fit the ensemble on normalized data."""

        s_norm, a_norm = dataset.normalize_sa(dataset.states, dataset.actions)
        x = np.concatenate([s_norm, a_norm], axis=-1)
        y = (dataset.deltas - dataset.d_mean) / (dataset.d_std + 1e-6)
        x_tensor = torch.tensor(x, dtype=torch.float32)
        y_tensor = torch.tensor(y, dtype=torch.float32)
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
        x_tensor = torch.tensor(x, dtype=torch.float32)
        preds = []
        for model in self.models:
            with torch.no_grad():
                preds.append(model(x_tensor).numpy()[0])
        preds_np = np.stack(preds)
        mean = preds_np.mean(axis=0)
        std = preds_np.std(axis=0)
        return mean, std

    def state_dicts(self) -> List[dict]:
        """Return state dicts for ensemble members."""

        return [model.state_dict() for model in self.models]

    def load_state_dicts(self, state_dicts: List[dict]) -> None:
        """Load state dicts for ensemble members."""

        for model, state in zip(self.models, state_dicts):
            model.load_state_dict(state)
