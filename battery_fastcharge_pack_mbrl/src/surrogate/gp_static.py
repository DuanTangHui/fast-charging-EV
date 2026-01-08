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
        mean, std = self.model.predict(self.dataset, state, action)
        delta = self.dataset.denormalize_delta(mean)
        delta_std = std * (self.dataset.d_std + 1e-6)
        return delta, delta_std

    def rollout(self, state: np.ndarray, policy: Callable[[np.ndarray], np.ndarray], horizon: int) -> np.ndarray:
        """Rollout the surrogate model for a horizon."""

        traj = [state]
        current = state
        for _ in range(horizon):
            action = policy(current)
            delta, _ = self.predict(current, action)
            current = current + delta
            traj.append(current)
        return np.stack(traj)
