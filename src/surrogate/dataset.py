"""Dataset utilities for surrogate modeling."""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np


@dataclass
class TransitionDataset:
    """Storage for (s, a, ds) transitions with normalization."""

    states: np.ndarray
    actions: np.ndarray
    deltas: np.ndarray
    s_mean: np.ndarray
    s_std: np.ndarray
    a_mean: np.ndarray
    a_std: np.ndarray
    d_mean: np.ndarray
    d_std: np.ndarray

    def normalize_sa(self, states: np.ndarray, actions: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Normalize state and action arrays."""

        s_norm = (states - self.s_mean) / (self.s_std + 1e-6)
        a_norm = (actions - self.a_mean) / (self.a_std + 1e-6)
        return s_norm, a_norm

    def denormalize_delta(self, delta_norm: np.ndarray) -> np.ndarray:
        """Denormalize delta predictions."""

        return delta_norm * (self.d_std + 1e-6) + self.d_mean


def build_dataset(transitions: List[Tuple[np.ndarray, np.ndarray, np.ndarray]]) -> TransitionDataset:
    """Build a normalized dataset from transitions."""

    states = np.stack([t[0] for t in transitions])
    actions = np.stack([t[1] for t in transitions])
    deltas = np.stack([t[2] for t in transitions])
    s_mean = states.mean(axis=0)
    s_std = states.std(axis=0)
    a_mean = actions.mean(axis=0)
    a_std = actions.std(axis=0)
    d_mean = deltas.mean(axis=0)
    d_std = deltas.std(axis=0)
    return TransitionDataset(states, actions, deltas, s_mean, s_std, a_mean, a_std, d_mean, d_std)
