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
        s_mean = self.s_mean.astype(np.float64)
        a_mean = self.a_mean.astype(np.float64)
        s_std = np.clip(self.s_std.astype(np.float64), 1e-4, None)
        a_std = np.clip(self.a_std.astype(np.float64), 1e-4, None)
        s_norm = (states.astype(np.float64) - s_mean) / s_std
        a_norm = (actions.astype(np.float64) - a_mean) / a_std
        return np.clip(s_norm, -1e6, 1e6), np.clip(a_norm, -1e6, 1e6)

    def denormalize_delta(self, delta_norm: np.ndarray) -> np.ndarray:
        """Denormalize delta predictions."""
        d_std = np.clip(self.d_std.astype(np.float64), 1e-4, None)
        d_mean = self.d_mean.astype(np.float64)
        delta = delta_norm.astype(np.float64) * d_std + d_mean
        return np.clip(delta, -1e6, 1e6)


def build_dataset(transitions: List[Tuple[np.ndarray, np.ndarray, np.ndarray]]) -> TransitionDataset:
    """Build a normalized dataset from transitions."""
    states = np.stack([t[0] for t in transitions]).astype(np.float64)
    actions = np.stack([t[1] for t in transitions]).astype(np.float64)
    deltas = np.stack([t[2] for t in transitions]).astype(np.float64)
    s_mean = states.mean(axis=0)
    s_std = np.clip(states.std(axis=0), 1e-4, None)
    a_mean = actions.mean(axis=0)
    a_std = np.clip(actions.std(axis=0), 1e-4, None)
    d_mean = deltas.mean(axis=0)
    d_std = np.clip(deltas.std(axis=0), 1e-4, None)
    return TransitionDataset(states, actions, deltas, s_mean, s_std, a_mean, a_std, d_mean, d_std)
