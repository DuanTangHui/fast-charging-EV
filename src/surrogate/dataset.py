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
        states = np.asarray(states, dtype=np.float32)
        actions = actions.astype(np.float32)
        # 用高精度算统计量
        s_mean = states.mean(axis=0, dtype=np.float64)
        s_var  = states.var(axis=0, dtype=np.float64)
        s_std  = np.sqrt(s_var)

        # 给 std 一个“物理合理下限”
        STD_FLOOR = 1e-3    # 对你这个量级非常合适
        s_std = np.maximum(s_std, STD_FLOOR)

        self.s_mean = s_mean.astype(np.float32)
        self.s_std  = s_std.astype(np.float32)

        s_norm = (states - self.s_mean) / self.s_std 
        a_norm = (actions - self.a_mean) / self.a_std
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
    # Make normalization robust: replace nan/inf and enforce a minimum std to avoid
    # extremely large normalized values (which can lead to overflow in training).
    s_std = np.nan_to_num(s_std, nan=1.0, posinf=1.0, neginf=1.0)
    a_std = np.nan_to_num(a_std, nan=1.0, posinf=1.0, neginf=1.0)
    d_std = np.nan_to_num(d_std, nan=1.0, posinf=1.0, neginf=1.0)
    min_std = 1e-6
    s_std[s_std < min_std] = min_std
    a_std[a_std < min_std] = min_std
    d_std[d_std < min_std] = min_std
    return TransitionDataset(states, actions, deltas, s_mean, s_std, a_mean, a_std, d_mean, d_std)
