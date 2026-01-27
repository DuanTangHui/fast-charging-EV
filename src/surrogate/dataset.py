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
        states = np.asarray(states, dtype=np.float32)
        actions = np.asarray(actions, dtype=np.float32)

        if states.shape[-1] != self.s_mean.shape[0]:
            raise ValueError(f"Unexpected state dim {states.shape[-1]} (expected {self.s_mean.shape[0]}).")

        s_std = np.maximum(self.s_std.astype(np.float32), 1e-6)
        a_std = np.maximum(self.a_std.astype(np.float32), 1e-6)

        s_norm = (states - self.s_mean.astype(np.float32)) / s_std
        a_norm = (actions - self.a_mean.astype(np.float32)) / a_std
        return s_norm, a_norm

    def denormalize_delta(self, delta_norm: np.ndarray) -> np.ndarray:
        """Denormalize delta predictions."""

        return delta_norm * (self.d_std + 1e-6) + self.d_mean

def build_dataset(transitions: List[Tuple[np.ndarray, np.ndarray, np.ndarray]]) -> TransitionDataset:
    """ 
    transitions: List[(state7, action1, delta6)]
    - states: (N,7)
    - actions: (N,1)
    - deltas: (N,6)  # 只预测前6维变化（不含 Iprev）
    """
    states = np.stack([t[0] for t in transitions]).astype(np.float32)
    actions = np.stack([t[1] for t in transitions]).astype(np.float32)
    deltas = np.stack([t[2] for t in transitions]).astype(np.float32)

    if states.shape[-1] != 7:
        raise ValueError(f"Expected state dim 7, got {states.shape[-1]}")
    if actions.shape[-1] != 1:
        raise ValueError(f"Expected action dim 1, got {actions.shape[-1]}")
    if deltas.shape[-1] != 6:
        raise ValueError(f"Expected delta dim 6, got {deltas.shape[-1]}")

    s_mean = states.mean(axis=0)
    s_std = states.std(axis=0)
    a_mean = actions.mean(axis=0)
    a_std = actions.std(axis=0)
    d_mean = deltas.mean(axis=0)
    d_std = deltas.std(axis=0)

    # 数值健壮性
    s_std = np.nan_to_num(s_std, nan=1.0, posinf=1.0, neginf=1.0)
    a_std = np.nan_to_num(a_std, nan=1.0, posinf=1.0, neginf=1.0)
    d_std = np.nan_to_num(d_std, nan=1.0, posinf=1.0, neginf=1.0)

    s_std[s_std < 1e-3] = 1e-3
    a_std[a_std < 1e-6] = 1e-6
    d_std[d_std < 1e-6] = 1e-6

    return TransitionDataset(
        states=states,
        actions=actions,
        deltas=deltas,
        s_mean=s_mean,
        s_std=s_std,
        a_mean=a_mean,
        a_std=a_std,
        d_mean=d_mean,
        d_std=d_std,
    )
