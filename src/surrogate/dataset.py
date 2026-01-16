"""Dataset utilities for surrogate modeling."""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
SOH_INDEX = 10  # SOH_pack 在 obs 中的索引
def fast_idxs(obs_dim: int) -> np.ndarray:
    return np.array([i for i in range(obs_dim) if i != SOH_INDEX], dtype=int)

@dataclass
class TransitionDataset:
    """Storage for (s, a, ds) transitions with normalization."""

    states: np.ndarray
    actions: np.ndarray
    deltas: np.ndarray          # 保留原始 12维（可选，SOH不变的）
    deltas_fast: np.ndarray     # 新增：11维
    s_mean: np.ndarray
    s_std: np.ndarray
    a_mean: np.ndarray
    a_std: np.ndarray
    d_mean: np.ndarray
    d_std: np.ndarray
    d_mean_fast: np.ndarray     # 新增：11维
    d_std_fast: np.ndarray      # 新增：11维

    def normalize_sa(self, states: np.ndarray, actions: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Normalize state and action arrays."""
        states = np.asarray(states, dtype=np.float32)
        actions = actions.astype(np.float32)

        s_std = np.maximum(self.s_std.astype(np.float32), 1e-6)
        a_std = np.maximum(self.a_std.astype(np.float32), 1e-6)

        s_norm = (states - self.s_mean.astype(np.float32)) / s_std
        # ---- HOTFIX: temperature dims are (almost) constant in current dataset ----
        # build_observation: dim4=T_cell_max, dim5=T_cell_min
        # if self.s_std[4] < 0.01:
        #     s_norm[:,4] = 0
        # if self.s_std[5] < 0.01:
        #     s_norm[:,5] = 0
        a_norm = (actions - self.a_mean.astype(np.float32)) / a_std
        return s_norm, a_norm

    def denormalize_delta(self, delta_norm: np.ndarray) -> np.ndarray:
        """Denormalize delta predictions."""

        return delta_norm * (self.d_std + 1e-6) + self.d_mean
    
    def denormalize_delta_fast(self, delta_norm_fast: np.ndarray) -> np.ndarray:
        """Denormalize FAST delta predictions (excluding SOH dim)."""
        return delta_norm_fast * (self.d_std_fast + 1e-6) + self.d_mean_fast

def build_dataset(transitions: List[Tuple[np.ndarray, np.ndarray, np.ndarray]]) -> TransitionDataset:
    """Build a normalized dataset from transitions."""

    states = np.stack([t[0] for t in transitions])
    actions = np.stack([t[1] for t in transitions])
    deltas = np.stack([t[2] for t in transitions])
    obs_dim = states.shape[1]
    idxs = fast_idxs(obs_dim)

    # fast deltas: remove SOH dim
    deltas_fast = deltas[:, idxs]
    s_mean = states.mean(axis=0)
    s_std = states.std(axis=0)
    a_mean = actions.mean(axis=0)
    a_std = actions.std(axis=0)
    d_mean = deltas.mean(axis=0)
    d_std = deltas.std(axis=0)
    
    d_mean_fast = deltas_fast.mean(axis=0)
    d_std_fast = deltas_fast.std(axis=0)
    
    # Make normalization robust: replace nan/inf and enforce a minimum std to avoid
    # extremely large normalized values (which can lead to overflow in training).
    s_std = np.nan_to_num(s_std, nan=1.0, posinf=1.0, neginf=1.0)
    a_std = np.nan_to_num(a_std, nan=1.0, posinf=1.0, neginf=1.0)
    d_std = np.nan_to_num(d_std, nan=1.0, posinf=1.0, neginf=1.0)
    d_std_fast = np.nan_to_num(d_std_fast, nan=1.0, posinf=1.0, neginf=1.0)
    
    min_std_s = 1e-3   # 给 state 用更高下限
    min_std_a = 1e-6
    min_std_d = 1e-6

    s_std[s_std < min_std_s] = min_std_s
    a_std[a_std < min_std_a] = min_std_a
    d_std[d_std < min_std_d] = min_std_d
    d_std_fast[d_std_fast < min_std_d] = min_std_d

    return TransitionDataset(
        states=states,
        actions=actions,
        deltas=deltas,
        deltas_fast=deltas_fast,
        s_mean=s_mean,
        s_std=s_std,
        a_mean=a_mean,
        a_std=a_std,
        d_mean=d_mean,
        d_std=d_std,
        d_mean_fast=d_mean_fast,
        d_std_fast=d_std_fast,)
