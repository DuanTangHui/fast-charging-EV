"""Base environment interfaces for pack charging."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import gymnasium as gym
import numpy as np


@dataclass
class PackObservation:
    """Pack-level observation container."""

    SOC_pack: float
    V_cell_max: float
    V_cell_min: float
    dV: float
    T_cell_max: float
    T_cell_min: float
    dT: float
    std_V: float
    std_T: float
    std_SOC: float
    SOH_pack: float
    I_prev: float

    def as_array(self) -> np.ndarray:
        """Return observation as numpy array."""

        return np.array(
            [
                self.SOC_pack,
                self.V_cell_max,
                self.V_cell_min,
                self.dV,
                self.T_cell_max,
                self.T_cell_min,
                self.dT,
                self.std_V,
                self.std_T,
                self.std_SOC,
                self.SOH_pack,
                self.I_prev,
            ],
            dtype=np.float32,
        )


class BasePackEnv(gym.Env):
    """Base pack charging environment."""

    def __init__(self, dt: float, max_steps: int, v_max: float, t_max: float) -> None:
        self.dt = dt
        self.max_steps = max_steps
        self.v_max = v_max
        self.t_max = t_max
        self.action_space = gym.spaces.Box(
            low=np.array([0.0], dtype=np.float32),
            high=np.array([120.0], dtype=np.float32),
        )
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(12,), dtype=np.float32
        )

    def reset(self, *, seed: int | None = None, options: Dict | None = None) -> Tuple[np.ndarray, Dict]:
        raise NotImplementedError

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        raise NotImplementedError
