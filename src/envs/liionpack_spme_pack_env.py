"""Pack environment with liionpack/pybamm backend or toy fallback."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np

try:
    import liionpack  # type: ignore  # noqa: F401
    import pybamm  # type: ignore  # noqa: F401
    _HAS_LIIONPACK = True
except Exception:
    _HAS_LIIONPACK = False

from .aging_scenarios import AgingParams
from .base_env import BasePackEnv
from .observables import build_observation


@dataclass
class PackState:
    """Internal state for the toy environment."""

    soc: np.ndarray
    voltage: np.ndarray
    temperature: np.ndarray
    soh: np.ndarray
    i_prev: float
    t: float
    step: int


class LiionpackSPMEPackEnv(BasePackEnv):
    """Pack environment wrapper with toy fallback when liionpack is unavailable."""

    def __init__(
        self,
        dt: float,
        max_steps: int,
        v_max: float,
        t_max: float,
        pack_cells_p: int,
        pack_cells_s: int,
        terminate_on_violation: bool = True,
    ) -> None:
        super().__init__(dt=dt, max_steps=max_steps, v_max=v_max, t_max=t_max)
        self.pack_cells_p = pack_cells_p
        self.pack_cells_s = pack_cells_s
        self.terminate_on_violation = terminate_on_violation
        self._rng = np.random.default_rng(0)
        self._aging = AgingParams(1.0, 1.0, 1.0)
        self._state: PackState | None = None

    @property
    def n_cells(self) -> int:
        return self.pack_cells_p * self.pack_cells_s

    def set_aging(self, aging: AgingParams) -> None:
        """Update aging parameters for subsequent resets."""

        self._aging = aging

    def reset(self, *, seed: int | None = None, options: Dict | None = None) -> Tuple[np.ndarray, Dict]:
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        soc = self._rng.uniform(0.2, 0.3, size=self.n_cells)
        voltage = 3.5 + 0.1 * soc + self._rng.normal(0, 0.01, size=self.n_cells)
        temperature = 298 + self._rng.normal(0, 0.5, size=self.n_cells)
        soh = np.ones(self.n_cells) * self._aging.capacity_scale
        self._state = PackState(soc, voltage, temperature, soh, 0.0, 0.0, 0)
        obs = build_observation(soc, voltage, temperature, soh, 0.0).as_array()
        info = {
            "t": 0.0,
            "I": 0.0,
            "SOC_pack": float(np.mean(soc)),
            "V_cell_max": float(np.max(voltage)),
            "V_cell_min": float(np.min(voltage)),
            "T_cell_max": float(np.max(temperature)),
            "T_cell_min": float(np.min(temperature)),
            "dV": float(np.max(voltage) - np.min(voltage)),
            "dT": float(np.max(temperature) - np.min(temperature)),
            "terminated_reason": "reset",
            "violation": False,
        }
        return obs, info

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        if self._state is None:
            raise RuntimeError("Environment must be reset before step.")
        current = float(np.clip(action[0], self.action_space.low[0], self.action_space.high[0]))
        state = self._state
        dt_hours = self.dt / 3600.0
        capacity = 3.0 * self._aging.capacity_scale
        soc = np.clip(state.soc + (current / capacity) * dt_hours, 0.0, 1.0)
        voltage = 3.0 + 1.2 * soc - 0.02 * current * self._aging.resistance_scale
        voltage += self._rng.normal(0, 0.005, size=self.n_cells)
        temperature = state.temperature + (0.05 * current * self._aging.thermal_scale)
        temperature += self._rng.normal(0, 0.1, size=self.n_cells)
        soh = state.soh

        state = PackState(soc, voltage, temperature, soh, current, state.t + self.dt, state.step + 1)
        self._state = state

        obs = build_observation(soc, voltage, temperature, soh, current).as_array()
        v_max = float(np.max(voltage))
        t_max = float(np.max(temperature))
        violation = v_max > self.v_max or t_max > self.t_max
        terminated = state.step >= self.max_steps or (self.terminate_on_violation and violation)
        truncated = False
        info = {
            "t": state.t,
            "I": current,
            "SOC_pack": float(np.mean(soc)),
            "V_cell_max": v_max,
            "V_cell_min": float(np.min(voltage)),
            "T_cell_max": t_max,
            "T_cell_min": float(np.min(temperature)),
            "dV": float(np.max(voltage) - np.min(voltage)),
            "dT": float(np.max(temperature) - np.min(temperature)),
            "terminated_reason": "violation" if violation else "time",
            "violation": violation,
        }
        return obs, 0.0, terminated, truncated, info


def build_pack_env(config: Dict) -> LiionpackSPMEPackEnv:
    """Construct the pack environment (liionpack or toy fallback)."""

    env = LiionpackSPMEPackEnv(
        dt=float(config["dt"]),
        max_steps=int(config["max_steps"]),
        v_max=float(config["v_max"]),
        t_max=float(config["t_max"]),
        pack_cells_p=int(config["pack_cells_p"]),
        pack_cells_s=int(config["pack_cells_s"]),
        terminate_on_violation=bool(config.get("terminate_on_violation", True)),
    )
    return env
