"""Constraint helpers and state parsing."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np


@dataclass
class StateIndex:
    """Indices for observation components."""

    SOC_pack: int = 0
    V_cell_max: int = 1
    V_cell_min: int = 2
    dV: int = 3
    T_cell_max: int = 4
    T_cell_min: int = 5
    dT: int = 6
    std_V: int = 7
    std_T: int = 8
    std_SOC: int = 9
    SOH_pack: int = 10
    I_prev: int = 11



def state_to_info(state: np.ndarray, t: float, i: float) -> Dict[str, float]:
    """Convert a state vector into an info-like dict."""

    idx = StateIndex()
    return {
        "t": t,
        "I": i,
        "SOC_pack": float(state[idx.SOC_pack]),
        "V_cell_max": float(state[idx.V_cell_max]),
        "V_cell_min": float(state[idx.V_cell_min]),
        "T_cell_max": float(state[idx.T_cell_max]),
        "T_cell_min": float(state[idx.T_cell_min]),
        "dV": float(state[idx.dV]),
        "dT": float(state[idx.dT]),
        "std_V": float(state[idx.std_V]),
        "std_T": float(state[idx.std_T]),
        "std_SOC": float(state[idx.std_SOC]),
        "SOH_pack": float(state[idx.SOH_pack]),
        "I_prev": float(state[idx.I_prev]),
    }


def is_violation(state: np.ndarray, v_max: float, t_max: float) -> bool:
    """Check whether constraints are violated."""

    idx = StateIndex()
    return state[idx.V_cell_max] > v_max or state[idx.T_cell_max] > t_max
