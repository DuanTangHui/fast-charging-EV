"""Constraint helpers and state parsing."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np


@dataclass
class StateIndex:
    """Indices for observation components."""

    SOC_pack: int = 0
    std_SOC: int = 1
    V_cell_max: int = 2
    dV: int = 3
    T_cell_max: int = 4
    T_cell_min: int = 5
    I_prev: int = 6



def state_to_info(state: np.ndarray, t: float, i: float) -> Dict[str, float]:
    """Convert a state vector into an info-like dict."""

    idx = StateIndex()
    return {
        "t": t,
        "I": i,
        "SOC_pack": float(state[idx.SOC_pack]),
        "std_SOC": float(state[idx.std_SOC]),
        "V_cell_max": float(state[idx.V_cell_max]),
        "T_cell_max": float(state[idx.T_cell_max]),
        "T_cell_min": float(state[idx.T_cell_min]),
        "dV": float(state[idx.dV]),
        "I_prev": float(state[idx.I_prev]),
    }


def is_violation(state: np.ndarray, v_max: float, t_max: float) -> bool:
    """Check whether constraints are violated."""

    idx = StateIndex()
    v_soft = v_max - 0.03   # 30mV 裕度：4.17V
    t_soft = t_max - 2.0
    return state[idx.V_cell_max] >= v_soft or state[idx.T_cell_max] >= t_soft
