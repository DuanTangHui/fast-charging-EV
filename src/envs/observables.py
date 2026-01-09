"""Observable extraction helpers for pack simulations."""
from __future__ import annotations

from typing import Dict, Sequence

import numpy as np

from .base_env import PackObservation


def build_observation(
    soc: np.ndarray,
    voltage: np.ndarray,
    temperature: np.ndarray,
    soh: np.ndarray,
    i_prev: float,
) -> PackObservation:
    """Aggregate cell-level arrays into a pack observation."""

    soc_pack = float(np.mean(soc))
    v_max = float(np.max(voltage))
    v_min = float(np.min(voltage))
    t_max = float(np.max(temperature))
    t_min = float(np.min(temperature))
    return PackObservation(
        SOC_pack=soc_pack,
        V_cell_max=v_max,
        V_cell_min=v_min,
        dV=v_max - v_min,
        T_cell_max=t_max,
        T_cell_min=t_min,
        dT=t_max - t_min,
        std_V=float(np.std(voltage)),
        std_T=float(np.std(temperature)),
        std_SOC=float(np.std(soc)),
        SOH_pack=float(np.mean(soh)),
        I_prev=i_prev,
    )


def curve_from_infos(infos: Sequence[Dict]) -> Dict[str, list[float]]:
    """Convert info dictionaries into plotting curves."""

    curves: Dict[str, list[float]] = {
        "t": [],
        "I": [],
        "SOC_pack": [],
        "V_cell_max": [],
        "V_cell_min": [],
        "T_cell_max": [],
        "T_cell_min": [],
        "dV": [],
        "dT": [],
        "reward": [],
    }
    for info in infos:
        for key in curves:
            curves[key].append(float(info.get(key, 0.0)))
    return curves
