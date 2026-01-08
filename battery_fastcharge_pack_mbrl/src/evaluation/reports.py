"""Reporting utilities for rollouts."""
from __future__ import annotations

from typing import Dict, List

import numpy as np


def summarize_episode(infos: List[Dict]) -> Dict[str, float]:
    """Summarize episode metrics from info list."""

    soc_end = float(infos[-1]["SOC_pack"]) if infos else 0.0
    t_end = float(infos[-1]["t"]) if infos else 0.0
    v_max = max(float(info["V_cell_max"]) for info in infos) if infos else 0.0
    t_max = max(float(info["T_cell_max"]) for info in infos) if infos else 0.0
    reward_sum = float(np.sum([info.get("reward", 0.0) for info in infos]))
    return {
        "soc_end": soc_end,
        "t_end": t_end,
        "v_max": v_max,
        "t_max": t_max,
        "reward_sum": reward_sum,
    }
