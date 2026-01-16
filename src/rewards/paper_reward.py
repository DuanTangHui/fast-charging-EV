"""Reward implementation aligned with paper equation (24)."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict


@dataclass
class PaperRewardConfig:
    """Reward weights for SOC, time, voltage, and temperature penalties."""

    w_soc: float = 10.0
    w_time: float = 0.01
    w_v: float = 2.0
    w_t: float = 1.0


def compute_paper_reward(
    soc_prev: float,
    soc_next: float,
    t_prev: float,
    t_next: float,
    v_max_next: float,
    t_max_next: float,
    v_limit: float,
    t_limit: float,
    config: PaperRewardConfig,
) -> float:
    """Compute paper-style reward using pack-level max values."""


    r_soc = config.w_soc * (soc_next - soc_prev)
    r_time = -config.w_time
    r_v = -config.w_v * max(0.0, v_max_next - v_limit)
    r_t = -config.w_t * max(0.0, t_max_next - t_limit) 
    return r_soc + r_time + r_v + r_t 


def reward_from_info(prev: Dict, next_info: Dict, config: PaperRewardConfig, v_limit: float, t_limit: float) -> float:
    """Convenience function to compute reward from info dicts."""

    r = compute_paper_reward(
        soc_prev=float(prev["SOC_pack"]),
        soc_next=float(next_info["SOC_pack"]),
        t_prev=float(prev["t"]),
        t_next=float(next_info["t"]),
        v_max_next=float(next_info["V_cell_max"]),
        t_max_next=float(next_info["T_cell_max"]),
        v_limit=v_limit,
        t_limit=t_limit,
        config=config,
    )

    if next_info.get("violation", False):
          r -= 50.0
    return r
