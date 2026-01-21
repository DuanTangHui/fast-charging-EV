"""Reward implementation aligned with updated charging objectives."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict


@dataclass
class PaperRewardConfig:
    """Reward weights for SOC gain, time, voltage, temperature, and consistency penalties."""

    w_soc: float = 10.0
    w_time: float = 0.05
    w_v: float = 100.0
    w_t: float = 1.0
    w_const: float = 20.0


def compute_paper_reward(
    soc_prev: float,
    soc_next: float,
    v_max_next: float,
    t_max_next: float,
    std_soc_next: float,
    v_limit: float,
    t_limit: float,
    soc_std_next: float,
    config: PaperRewardConfig,
) -> float:
    """Compute reward using pack-level max values and SOC consistency."""


   # 1. SOC 奖励
    r_soc = config.w_soc * (soc_next - soc_prev)
    
    # 2. 时间惩罚
    r_time = -config.w_time
    
    # 3. 电压惩罚 (强约束)
    r_v = -config.w_v * max(0.0, v_max_next - v_limit)
    
    # 4. 温度惩罚 (强约束)
    r_t = -config.w_t * max(0.0, t_max_next - t_limit)

    # 5. SOC 一致性惩罚
    r_const = -config.w_const * max(0.0, std_soc_next)
    return r_soc + r_time + r_v + r_t + r_const


def reward_from_info(prev: Dict, next_info: Dict, config: PaperRewardConfig, v_limit: float, t_limit: float) -> float:
    """Convenience function to compute reward from info dicts."""

    r = compute_paper_reward(
        soc_prev=float(prev["SOC_pack"]),
        soc_next=float(next_info["SOC_pack"]),
        v_max_next=float(next_info["V_cell_max"]),
        t_max_next=float(next_info["T_cell_max"]),
        std_soc_next=float(next_info.get("std_SOC", 0.0)),
        v_limit=v_limit,
        t_limit=t_limit,
        soc_std_next=float(next_info.get("SOC_std", 0.0)),
        config=config,
    )
    
    return r