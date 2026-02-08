"""Reward implementation aligned with updated charging objectives."""
from __future__ import annotations
import numpy as np

from dataclasses import dataclass
from logging import config
from typing import Dict, Tuple


@dataclass
class PaperRewardConfig:
    """ reward 权重"""

    w_soc: float = 50.0     # 提高 SOC 奖励
    w_time: float = 0.003    # 降低时间惩罚
    w_v: float = 100.0      # 强电压惩罚
    w_t: float = 1.0        # 温度惩罚
    w_const: float = 20.0   # 一致性惩罚
    w_action: float = 0.02  # 【关键】动作(电流)惩罚，防止一直最大电流


def compute_paper_reward(
    soc_prev: float,
    soc_next: float,
    v_max_next: float,
    t_max_next: float,
    std_soc_next: float,
    action_current: float,  # 【关键】传入当前动作
    v_limit: float,
    t_limit: float,
    config: PaperRewardConfig,
) -> Tuple[float, float, float, float, float, float, float]:
    """
    核心奖励计算函数 (Core Reward Logic).
    Arguments:
        soc_prev: 上一时刻 SOC
        soc_next: 当前时刻 SOC
        v_max_next: 当前时刻最高单体电压
        t_max_next: 当前时刻最高单体温度
        std_soc_next: 当前时刻 SOC 标准差 (一致性)
        action_current: 当前执行的电流动作 (用于惩罚大电流)
        v_limit: 电压限制 (e.g. 4.2)
        t_limit: 温度限制 (e.g. 333.15)
        config: 权重配置
    """

    # 1. SOC 奖励 (限制为非负)
    delta_soc = max(0, soc_next - soc_prev)
    r_soc = config.w_soc * delta_soc
    
    # 2. 时间惩罚
    r_time = -config.w_time
    
    # 3. 电压惩罚 (软约束 + 硬约束)
    r_v = 0.0
    v_soft_limit = v_limit - 0.05  # e.g. 4.15V
    
    # A. 硬约束：越界直接重罚
    if v_max_next > v_limit:
        r_v -= config.w_v * (v_max_next - v_limit)
        r_v -= 10.0 # 额外触网费
        
    # B. 软约束：在 4.15V ~ 4.2V 之间，施加指数惩罚
    elif v_max_next > v_soft_limit:
        r_v -= 50.0 * ((v_max_next - v_soft_limit) ** 2)
    
    # 4. 温度约束
    r_t = -config.w_t * max(0.0, t_max_next - t_limit)

    # 5. SOC 一致性惩罚
    # r_const = -config.w_const * max(0.0, std_soc_next)
    r_const = -config.w_const * max(0.0, std_soc_next - 0.012)
    
    # 6. 动作惩罚 (Action Penalty)
    # 抑制无脑大电流，引导平滑充电
    soc = soc_next  # 或 soc_prev

    # 0~1 的门控：soc<0.85 几乎不惩罚；soc>0.95 惩罚拉满
    gate = np.clip((soc - 0.85) / (0.95 - 0.85), 0.0, 1.0)

    # action_current ∈ [-30, 0]
    r_action = -(config.w_action * (1.0 + 9.0 * gate)) * (action_current / 30.0) ** 2

    # r_action = -config.w_action * (action_current / 20.0) ** 2

    # r_action = -config.w_action * (action_current ** 2)

    # 或者至少打印：
    # print("[ACTPEN] w_action=", config.w_action, "a=", action_current, "term=", (action_current/20.0)**2)

    return r_soc + r_time + r_v + r_t + r_const + r_action,r_soc ,r_time ,r_v , r_t , r_const , r_action


def reward_from_info(prev: Dict, next_info: Dict, config: PaperRewardConfig, v_limit: float, t_limit: float) -> float:
    """方便函数：从 info 字典计算奖励"""

    # 尝试获取电流，如果没有则为0
    current_i = float(next_info.get("I", next_info.get("I_prev", 0.0)))
    
    # 尝试获取 SOC std
    std_soc = float(next_info.get("std_SOC", next_info.get("SOC_std", 0.0)))

    r,r_soc ,r_time ,r_v , r_t , r_const , r_action = compute_paper_reward(
        soc_prev=float(prev["SOC_pack"]),
        soc_next=float(next_info["SOC_pack"]),
        v_max_next=float(next_info["V_cell_max"]),
        t_max_next=float(next_info["T_cell_max"]),
        std_soc_next=std_soc,
        action_current=current_i,
        v_limit=v_limit,
        t_limit=t_limit,
        config=config,
    )
    
    return r,r_soc ,r_time ,r_v , r_t , r_const , r_action