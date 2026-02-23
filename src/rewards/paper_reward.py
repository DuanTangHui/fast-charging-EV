"""Reward implementation aligned with updated charging objectives."""
from __future__ import annotations
import numpy as np

from dataclasses import dataclass
from logging import config
from typing import Dict, Tuple


@dataclass
class PaperRewardConfig:
    """ reward 权重"""

    # w_soc: float = 80.0     # 提高 SOC 奖励
    # w_time: float = 0.03    # 降低时间惩罚
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
    dt: float = 10.0 # 你的观测步长
) -> Tuple[float, ...]:
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
    # 1. SOC 进度奖励 (论文系数为 10) [cite: 346]
    r_soc = 50.0 * (soc_next - soc_prev) 
    
    # 2. 时间惩罚 (关键改进点)
    # 论文公式: -0.01 * (t_next - t_prev) [cite: 346]
    r_time = -0.05
    
    # 3. 电压约束 (硬约束优化) [cite: 347]
    r_v = 0.0
    v_warning = v_limit - 0.03
    
    if v_max_next > v_warning:
        # 进入黄灯区，开始施加温和的“制动力” (线性平滑惩罚)
        # 比如电压达到 4.18V 时，惩罚是 -50.0 * 0.01 = -0.5 分
        # 这会稍微抵消一点 SOC 的收益，提示 Agent 开始减小电流
        r_v = -50.0 * (v_max_next - v_warning)
        
    if v_max_next > v_limit:
        r_v -= 500.0 * (v_max_next - v_limit) 
      
    
    # 4. 温度约束 [cite: 348]
    # 论文逻辑：不越界为 0，越界则扣分 (系数为 -1)
    r_t = 0.0
    if t_max_next > t_limit:
        r_t = -1.0 * (t_max_next - t_limit)

    # 5. 一致性惩罚 (保留你的原始设计，但调低权重)
    # 因为 1000s 的快充必然会牺牲一部分一致性
    r_const = -5.0 * max(0.0, std_soc_next - 0.012)

    total_reward = r_soc + r_time + r_v + r_t + r_const


    return total_reward, r_soc, r_time, r_v, r_t, r_const, 0.0
    

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
    
    return r