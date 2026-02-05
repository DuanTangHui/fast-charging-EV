"""Reward implementation aligned with updated charging objectives."""
from __future__ import annotations
import numpy as np

from dataclasses import dataclass
from logging import config
from typing import Dict, Tuple


@dataclass
class PaperRewardConfig:
    """ reward 权重"""

    w_soc: float = 500.0     # 提高 SOC 奖励

    w_time: float = 0.1   # 降低时间惩罚

    w_track: float = 0.5      # 追债系数：离 1.0 越远，扣分越狠
    w_finish: float = 50.0    # 完赛奖：充满电给一大笔钱


    w_v: float = 200.0      # 强电压惩罚
    w_t: float = 2.0        # 温度惩罚
    w_const: float = 1.0   # 一致性惩罚
    w_action: float = 0.02  # 动作(电流)惩罚，防止一直最大电流
    w_v_soft: float = 50.0  # 软电压惩罚 
    w_discharge: float = 10.0 # 放电惩罚：禁止在此任务中放电


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
) -> Tuple[float, Dict[str, float]]:
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
    delta_soc = soc_next - soc_prev
    # r_soc = config.w_soc * delta_soc
    r_soc = config.w_soc * delta_soc * (1 + 2.0 * soc_next)


    # 时间惩罚
    r_time = -config.w_time 

    # 2. 目标距离惩罚 (Gravity)
    # 只要没到 1.0，就一直扣分。
    # SOC=0.6 时： -2.0 * (1.0 - 0.6) = -0.8 (每步扣 0.8)
    # SOC=0.9 时： -2.0 * (1.0 - 0.9) = -0.2 (亏损减少，鼓励它坚持到底)
    r_track = config.w_track * (soc_next - soc_prev)

    # 3. 完赛奖励 (Terminal Reward) - 你的代码外部可能有，但这里加个保险
    # 引导它跨过最后 1%
    r_finish = 0.0
    if soc_next > 0.8:
        r_finish += 0.2 * (soc_next - 0.8)

    
    # 3. 电压惩罚 (软约束 + 硬约束)
    r_v = 0.0
    v_soft_limit =  min(v_limit - 0.05, 4.05) # 4.15V
    
    # A. 硬约束：越界直接重罚
    if v_max_next > v_limit:
        r_v -= config.w_v * (v_max_next - v_limit)
        r_v -= 10.0 # 额外触网费
        
    # B. 软约束：在 4.15V ~ 4.2V 之间，施加指数惩罚
    elif v_max_next > v_soft_limit:
        x = (v_max_next - v_soft_limit) / (v_limit - v_soft_limit)  # 0~1
        r_v -= config.w_v_soft * (x ** 2)   # 4次方让靠近上限才猛扣
    
    # 4. 温度约束
    r_t = -config.w_t * max(0.0, t_max_next - t_limit)

    # 5. SOC 一致性惩罚
    # r_const = -config.w_const * max(0.0, std_soc_next)
    r_const = -config.w_const * max(0.0, std_soc_next - 0.012)
    
    # 6. 动作惩罚 (Action Penalty)
    
    soc = soc_next  # 或 soc_prev

    # 0~1 的门控：soc<0.85 几乎不惩罚；soc>0.95 惩罚拉满
    gate = np.clip((soc - 0.85) / (0.95 - 0.85), 0.0, 1.0)

    # action_current ∈ [-20, 0]
    r_action = -(config.w_action * (1.0 + 9.0 * gate)) * (action_current / 20.0) ** 2

    # 7. 放电惩罚
    # 如果 action_current > 0 (表示放电)，给予重罚
    if action_current > 0.0:
        r_action -= config.w_discharge * action_current

   
    total = r_soc + r_track + r_finish + r_v + r_t + r_const + r_action + r_time
    terms = {
        "total": float(total),
        "r_soc": float(r_soc),
        "delta_soc": float(delta_soc),
        "r_track": float(r_track),
        "r_finish": float(r_finish),
        "r_v": float(r_v),
        "r_t": float(r_t),
        "r_const": float(r_const),
        "r_action": float(r_action),
        "r_time": float(r_time),
    }

    return total,terms

def compute_minimal_reward(soc_prev, soc_next, v_max, t_max, v_limit, t_limit):
    # 权重配置 (参考论文 6.4.1 节)
    w_charge = 20.0
    w_time = 0.01
    w_penal = -10.0
    
    # 1. 进度奖：鼓励充入更多 SOC
    # 这样设计的好处是：无论步长多大，充满电获得的总分是一定的
    r_charging = w_charge * (soc_next - soc_prev) / (1.0 - 0.2)
    
    # 2. 时间成本：每走一步扣 0.01 分
    # Agent 为了让扣分变少，会自动寻找大电流方案
    r_time = -w_time
    
    # 3. 安全惩罚：只针对越界行为（硬红线）
    # 这种“触网才罚”的方式能避免 Agent 在安全区内缩手缩脚
    r_penal = 0.0
    if v_max > v_limit or t_max > t_limit:
        r_penal = w_penal
        
    total = r_charging + r_time + r_penal
    return total

def reward_from_info(prev: Dict, next_info: Dict, config: PaperRewardConfig, v_limit: float, t_limit: float) -> float:
    """方便函数：从 info 字典计算奖励"""

    # 尝试获取电流，如果没有则为0
    current_i = float(next_info.get("I", next_info.get("I_prev", 0.0)))
    
    # 尝试获取 SOC std
    std_soc = float(next_info.get("std_SOC", next_info.get("SOC_std", 0.0)))

    total, terms= compute_paper_reward(
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
    
    return total