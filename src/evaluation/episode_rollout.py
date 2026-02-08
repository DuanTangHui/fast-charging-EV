"""Episode rollout utilities for environments and surrogates."""
from __future__ import annotations

from typing import Callable, Dict, List, Tuple

import numpy as np

from ..envs.base_env import BasePackEnv
from ..rewards.paper_reward import PaperRewardConfig, reward_from_info, compute_paper_reward


def rollout_env(
    env: BasePackEnv,
    policy: Callable[[np.ndarray], np.ndarray],
    reward_cfg: PaperRewardConfig,
) -> Tuple[float, List[Dict]]:
    """Rollout in a real environment."""
    
    state, info = env.reset()

    infos: List[Dict] = [info]
    total_reward = 0.0
    prev_info = info
    done = False
    
    while not done:
        action = policy(state)
        next_state, _, terminated, truncated, next_info = env.step(action)
        
        # 使用 reward_from_info 自动提取参数
        reward = reward_from_info(prev_info, next_info, reward_cfg, env.v_max, env.t_max)
        
        next_info["reward"] = reward
        infos.append(next_info)
        total_reward += reward
        state = next_state
        prev_info = next_info
        done = terminated or truncated

    print(
        "episode_end:",
        "steps=", len(infos)-1,
        "R=", round(total_reward, 2),
        "SOC_end=", round(infos[-1]["SOC_pack"], 4),
        "Vmax=", round(max(i["V_cell_max"] for i in infos), 4),
        "Tmax=", round(max(i["T_cell_max"] for i in infos), 2),
        "viol=", infos[-1].get("violation", None),
        "reason=", infos[-1].get("terminated_reason", None),
        "I_mean=", round(np.mean([i["I"] for i in infos[1:]]), 2) if len(infos) > 1 else None,
    )

    return total_reward, infos


def rollout_surrogate(
    state: np.ndarray,
    surrogate: Callable, 
    policy: Callable,
    horizon: int,
    reward_cfg: PaperRewardConfig,
    dt: float,
    v_max: float,
    t_max: float
) -> Tuple[float, List[Dict]]:
    """
    用静态代理模型进行 rollout：
    - 状态 7 维: [SOCmean, stdSOC, Vmax, dV, Tmax, Tmin, Iprev]
    - surrogate 预测 delta 6 维（不含 Iprev）
    - next_state[:6] = state[:6] + delta6
    - next_state[6] = action (近似 I_true)
    同时加入不确定性悲观惩罚（Hao et al. 风格）。
    """
    # 状态索引定义 
    IDX_SOC = 0
    IDX_STD_SOC = 1
    IDX_VMAX = 2
    IDX_DV = 3
    IDX_TMAX = 4
    IDX_TMIN = 5
    IDX_IPREV = 6
    # ---------- delta 索引----------
    D_VMAX = 2

    curr_state = state.copy()
    total_reward = 0.0
    infos = []
    
    # 初始化 info
    curr_info = {
        "t": 0.0,
        "SOC_pack": float(curr_state[IDX_SOC]),
        "std_SOC": float(curr_state[IDX_STD_SOC]),
        "V_cell_max": float(curr_state[IDX_VMAX]),
        "dV": float(curr_state[IDX_DV]),
        "T_cell_max": float(curr_state[IDX_TMAX]),
        "T_cell_min": float(curr_state[IDX_TMIN]),
        "I": float(curr_state[IDX_IPREV]),       # 当前步采用的电流（这里等于初始 Iprev）
        "I_prev": float(curr_state[IDX_IPREV]),  # 明确写出口径
        "reward": 0.0,
        "violation": False,
        "viol_hard": False,
        "is_risky": False,
        "terminated_reason": "reset",
    }
    BASE_TERMS = {
        "delta_soc": 0.0,
        "v_soft": False,
        "i_abs": 0.0,
        "r_soc": 0.0,
        "r_track": 0.0,
        "r_finish": 0.0,
        "r_v": 0.0,
        "r_t": 0.0,
        "r_const": 0.0,
        "r_action": 0.0,
        "r_time": 0.0,
    }
    curr_info.update(BASE_TERMS)
    infos.append(curr_info)

    for k in range(horizon):
        # 1. 策略动作
        action = policy(curr_state) 
        a_val = float(action[0])
        
        # 2. 预测 (Mean, Std)
        delta_mean, delta_std = surrogate(curr_state, action)
       
        # 3. 状态更新 只更新前6维；Iprev=action）
        next_state = curr_state.copy()
        next_state[:6] = curr_state[:6] + delta_mean
        next_state[IDX_IPREV] = a_val

        # 4. 安全检查 (Hao et al. 核心)
        v_mean = next_state[IDX_VMAX]
        v_std = delta_std[D_VMAX]
        v_risk = v_mean + 3.0 * v_std
        
        # 硬约束检查 (Hard Constraint)
        viol_hard = (next_state[IDX_VMAX] > v_max) or (next_state[IDX_TMAX] > t_max)
        is_risky = (v_risk > v_max)
        
        violation = bool(viol_hard)

        # 5. 计算奖励
        r_phys,r_soc ,r_time ,r_v , r_t , r_const , r_action = compute_paper_reward(
            soc_prev=curr_state[IDX_SOC],
            soc_next=next_state[IDX_SOC],
            v_max_next=next_state[IDX_VMAX],
            t_max_next=next_state[IDX_TMAX],
            std_soc_next=next_state[IDX_STD_SOC], # 参数名对应 paper_reward定义
            action_current=a_val,                 # 传入当前动作用于惩罚
            v_limit=v_max,
            t_limit=t_max,
            config=reward_cfg
        )

        # 6.【加入电压势垒 (Voltage Barrier)
        # 仅有撞墙后的惩罚不够，Agent 需要在撞墙前(4.15V)就感到疼痛。
        # 假设 compute_paper_reward 里没有这个逻辑，我们需要在这里手动补上。
        barrier_penalty = 0.0
        if v_mean > 4.15: # 软约束阈值
            # 随着电压接近 4.2，惩罚呈指数增长
            # 4.15V -> -0.2
            # 4.18V -> -2.0
            # 4.20V -> -20.0 (加上下面的 safety_penalty，总计 -70)
            barrier_penalty = -2.0 * np.exp(30.0 * (v_mean - 4.18))
        
        # 违规惩罚 (Violation Penalty)
        safety_penalty = 0.0
        if violation:
            safety_penalty = -50.0 # 给予重罚

        step_reward = r_phys + barrier_penalty + safety_penalty
        total_reward += step_reward

        # 6) violation 与终止条件（尽量贴近真实环境口径）
        soc_done = float(next_state[IDX_SOC]) >= 0.995

        terminated = soc_done or viol_hard or (k + 1 >= horizon)
        
        if violation:
            reason = "violation"
        elif soc_done:
            reason = "soc_full"
        elif k + 1 >= horizon:
            reason = "horizon"
        else:
            reason = "running"

        # 加入soc变化量
        delta_soc = next_state[IDX_SOC] - curr_state[IDX_SOC]
        v_soft = next_state[IDX_VMAX] > (v_max - 0.05)
        infos.append({
            "t": float((k + 1) * dt),
            "SOC_pack": float(next_state[IDX_SOC]),
            "std_SOC": float(next_state[IDX_STD_SOC]),
            "V_cell_max": float(next_state[IDX_VMAX]),
            "dV": float(next_state[IDX_DV]),
            "T_cell_max": float(next_state[IDX_TMAX]),
            "T_cell_min": float(next_state[IDX_TMIN]),
            "I": a_val,            # 当前动作占位符，将在下一轮 k+1 的开头被填入
            "I_prev": float(curr_state[IDX_IPREV]),       # 上一状态的历史电流（口径统一）
            "reward": step_reward,
            "violation": violation,
            "viol_hard": bool(viol_hard),
            "is_risky": bool(is_risky),
            "v_risk": float(v_risk),
            "terminated_reason": reason,
            "delta_soc": delta_soc,
            "v_soft": v_soft,
            "i_abs": float(abs(a_val)),    
        })
        
        curr_state = next_state
        if terminated:
            break
        # if len(infos) < 12:
        #     print("a_val", a_val)
    return total_reward, infos
