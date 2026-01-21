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
    print("real env info keys:", sorted(info.keys()))

    infos: List[Dict] = [info]
    total_reward = 0.0
    prev_info = info
    done = False
    while not done:
        action = policy(state)
        next_state, _, terminated, truncated, next_info = env.step(action)
        reward = reward_from_info(prev_info, next_info, reward_cfg, env.v_max, env.t_max)
        next_info["reward"] = reward
        infos.append(next_info)
        total_reward += reward
        state = next_state
        prev_info = next_info
        done = terminated or truncated
        if len(infos) < 5:
            print("policy_action:", action)

        # rollout_env 末尾 return 前
    print("policy:", getattr(policy, "__name__", type(policy).__name__))

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
        "I_min=", round(np.min([i["I"] for i in infos[1:]]), 2) if len(infos) > 1 else None,
        "I_exec_mean=", round(np.mean([i.get("I_pack_est", i["I"]) for i in infos[1:]]), 2),
        "I_exec_min=", round(np.min([i.get("I_pack_est", i["I"]) for i in infos[1:]]), 2),
    )

    return total_reward, infos


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
    Hao et al. 风格的安全引导 Rollout。
    处理 (Mean, Std) 预测，并进行不确定性惩罚。
    """
    # 状态索引定义 (Hardcoded for 7-dim state)
    IDX_SOC = 0
    IDX_STD_SOC = 1
    IDX_V_MAX = 2
    IDX_DV = 3
    IDX_T_MAX = 4
    IDX_T_MIN = 5
    IDX_I_PREV = 6

    curr_state = state.copy()
    total_reward = 0.0
    infos = []
    
    # 初始化 info
    curr_info = {
        "SOC_pack": curr_state[IDX_SOC],
        "std_SOC": curr_state[IDX_STD_SOC],
        "V_cell_max": curr_state[IDX_V_MAX],
        "dV": curr_state[IDX_DV],
        "T_cell_max": curr_state[IDX_T_MAX],
        "T_cell_min": curr_state[IDX_T_MIN],
        "I": curr_state[IDX_I_PREV], 
        "reward": 0.0,
        "violation": False
    }
    infos.append(curr_info)

    for _ in range(horizon):
        # 1. 策略动作
        action = policy(curr_state) 
        a_val = float(action[0])

        # 2. 预测 (Mean, Std)
        delta_mean, delta_std = surrogate(curr_state, action)

        # 3. 状态更新
        next_state = np.zeros_like(curr_state)
        # 模型预测前6维变化
        next_state[:6] = curr_state[:6] + delta_mean[:6] 
        # 第7维 (I_prev) 直接更新为当前动作
        next_state[IDX_I_PREV] = a_val 

        # 4. 安全检查 (Hao et al. 核心)
        v_pred_mean = next_state[IDX_V_MAX]
        v_pred_std = delta_std[IDX_V_MAX]
        
        # 悲观估计: Upper Bound
        v_risk = v_pred_mean + 3.0 * v_pred_std
        
        safety_penalty = 0.0
        is_risky = False
        
        if v_risk > v_max:
            is_risky = True
            safety_penalty = -50.0 

        # 5. 计算奖励
        # 【修改修复点】删除了 t_prev, t_next，修正了参数名
        r_phys = compute_paper_reward(
            soc_prev=curr_state[IDX_SOC],
            soc_next=next_state[IDX_SOC],
            v_max_next=next_state[IDX_V_MAX],
            t_max_next=next_state[IDX_T_MAX],
            std_soc_next=next_state[IDX_STD_SOC], # 参数名对应 paper_reward定义
            action_current=a_val,                 # 传入当前动作用于惩罚
            v_limit=v_max,
            t_limit=t_max,
            config=reward_cfg
        )

        step_reward = r_phys + safety_penalty
        total_reward += step_reward

        info = {
            "SOC_pack": next_state[IDX_SOC],
            "std_SOC": next_state[IDX_STD_SOC],
            "V_cell_max": next_state[IDX_V_MAX],
            "dV": next_state[IDX_DV],
            "T_cell_max": next_state[IDX_T_MAX],
            "T_cell_min": next_state[IDX_T_MIN],
            "I": a_val,
            "reward": step_reward,
            "violation": is_risky or (next_state[IDX_V_MAX] > v_max)
        }
        infos.append(info)

        # 简单的终止条件
        if next_state[IDX_SOC] >= 1.0 or next_state[IDX_V_MAX] > (v_max + 0.1):
            break
            
        curr_state = next_state

    return total_reward, infos
