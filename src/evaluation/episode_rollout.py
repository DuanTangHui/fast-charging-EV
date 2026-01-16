"""Episode rollout utilities for environments and surrogates."""
from __future__ import annotations

from typing import Callable, Dict, List, Tuple

import numpy as np

from ..envs.base_env import BasePackEnv
from ..rewards.paper_reward import PaperRewardConfig, reward_from_info
from .constraints import StateIndex, state_to_info, is_violation


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


def rollout_surrogate(
    state: np.ndarray,
    surrogate: Callable[[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]],
    policy: Callable[[np.ndarray], np.ndarray],
    horizon: int,
    reward_cfg: PaperRewardConfig,
    dt: float,
    v_max: float,
    t_max: float,
) -> Tuple[float, List[Dict]]:
    """Rollout on surrogate by iterating delta predictions."""

    infos: List[Dict] = []
    total_reward = 0.0
    t = 0.0
    prev_info = state_to_info(state, t, 0.0)
    for _ in range(horizon):
        action = policy(state)
        delta, _ = surrogate(state, action)
        next_state = state + delta
        # 防止发散的安全裁剪
        idx = StateIndex()
        next_state[idx.SOC_pack] = np.clip(next_state[idx.SOC_pack], 0.0, 1.0)
        next_state[idx.V_cell_max] = np.clip(next_state[idx.V_cell_max], 0.0, v_max)
        next_state[idx.T_cell_max] = np.clip(next_state[idx.T_cell_max], 0.0, t_max)
      
        next_state[idx.I_prev] = float(action[0])
        t += dt
        next_info = state_to_info(next_state, t, float(action[0]))
        next_info["violation"] = is_violation(next_state, v_max, t_max) 
        reward = reward_from_info(prev_info, next_info, reward_cfg, v_max, t_max)
        next_info["reward"] = reward
        infos.append(next_info)
        total_reward += reward
        state = next_state
        prev_info = next_info
    return total_reward, infos
