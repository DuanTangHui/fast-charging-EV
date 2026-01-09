"""Episode rollout utilities for environments and surrogates."""
from __future__ import annotations

from typing import Callable, Dict, List, Tuple

import numpy as np

from ..envs.base_env import BasePackEnv
from ..rewards.paper_reward import PaperRewardConfig, reward_from_info
from .constraints import state_to_info


def rollout_env(
    env: BasePackEnv,
    policy: Callable[[np.ndarray], np.ndarray],
    reward_cfg: PaperRewardConfig,
) -> Tuple[float, List[Dict]]:
    """Rollout in a real environment."""

    state, info = env.reset()
    infos: List[Dict] = []
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
        t += dt
        next_info = state_to_info(next_state, t, float(action[0]))
        reward = reward_from_info(prev_info, next_info, reward_cfg, v_max, t_max)
        next_info["reward"] = reward
        infos.append(next_info)
        total_reward += reward
        state = next_state
        prev_info = next_info
    return total_reward, infos
