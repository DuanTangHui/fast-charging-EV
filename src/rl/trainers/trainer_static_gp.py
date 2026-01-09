"""Trainer implementing Algorithm 1 (Cycle0) with static surrogate."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List

import numpy as np

from ...envs.base_env import BasePackEnv
from ...evaluation.episode_rollout import rollout_env, rollout_surrogate
from ...evaluation.reports import summarize_episode
from ...rewards.paper_reward import PaperRewardConfig
from ...surrogate.dataset import build_dataset
from ...surrogate.gp_static import StaticSurrogate
from ..actor_critic_ddpg import DDPGAgent
from ..noise import GaussianNoise
from ...utils.logging import log_metrics
from ...envs.observables import curve_from_infos
from ...utils.plotting import plot_episode


@dataclass
class Cycle0Config:
    """Configuration for cycle0 training."""

    real_episodes: int
    surrogate_epochs: int
    policy_epochs: int


def collect_real_data(
    env: BasePackEnv,
    reward_cfg: PaperRewardConfig,
    agent: DDPGAgent,
    noise: GaussianNoise,
    episodes: int,
) -> List[tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Collect transitions from real environment."""

    transitions: List[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
    for _ in range(episodes):
        state, info = env.reset()
        done = False
        prev_info = info
        while not done:
            action = agent.act(state)
            action = np.array([action[0] + noise.sample()], dtype=np.float32)
            next_state, _, terminated, truncated, next_info = env.step(action)
            reward = reward_cfg
            _ = reward
            delta = next_state - state
            transitions.append((state, action, delta))
            state = next_state
            prev_info = next_info
            done = terminated or truncated
    return transitions


def train_cycle0(
    env: BasePackEnv,
    agent: DDPGAgent,
    reward_cfg: PaperRewardConfig,
    surrogate: StaticSurrogate,
    config: Cycle0Config,
    run_dir: str,
) -> Dict[str, float]:
    """Run Cycle0 training pipeline."""

    noise = GaussianNoise(sigma=0.2)
    transitions = collect_real_data(env, reward_cfg, agent, noise, config.real_episodes)
    dataset = build_dataset(transitions)
    surrogate.fit(dataset, epochs=config.surrogate_epochs)

    for epoch in range(config.policy_epochs):
        def policy(state: np.ndarray) -> np.ndarray:
            return agent.act(state)

        state, _ = env.reset()
        total_reward, infos = rollout_surrogate(
            state=state,
            surrogate=surrogate.predict,
            policy=policy,
            horizon=env.max_steps,
            reward_cfg=reward_cfg,
            dt=env.dt,
            v_max=env.v_max,
            t_max=env.t_max,
        )
        metrics = summarize_episode(infos)
        metrics.update({"epoch": epoch, "phase": "cycle0", "reward": total_reward})
        log_metrics(f"{run_dir}/metrics.jsonl", metrics)
        curve = curve_from_infos(infos)
        plot_episode(curve, f"{run_dir}/episode_{epoch}.png")

        for info in infos:
            agent.buffer.push(
                np.array(
                    [
                        info["SOC_pack"],
                        info["V_cell_max"],
                        info["V_cell_min"],
                        info["dV"],
                        info["T_cell_max"],
                        info["T_cell_min"],
                        info["dT"],
                        0.0,
                        0.0,
                        0.0,
                        1.0,
                        info["I"],
                    ],
                    dtype=np.float32,
                ),
                np.array([info["I"]], dtype=np.float32),
                info["reward"],
                np.array(
                    [
                        info["SOC_pack"],
                        info["V_cell_max"],
                        info["V_cell_min"],
                        info["dV"],
                        info["T_cell_max"],
                        info["T_cell_min"],
                        info["dT"],
                        0.0,
                        0.0,
                        0.0,
                        1.0,
                        info["I"],
                    ],
                    dtype=np.float32,
                ),
                False,
            )
        for _ in range(50):
            agent.update()

    return {"transitions": len(transitions)}
