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

    real_episodes: int  # 跑真实环境，收集 (s, a, Δs)
    surrogate_epochs: int # 用这些数据训练静态代理模型的 epoch 数
    policy_epochs: int #用 surrogate 做 rollout，产生 “伪经验” 来训练 RL policy（DDPG）

"""
真实环境采样的关键逻辑 
"""
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
            transitions.append((state.copy(), action.copy(), delta.copy()))
            state = next_state
            prev_info = next_info
            done = terminated or truncated
    return transitions

def obs_from_info(info: dict) -> np.ndarray:
    return np.array(
        [
            info["SOC_pack"],
            info["V_cell_max"],
            info["V_cell_min"],
            info["dV"],
            info["T_cell_max"],
            info["T_cell_min"],
            info["dT"],
            info["std_V"],
            info["std_T"],
            info["std_SOC"],
            info["SOH_pack"],
            info["I_prev"],
        ],
        dtype=np.float32,
    )

def train_cycle0(
    env: BasePackEnv,
    agent: DDPGAgent,
    reward_cfg: PaperRewardConfig,
    surrogate: StaticSurrogate,
    config: Cycle0Config,
    run_dir: str,
) -> Dict[str, float]:
    """Run Cycle0 training pipeline."""

    noise = GaussianNoise(sigma=1.0)
    transitions = collect_real_data(env, reward_cfg, agent, noise, config.real_episodes)
    dataset = build_dataset(transitions)
    surrogate.fit(dataset, epochs=config.surrogate_epochs)
    # 测试 surrogate 单步误差
    s_test = dataset.states[:100]
    a_test = dataset.actions[:100]
    d_true = dataset.deltas[:100]

    errs = []
    for s, a, d in zip(s_test, a_test, d_true):
        d_pred, _ = surrogate.predict(s, a)
        errs.append(np.linalg.norm(d_pred - d))

    print("mean delta error:", np.mean(errs))
    print("max delta error:", np.max(errs))
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
        print(
            "epoch", epoch,
            "R", round(total_reward, 2),
            "SOC_end", round(infos[-1]["SOC_pack"], 4),
            "Vmax", round(max(i["V_cell_max"] for i in infos), 4),
            "Tmax", round(max(i["T_cell_max"] for i in infos), 2),
            "I_mean", round(np.mean([i["I"] for i in infos]), 3),
        )
        # ===== 验证 violation 是否生效 =====
        # print("violations:", sum(1 for x in infos if x.get("violation", False)))
        metrics = summarize_episode(infos)
        metrics.update({"epoch": epoch, "phase": "cycle0", "reward": total_reward})
        log_metrics(f"{run_dir}/metrics.jsonl", metrics)
        curve = curve_from_infos(infos)
        plot_episode(curve, f"{run_dir}/episode_{epoch}.png")

        for t in range(len(infos) - 1):
            s = obs_from_info(infos[t])
            a = np.array([infos[t]["I"]], dtype=np.float32)
            r = float(infos[t]["reward"])
            s_next = obs_from_info(infos[t + 1])

            done = (t == len(infos) - 2)
            agent.buffer.push(s, a, r, s_next, done)
            
        
        for _ in range(50):
            agent.update()

    return {"transitions": len(transitions)}
