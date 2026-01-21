"""Trainer implementing Algorithm 1 (Cycle0) with static surrogate."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Tuple
import csv

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

def save_transitions_to_csv(
    transitions: List[Tuple[np.ndarray, np.ndarray, np.ndarray]],
    filepath: str
) -> None:

    # 生成表头
    state_cols = [f"s_{i}" for i in range(len(transitions[0][0]))]
    action_cols = ["action"]
    delta_cols = [f"d_{i}" for i in range(len(transitions[0][2]))]

    header = state_cols + action_cols + delta_cols

    with open(filepath, mode="w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)

        for state, action, delta in transitions:
            row = (
                list(np.asarray(state).reshape(-1)) +
                list(np.asarray(action).reshape(-1)) +
                list(np.asarray(delta).reshape(-1))
            )
            writer.writerow(row)

    print(f"[OK] Saved {len(transitions)} transitions to: {filepath}")

@dataclass
class Cycle0Config:
    """Configuration for cycle0 training."""

    real_episodes: int  # 跑真实环境，收集 (s, a, Δs)
    surrogate_epochs: int # 用这些数据训练静态代理模型的 epoch 数
    policy_epochs: int #用 surrogate 做 rollout，产生 “伪经验” 来训练 RL policy（DDPG）
    policy_rollouts_per_epoch: int = 3  # 每个 epoch 用 surrogate rollout 的条数
    updates_per_epoch: int = 50  # 每个 epoch 更新 policy 的次数
    plot_interval: int = 1  # 每隔多少个 epoch 保存一次轨迹图（1=每次）
    # ====== collect_real_data 探索参数 ======
    eps_random_start: float = 0.85     # 你的环境很容易撞 Vmax，随机比例建议更高
    eps_random_end: float = 0.25

    # 这里写“比例”，真正 sigma 会乘 (high-low)
    noise_sigma_start: float = 0.20
    noise_sigma_end: float = 0.05

    hold_steps: int = 5               # dt=10s，hold 5步=50s，更像阶梯恒流
    v_soft_max: float = 4.17   # 或 env.v_max - 0.03
    t_soft_max: float = 318.5  # 或 env.t_max - 1.5

"""
真实环境采样的关键逻辑 
"""
def collect_real_data(
    env: BasePackEnv,
    reward_cfg: PaperRewardConfig,
    agent: DDPGAgent,
    config: Cycle0Config,
) -> List[tuple[np.ndarray, np.ndarray, np.ndarray]]:
    low = float(agent.config.action_low)   # -20
    high = float(agent.config.action_high) # 0

    transitions: List[tuple[np.ndarray, np.ndarray, np.ndarray]] = []

    # ===== Guard 超参数（建议你先用这套）=====
    v_soft = float(getattr(config, "v_soft_max", env.v_max - 0.03))  # 比硬约束低一点，比如 4.17
    t_soft = float(getattr(config, "t_soft_max", env.t_max - 1.5))   # 比硬约束低一点，比如 318.5K

    # 当接近边界时，把电流往 0A 拉
    def guard_action(prev_action: float, info: dict) -> float:
        v = float(info.get("V_cell_max", -1e9))
        t = float(info.get("T_cell_max", -1e9))
        viol = bool(info.get("violation", False))

        # 已经违规：立即大幅减小电流（靠近0）
        if viol:
            return min(high, prev_action + 5.0)  # prev_action是负数，加5就是减小幅值

        # 接近电压或温度软阈值：渐进减小电流
        if v >= v_soft or t >= t_soft:
            return min(high, prev_action + 2.0)

        return prev_action

    for ep in range(config.real_episodes):
        frac = ep / max(1, config.real_episodes - 1)

        eps_random = (1 - frac) * config.eps_random_start + frac * config.eps_random_end
        sigma = ((1 - frac) * config.noise_sigma_start + frac * config.noise_sigma_end) * (high - low)
        noise = GaussianNoise(sigma=float(sigma))

        state, info = env.reset()
        done = False

        hold = 0
        held_action = None

        while not done:
            if held_action is None or hold <= 0:
                # ---------- 先生成“候选动作” ----------
                if np.random.rand() < eps_random:
                    # 随机动作：偏向温和（靠近0A），避免大量episode早终止
                    u = np.random.beta(5.0, 2.0)  # 偏向1
                    a = low + u * (high - low)    # 更接近 high=0
                else:
                    a = float(agent.act(state)[0]) + noise.sample()

                a = float(np.clip(a, low, high))

                # ---------- 再做“安全守卫” ----------
                # 用上一时刻info判断是否需要回退
                if info is not None:
                    a = guard_action(a, info)
                    a = float(np.clip(a, low, high))

                held_action = np.array([a], dtype=np.float32)
                hold = int(config.hold_steps)

            action = held_action
            hold -= 1

            next_state, _, terminated, truncated, next_info = env.step(action)

            delta = next_state - state
            delta = next_state[:11] - state[:11]
            transitions.append((state.copy(), action.copy(), delta.copy()))

            state = next_state
            info = next_info
            done = bool(terminated or truncated)

    return transitions

def obs_from_info(info: dict) -> np.ndarray:
    return np.array(
        [
            info["SOC_pack"],
            info.get("std_SOC", 0.0),
            info["V_cell_max"],
            info["dV"],
            info["T_cell_max"],
            info["T_cell_min"],
            info.get("I_prev", info.get("I", 0.0)),
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

    transitions = collect_real_data(env, reward_cfg, agent, config)
    save_transitions_to_csv(transitions, "dataset.csv")
    actions = np.array([a[0] for _, a, _ in transitions], dtype=float)
    print("Action stats: mean", actions.mean(), "std", actions.std(), "min", actions.min(), "max", actions.max())

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
        for rollout_idx in range(config.policy_rollouts_per_epoch):
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
            if rollout_idx == 0:
                print(
                    "epoch", epoch,
                    "R", round(total_reward, 2),
                    "SOC_end", round(infos[-1]["SOC_pack"], 4),
                    "Vmax", round(max(i["V_cell_max"] for i in infos), 4),
                    "Tmax", round(max(i["T_cell_max"] for i in infos), 2),
                    "I_mean", round(np.mean([i["I"] for i in infos]), 3),
                )
                print("---- last 10 steps ----")
                for i, step in enumerate(infos[-10:]):
                    print(
                        f"step {-10 + i}:",
                        "SOC:", round(step["SOC_pack"], 4),
                        "Vmax:", round(step["V_cell_max"], 4),
                        "I:", round(step["I"], 3)
                    )
                print("-----------------------")
                # ===== 验证 violation 是否生效 =====
                print("violations:", sum(1 for x in infos if x.get("violation", False)))
                metrics = summarize_episode(infos)
                metrics.update({"epoch": epoch, "phase": "cycle0", "reward": total_reward})
               

                log_metrics(f"{run_dir}/metrics.jsonl", metrics)
                if config.plot_interval > 0 and epoch % config.plot_interval == 0:
                    curve = curve_from_infos(infos)
                    plot_episode(curve, f"{run_dir}/episode_{epoch}.png")

            for t in range(len(infos) - 1):
                s = obs_from_info(infos[t])
                a = np.array([infos[t]["I"]], dtype=np.float32)
                r = float(infos[t]["reward"])
                s_next = obs_from_info(infos[t + 1])

                done = (t == len(infos) - 2)
                agent.observe(s, a, r, s_next, done)

        for _ in range(config.updates_per_epoch):
            agent.update()

    return {"transitions": len(transitions)}
