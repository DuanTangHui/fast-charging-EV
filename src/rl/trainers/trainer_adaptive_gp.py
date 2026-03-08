"""Trainer implementing Algorithm 2 with adaptive differential surrogate (project-consistent)."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List

import matplotlib.pyplot as plt
import numpy as np

from src.rl import noise

from ...envs.base_env import BasePackEnv
from ...evaluation.episode_rollout import rollout_surrogate
from ...evaluation.reports import summarize_episode
from ...rewards.paper_reward import PaperRewardConfig
from ...surrogate.dataset import build_dataset
from ...surrogate.gp_combined import CombinedSurrogate
from ...surrogate.gp_differential import DifferentialSurrogate
from ...surrogate.gp_static import StaticSurrogate
from ...utils.logging import log_metrics
from ...envs.observables import curve_from_infos
from ...utils.plotting import plot_episode
from .trainer_static_gp import Cycle0Config, collect_real_data

def _is_on_policy_agent(agent: Any) -> bool:
    return bool(getattr(agent, "is_on_policy", False))


def _agent_buffer_len(agent: Any) -> int:
    buffer_obj = getattr(agent, "buffer", None)
    return len(buffer_obj) if buffer_obj is not None else 0


def _agent_ready_to_update(agent: Any) -> bool:
    batch_size = int(getattr(getattr(agent, "config", object()), "batch_size", 1))
    return _agent_buffer_len(agent) >= batch_size



def obs_from_info(info: dict) -> np.ndarray:
    """
    与 trainer_static_gp.py 保持一致：
    state = [SOC, stdSOC, Vmax, dV, Tmax, Tmin, I_prev]
    注意：surrogate rollout 的 info[t] 中 I_prev 语义是“上一步动作”。
    """
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

def validate_full_charge_comparison_adaptive(
    env: BasePackEnv,
    agent: Any,
    static_surrogate: StaticSurrogate,
    combined_surrogate: CombinedSurrogate,
    save_path: str,
    hold_steps: int = 1,
) -> None:
    """对比一次完整充电：真实老化环境 vs 静态代理模型 vs 组合代理模型。"""
    real_traj = {"SOC": [], "V": [], "T": [], "I": []}

    state_real, _ = env.reset(seed=123)
    done_real = False
    while not done_real:
        raw_action = float(agent.act(state_real)[0])
        action = np.array([raw_action], dtype=np.float32)

        real_traj["SOC"].append(float(state_real[0]))
        real_traj["V"].append(float(state_real[2]))
        real_traj["T"].append(float(state_real[4]))
        real_traj["I"].append(raw_action)

        for _ in range(hold_steps):
            state_real, _, terminated, truncated, _ = env.step(action)
            if terminated or truncated:
                done_real = True
                break

    horizon = len(real_traj["SOC"])

    static_traj = {"SOC": [], "V": [], "T": [], "I": []}
    combined_traj = {"SOC": [], "V": [], "T": [], "I": []}

    state0, _ = env.reset(seed=123)
    curr_static = state0.copy()
    curr_combined = state0.copy()

    for _ in range(horizon):
        act_static = float(agent.act(curr_static)[0])
        static_traj["SOC"].append(float(curr_static[0]))
        static_traj["V"].append(float(curr_static[2]))
        static_traj["T"].append(float(curr_static[4]))
        static_traj["I"].append(act_static)

        delta_static, _ = static_surrogate.predict(curr_static, np.array([act_static], dtype=np.float32))
        curr_static[:6] += delta_static
        curr_static[6] = act_static

        act_combined = float(agent.act(curr_combined)[0])
        combined_traj["SOC"].append(float(curr_combined[0]))
        combined_traj["V"].append(float(curr_combined[2]))
        combined_traj["T"].append(float(curr_combined[4]))
        combined_traj["I"].append(act_combined)

        delta_combined, _ = combined_surrogate.predict(curr_combined, np.array([act_combined], dtype=np.float32))
        curr_combined[:6] += delta_combined
        curr_combined[6] = act_combined

    fig, axes = plt.subplots(4, 1, figsize=(10, 15), sharex=True)
    time_step = hold_steps * float(env.dt)
    keys = ["SOC", "V", "T", "I"]
    ylabels = ["SOC", "Voltage (V)", "Temperature (K)", "Current (A)"]

    for i, key in enumerate(keys):
        ax = axes[i]
        t = np.arange(len(real_traj[key])) * time_step
        ax.plot(t, real_traj[key], "k-", label="Real Aging Env", linewidth=2.0)
        ax.plot(t, static_traj[key], "r--", label="Static Surrogate", linewidth=1.8)
        ax.plot(t, combined_traj[key], "b-.", label="Combined Surrogate", linewidth=1.8)
        ax.set_ylabel(ylabels[i])
        ax.grid(True, linestyle=":", alpha=0.6)
        if i == 0:
            ax.legend(loc="upper left")

    axes[-1].set_xlabel("Time (seconds)")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close(fig)


@dataclass
class AdaptiveConfig:
    """Configuration for adaptive training cycles (aligned with Cycle0Config style)."""

    cycles: int
    real_episodes_per_cycle: int
    surrogate_epochs: int
    policy_epochs: int

    # ====== NEW: keep same structure as cycle0 ======
    policy_rollouts_per_epoch: int = 3
    updates_per_epoch: int = 50
    plot_interval: int = 1

    # exploration noise on surrogate rollouts (absolute A)
    noise_sigma_start: float = 0.2
    noise_sigma_end: float = 0.05 
    hold_steps: int = 1
    v_soft_max: float = 4.17
    t_soft_max: float = 309.5


def train_adaptive_cycles(
    env: BasePackEnv,
    agent: Any,
    static_surrogate: StaticSurrogate,
    diff_surrogate: DifferentialSurrogate,
    combined: CombinedSurrogate,
    reward_cfg: PaperRewardConfig,
    config: AdaptiveConfig,
    run_dir: str,
    soh_enabled: bool = False,        # 先忽略 SOH，默认关
    lambda_prior: float = 0.0,
    theta_dim: int = 0,
    dummy_soh: float = 0.0,
) -> List[Dict[str, float]]:
    """
    Algorithm-2 style loop:
      cycle:
        1) real rollouts (policy = agent) -> transitions
        2) fit differential on residual: delta_hat = delta_real - delta_static
        3) surrogate rollouts (policy = agent + noise) -> agent.observe + agent.update
    """
    all_metrics: List[Dict[str, float]] = []

    low = float(agent.config.action_low)
    high = float(agent.config.action_high)

    for cycle in range(1, config.cycles + 1):
        # (A) Collect REAL transitions (use agent policy, not random)
        # ------------------------------------------------------------
        cycle0_like_cfg = Cycle0Config(
            real_episodes=config.real_episodes_per_cycle,
            surrogate_epochs=config.surrogate_epochs,
            policy_epochs=config.policy_epochs,
            policy_rollouts_per_epoch=config.policy_rollouts_per_epoch,
            updates_per_epoch=config.updates_per_epoch,
            plot_interval=config.plot_interval,
            noise_sigma_start=config.noise_sigma_start,
            noise_sigma_end=config.noise_sigma_end,
            hold_steps=config.hold_steps,
            v_soft_max=config.v_soft_max,
            t_soft_max=config.t_soft_max,
        )
        transitions, _ = collect_real_data(env, reward_cfg, agent, cycle0_like_cfg)
        # ------------------------------------------------------------
        # (B) Fit DIFFERENTIAL surrogate on residual
        #     delta_hat = delta_real - delta_static
        # ------------------------------------------------------------
        residual_transitions: List[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
        for (s, a, delta_real6) in transitions:
            delta_static6, _ = static_surrogate.predict(s, a)  # static 输出 6 维
            delta_hat6 = delta_real6 - delta_static6
            residual_transitions.append((s, a, delta_hat6))

        dataset = build_dataset(residual_transitions)
        diff_surrogate.fit(dataset, epochs=config.surrogate_epochs)

        validate_full_charge_comparison_adaptive(
            env=env,
            agent=agent,
            static_surrogate=static_surrogate,
            combined_surrogate=combined,
            save_path=f"{run_dir}/cycle_{cycle}_full_charge_compare.png",
            hold_steps=config.hold_steps,
        )

        # ------------------------------------------------------------
        # (C) Policy training on COMBINED surrogate (same as cycle0 style)
        # ------------------------------------------------------------
        actor_losses: List[float] = []
        critic_losses: List[float] = []
        print(f"训练开始前 Buffer 大小: {_agent_buffer_len(agent)}")
        for epoch in range(config.policy_epochs):
            # 噪声衰减（和静态 trainer 一样的思想）:contentReference[oaicite:3]{index=3}
            progress = epoch / max(1, config.policy_epochs - 1)
            current_sigma = config.noise_sigma_start * (1 - progress) + config.noise_sigma_end * progress
            sigma_start = 2.0
            sigma_end = 0.1
            current_sigma = max(sigma_end, sigma_start - (sigma_start - sigma_end) * progress)
            def policy_train(s: np.ndarray) -> np.ndarray:
                a_det = float(agent.act(s)[0])

                if _is_on_policy_agent(agent):
                    noise = 0.0
                else:
                    noise = np.random.normal(0.0, current_sigma)

                # 防止 0A 截断导致动作分布畸变（静态 trainer 的技巧）:contentReference[oaicite:4]{index=4}
                if a_det + noise > high:
                    noise = -abs(noise)
                elif a_det + noise < low:
                    noise = abs(noise)

                a_noisy = a_det + noise
                a_final = float(np.clip(a_noisy, low, high))
                return np.array([a_final], dtype=np.float32)

            epoch_a_loss: List[float] = []
            epoch_c_loss: List[float] = []
            for rollout_idx in range(config.policy_rollouts_per_epoch):
                if rollout_idx == 0 or dataset.states.shape[0] == 0:
                    state0, _ = env.reset()
                else:
                    state0 = dataset.states[np.random.randint(0, dataset.states.shape[0])].copy()

                total_reward, infos = rollout_surrogate(
                    state=state0,
                    surrogate=combined.predict,
                    policy=policy_train,
                    horizon=env.max_steps,
                    reward_cfg=reward_cfg,
                    dt=env.dt * config.hold_steps,
                    v_max=env.v_max,
                    t_max=env.t_max,
                )

                # 用 surrogate infos 喂给 agent（完全复用 cycle0 的口径）:contentReference[oaicite:6]{index=6}
                if rollout_idx == 0:
                    print(
                        "epoch", epoch,
                        "R", round(total_reward, 2),
                        "SOC_end", round(infos[-1]["SOC_pack"], 4),
                        "Vmax", round(max(i["V_cell_max"] for i in infos), 4),
                        "Tmax", round(max(i["T_cell_max"] for i in infos), 2),
                        "I_mean", round(np.mean([i["I"] for i in infos]), 3),
                        "violations:", sum(1 for x in infos if x.get("violation", False)),
                    )
                    metrics = summarize_episode(infos)
                    metrics.update({"epoch": epoch, "cycle": cycle, "phase": "adaptive", "reward": total_reward})
                    log_metrics(f"{run_dir}/metrics.jsonl", metrics)

                    if config.plot_interval > 0 and epoch % config.plot_interval == 0:
                        curve = curve_from_infos(infos)
                        plot_episode(curve, f"{run_dir}/episode_{epoch}.png")

                zero_cnt = 0
                total_cnt = 0
                a_min, a_max = 1e9, -1e9
                a_sum, a_sq = 0.0, 0.0

                for t in range(len(infos) - 1):
                    curr_info = infos[t]
                    next_info = infos[t + 1]

                    s = obs_from_info(curr_info)
                    s_next = obs_from_info(next_info)

                    a_val = float(next_info["I"])
                    a = np.array([a_val], dtype=np.float32)
                    r = float(next_info["reward"])

                    is_violation = bool(next_info.get("violation", False))
                    is_last_step_in_list = (t == len(infos) - 2)
                    done = is_violation or is_last_step_in_list

                    agent.observe(s, a, r, s_next, done)

                    total_cnt += 1
                    if abs(a_val) < 1e-8:
                        zero_cnt += 1
                    a_min = min(a_min, a_val)
                    a_max = max(a_max, a_val)
                    a_sum += a_val
                    a_sq += a_val * a_val
                
                if total_cnt > 0:
                    mean = a_sum / total_cnt
                    var = a_sq / total_cnt - mean**2
                    std = (var if var > 0 else 0.0) ** 0.5
                    print(
                        f"[BUF] actions: zero_ratio={zero_cnt/total_cnt:.3f} "
                        f"mean={mean:.3f} std={std:.3f} min={a_min:.3f} max={a_max:.3f}"
                    )

                if _is_on_policy_agent(agent) and _agent_ready_to_update(agent):
                    loss_a, loss_c = agent.update()
                    if loss_a is not None:
                        epoch_a_loss.append(float(loss_a))
                    if loss_c is not None:
                        epoch_c_loss.append(float(loss_c))

            for _ in range(config.updates_per_epoch):
                if (not _is_on_policy_agent(agent)) and _agent_ready_to_update(agent):
                    loss_a, loss_c = agent.update()
                    if loss_a is not None:
                        epoch_a_loss.append(float(loss_a))
                    if loss_c is not None:
                        epoch_c_loss.append(float(loss_c))

            if epoch_a_loss:
                avg_a = float(np.mean(epoch_a_loss))
                avg_c = float(np.mean(epoch_c_loss))
                print(f"       -> Loss | Actor: {avg_a:.4f} | Critic: {avg_c:.4f}")
                actor_losses.append(avg_a)
                critic_losses.append(avg_c)

            all_metrics.append(
                {
                    "epoch": epoch,
                    "cycle": cycle,
                    "phase": "adaptive",
                    "sigma": float(current_sigma),
                    "a_loss": float(np.mean(actor_losses)) if actor_losses else 0.0,
                    "c_loss": float(np.mean(critic_losses)) if critic_losses else 0.0,
                }
            )
            
        if hasattr(agent, "save"):
            agent.save(f"{run_dir}/cycle_{cycle}_agent_ckpt.pt")
            
    return all_metrics
