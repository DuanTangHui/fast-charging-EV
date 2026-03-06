"""Trainer implementing Algorithm 2 with adaptive differential surrogate (project-consistent)."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List
import csv
import os

import numpy as np
import matplotlib.pyplot as plt
import pybamm
from scipy.interpolate import interp1d
from ...envs.base_env import BasePackEnv
from ...evaluation.episode_rollout import rollout_env, rollout_surrogate
from ...evaluation.reports import summarize_episode
from ...rewards.paper_reward import PaperRewardConfig
from ..noise import GaussianNoise
from ...surrogate.dataset import build_dataset
from ...surrogate.gp_combined import CombinedSurrogate
from ...surrogate.gp_differential import DifferentialSurrogate
from ...surrogate.gp_static import StaticSurrogate
from ...utils.logging import log_metrics
from ...envs.observables import curve_from_infos
from ...utils.plotting import plot_episode


def _is_on_policy_agent(agent: Any) -> bool:
    return bool(getattr(agent, "is_on_policy", False))



def _agent_buffer_len(agent: Any) -> int:
    buffer_obj = getattr(agent, "buffer", None)
    return len(buffer_obj) if buffer_obj is not None else 0


def _agent_ready_to_update(agent: Any) -> bool:
    batch_size = int(getattr(getattr(agent, "config", object()), "batch_size", 1))
    return _agent_buffer_len(agent) >= batch_size


def get_chen2020_ocv_func():
    param = pybamm.ParameterValues("Chen2020")
    U_p = param["Positive electrode OCP [V]"]
    U_n = param["Negative electrode OCP [V]"]
    try:
        sto_n_0 = param["Lower stoichiometric limit in negative electrode"]
        sto_n_1 = param["Upper stoichiometric limit in negative electrode"]
        sto_p_0 = param["Upper stoichiometric limit in positive electrode"]
        sto_p_1 = param["Lower stoichiometric limit in positive electrode"]
    except KeyError:
        sto_n_0, sto_n_1 = 0.0279, 0.9014
        sto_p_0, sto_p_1 = 0.9077, 0.2661

    soc_range = np.linspace(0, 1, 100)
    ocv_values = []
    for soc in soc_range:
        curr_sto_n = sto_n_0 + soc * (sto_n_1 - sto_n_0)
        curr_sto_p = sto_p_0 - soc * (sto_p_0 - sto_p_1)
        v = param.evaluate(U_p(pybamm.Scalar(curr_sto_p))) - param.evaluate(U_n(pybamm.Scalar(curr_sto_n)))
        ocv_values.append(float(v))

    return interp1d(soc_range, ocv_values, kind="linear", fill_value="extrapolate")

def save_transitions_with_episode_to_csv(
    transitions: List[tuple[np.ndarray, np.ndarray, np.ndarray]],
    episode_ids: List[int],
    filepath: str,
) -> None:
    if not transitions:
        return
    if len(transitions) != len(episode_ids):
        raise ValueError("transitions and episode_ids length mismatch")

    state_cols = [f"s_{i}" for i in range(len(transitions[0][0]))]
    action_cols = ["action"]
    delta_cols = [f"d_{i}" for i in range(len(transitions[0][2]))]
    header = ["episode"] + state_cols + action_cols + delta_cols

    with open(filepath, mode="w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for ep, (state, action, delta) in zip(episode_ids, transitions):
            row = [int(ep)] + list(np.asarray(state).reshape(-1)) + list(np.asarray(action).reshape(-1)) + list(np.asarray(delta).reshape(-1))
            writer.writerow(row)

    print(f"[OK] Saved {len(transitions)} transitions with episode ids to: {filepath}")


def plot_transition_coverage(
    transitions: List[tuple[np.ndarray, np.ndarray, np.ndarray]],
    save_path: str,
) -> None:
    if not transitions:
        return

    soc_values = np.array([t[0][0] for t in transitions], dtype=float)
    actions = np.array([t[1][0] for t in transitions], dtype=float)

    plt.figure(figsize=(15, 6))
    plt.subplot(1, 2, 1)
    plt.hist(soc_values, bins=50, color="skyblue", edgecolor="black", alpha=0.7)
    plt.axvline(0.6, color="red", linestyle="--", label="SOC=0.6")
    plt.title("Adaptive Real Data SOC Distribution")
    plt.xlabel("SOC")
    plt.ylabel("Count")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.subplot(1, 2, 2)
    plt.scatter(soc_values, actions, alpha=0.4, s=15, c="darkblue")
    plt.axvline(0.6, color="red", linestyle="--", label="SOC=0.6")
    plt.title("Adaptive Real Data Action vs SOC")
    plt.xlabel("SOC")
    plt.ylabel("Action (A)")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"[OK] Saved transition coverage plot to: {save_path}")


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
def _plot_model_comparison(
    env: BasePackEnv,
    agent: Any,
    reward_cfg: PaperRewardConfig,
    combined: CombinedSurrogate,
    static_surrogate: StaticSurrogate,
    save_path: str,
    seed: int,
) -> None:
    """同一策略下对比：真实环境 vs 组合模型 vs 静态代理模型。"""
    low = float(agent.config.action_low)
    high = float(agent.config.action_high)

    def policy_eval(s: np.ndarray) -> np.ndarray:
        a = float(np.clip(agent.act(s)[0], low, high))
        return np.array([a], dtype=np.float32)

    # 固定种子拿同一起点
    state0, info0 = env.reset(seed=seed)

    # 真实环境轨迹
    infos_real: List[Dict[str, float]] = [info0]
    state = state0.copy()

    done = False
    while not done:
        a = policy_eval(state)
        next_state, _, terminated, truncated, next_info = env.step(a)
        infos_real.append(next_info)
        state = next_state
        done = terminated or truncated

    # surrogate 轨迹（同一初始状态）
    _, infos_combined = rollout_surrogate(
        state=state0.copy(),
        surrogate=combined.predict,
        policy=policy_eval,
        horizon=env.max_steps,
        reward_cfg=reward_cfg,
        dt=env.dt,
        v_max=env.v_max,
        t_max=env.t_max,
    )
    _, infos_static = rollout_surrogate(
        state=state0.copy(),
        surrogate=static_surrogate.predict,
        policy=policy_eval,
        horizon=env.max_steps,
        reward_cfg=reward_cfg,
        dt=env.dt,
        v_max=env.v_max,
        t_max=env.t_max,
    )

    curves = {
        "real": curve_from_infos(infos_real),
        "combined": curve_from_infos(infos_combined),
        "static": curve_from_infos(infos_static),
    }
    key_map = ["SOC_pack", "V_cell_max", "T_cell_max", "I"]
    labels = ["SOC", "Voltage (V)", "Temperature (K)", "Current (A)"]
    styles = {
        "real": ("k-", "Real"),
        "combined": ("b--", "Combined"),
        "static": ("r:", "Static"),
    }

    fig, axes = plt.subplots(4, 1, figsize=(10, 14), sharex=True)
    for idx, curve_key in enumerate(key_map):
        ax = axes[idx]
        for name in ["real", "combined", "static"]:
            style, legend = styles[name]
            t = np.asarray(curves[name]["t"], dtype=float)
            y = np.asarray(curves[name][curve_key], dtype=float)
            n = min(t.size, y.size)
            ax.plot(t[:n], y[:n], style, label=legend, linewidth=1.8)
        ax.set_ylabel(labels[idx])
        ax.grid(True, linestyle=":", alpha=0.5)
        if idx == 0:
            ax.legend(loc="best")
    axes[-1].set_xlabel("Time (s)")
    plt.suptitle(f"stage={infos_real[0].get('aging_stage', 'n/a')}, Ri_netlist={infos_real[0].get('netlist_ri_ohm', float('nan')):.6f}Ω, dR={infos_real[0].get('contact_resistance_ohm', float('nan')):.6f}Ω")
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig(save_path, dpi=150)
    plt.close(fig)

@dataclass
class AdaptiveConfig:
    """Configuration for adaptive training cycles (aligned with Cycle0Config style)."""

    cycles: int
    real_episodes_per_cycle: int
    surrogate_epochs: int
    policy_epochs: int
    virtual_episodes_per_cycle: int = 270
    # ====== NEW: keep same structure as cycle0 ======
    policy_rollouts_per_epoch: int = 1
    updates_per_epoch: int = 50
    updates_per_virtual_episode: int = 50
    plot_interval: int = 1

    # exploration noise on surrogate rollouts (absolute A)
    noise_sigma_start: float = 2.0
    noise_sigma_end: float = 0.1
    hold_steps: int = 1
    real_noise_sigma_start: float = 0.016
    real_noise_sigma_end: float = 0.003
    save_real_transitions_csv: bool = True
    plot_real_transitions_coverage: bool = True

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
        # ------------------------------------------------------------
        # (A) Collect REAL transitions (use agent policy, not random)
        # ------------------------------------------------------------
        transitions: List[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
        
        transition_episode_ids: List[int] = []

        get_ocv = get_chen2020_ocv_func()

        def physics_limit_action(action_from_agent: float, info: dict) -> float:
            soc_curr = float(info.get("SOC_cell_max", info.get("SOC_pack")))
            u_ocv = float(get_ocv(soc_curr))
            v_limit = 4.2
            r_internal = 0.025
            i_bound_single = max(0.0, (v_limit - u_ocv) / r_internal)
            i_bound_pack = i_bound_single * 3.0
            safe_action = np.clip(action_from_agent, -i_bound_pack, 0.0)
            return float(safe_action)
        
        for ep in range(config.real_episodes_per_cycle):
            frac = ep / max(1, config.real_episodes_per_cycle - 1)
            sigma = ((1 - frac) * config.real_noise_sigma_start + frac * config.real_noise_sigma_end) * (high - low)
            noise = GaussianNoise(sigma=float(sigma))

            def policy_real(state: np.ndarray) -> np.ndarray:
                return np.array([float(agent.act(state)[0])], dtype=np.float32)
                
            _, infos = rollout_env(
                env=env,
                policy=policy_real,
                reward_cfg=reward_cfg,
                hold_steps=config.hold_steps,
                action_low=low,
                action_high=high,
                apply_noise=True,
                noise_sampler=noise.sample,
                action_postprocess=physics_limit_action,
                observer=agent,
                update_agent=True,
                ready_to_update=_agent_ready_to_update,
                on_policy_agent=_is_on_policy_agent(agent),
                reset_options={"soc_low": 0.1, "soc_high": 0.9},
                collect_deltas=True,
            )

            macro_transitions = infos[0].get("macro_transitions", [])
            transitions.extend(macro_transitions)
            transition_episode_ids.extend([ep + 1] * len(macro_transitions))

        if config.save_real_transitions_csv and transitions:
            csv_path = os.path.join(run_dir, f"cycle_{cycle}_real_transitions.csv")
            save_transitions_with_episode_to_csv(transitions, transition_episode_ids, csv_path)
        
        if config.plot_real_transitions_coverage and transitions:
            plot_path = os.path.join(run_dir, f"cycle_{cycle}_real_data_coverage.png")
            plot_transition_coverage(transitions, plot_path)
          

        # ------------------------------------------------------------
        # (B) Fit DIFFERENTIAL surrogate on residual
        #     delta_hat = delta_real - delta_static
        # ------------------------------------------------------------
        residual_transitions: List[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
        for (s, a, delta_real6) in transitions:
            delta_static6, _ = static_surrogate.predict(s, a)  # static 输出 6 维
            delta_hat6 = delta_real6 - delta_static6
            residual_transitions.append((s, a, delta_hat6))

        dataset_hat = build_dataset(residual_transitions)
        diff_surrogate.fit(dataset_hat, epochs=config.surrogate_epochs)

        # 测试差分模型效果
        _plot_model_comparison(
            env=env,
            agent=agent,
            reward_cfg=reward_cfg,
            combined=combined,
            static_surrogate=static_surrogate,
            save_path=f"{run_dir}/cycle_{cycle}_comparison.png",
            seed=1000 + cycle,
        )

        # ------------------------------------------------------------
        # (C) Policy training on COMBINED surrogate (same as cycle0 style)
        # ------------------------------------------------------------
        actor_losses: List[float] = []
        critic_losses: List[float] = []

        for epoch in range(config.virtual_episodes_per_cycle):
            # 噪声衰减（和静态 trainer 一样的思想）:contentReference[oaicite:3]{index=3}
            progress = epoch / max(1, config.virtual_episodes_per_cycle - 1)
            current_sigma = max(
                config.noise_sigma_end,
                config.noise_sigma_start
                - (config.noise_sigma_start - config.noise_sigma_end) * progress,
            )

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

                a = float(np.clip(a_det + noise, low, high))
                return np.array([a], dtype=np.float32)

            # 每个 epoch 做多条 surrogate rollout（同构 cycle0 的 policy_rollouts_per_epoch）:contentReference[oaicite:5]{index=5}
            infos_for_plot: List[Dict] | None = None
            reward_for_plot: float = 0.0
            for rollout_idx in range(config.policy_rollouts_per_epoch):
                if rollout_idx == 0:
                    state0, _ = env.reset()
                elif dataset_hat.states.shape[0] > 0:
                    state0 = dataset_hat.states[np.random.randint(0, dataset_hat.states.shape[0])].copy()
                else:
                    state0, _ = env.reset()
                total_reward, infos = rollout_surrogate(
                    state=state0,
                    surrogate=static_surrogate.predict,
                    policy=policy_train,
                    horizon=env.max_steps,
                    reward_cfg=reward_cfg,
                    dt=env.dt,
                    v_max=env.v_max,
                    t_max=env.t_max,
                )

                if rollout_idx == 0:
                    infos_for_plot = infos
                    reward_for_plot = float(total_reward)
                    print(
                        "epoch", epoch,
                        "R", round(total_reward, 2),
                        "SOC_end", round(infos[-1]["SOC_pack"], 4),
                        "Vmax", round(max(i["V_cell_max"] for i in infos), 4),
                        "Tmax", round(max(i["T_cell_max"] for i in infos), 2),
                        "I_mean", round(np.mean([i["I"] for i in infos]), 3),
                        "violations:", sum(1 for x in infos if x.get("violation", False)),
                    )

                # 用 surrogate infos 喂给 agent（完全复用 cycle0 的口径）
                # 并按你的训练语义：每个 step 做一次梯度更新（off-policy）
                for t in range(len(infos) - 1):
                    curr_info = infos[t]
                    next_info = infos[t + 1]

                    s = obs_from_info(curr_info)
                    s_next = obs_from_info(next_info)

                    a_val = float(next_info["I"])
                    a = np.array([a_val], dtype=np.float32)
                    r = float(next_info["reward"])

                    is_violation = bool(next_info.get("violation", False))
                    is_last = (t == len(infos) - 2)
                    done = is_violation or is_last

                    agent.observe(s, a, r, s_next, done)

                    if (not _is_on_policy_agent(agent)) and _agent_ready_to_update(agent):
                        a_loss, c_loss = agent.update()
                        if a_loss is not None:
                            actor_losses.append(float(a_loss))
                        if c_loss is not None:
                            critic_losses.append(float(c_loss))
                if _is_on_policy_agent(agent) and _agent_ready_to_update(agent):
                        a_loss, c_loss = agent.update()
                        if a_loss is not None:
                            actor_losses.append(float(a_loss))
                        if c_loss is not None:
                            critic_losses.append(float(c_loss))
            if infos_for_plot is None:
                infos_for_plot = infos
                reward_for_plot = float(total_reward)
            # ---- logging / plotting（同构 cycle0）----
            metrics = summarize_episode(infos_for_plot)
            metrics.update(
                {
                    "epoch": epoch,
                    "cycle": cycle,
                    "phase": "adaptive",
                    "reward": float(reward_for_plot),
                    "sigma": float(current_sigma),
                    "a_loss": float(np.mean(actor_losses)) if actor_losses else 0.0,
                    "c_loss": float(np.mean(critic_losses)) if critic_losses else 0.0,
                }
            )
            log_metrics(f"{run_dir}/metrics.jsonl", metrics)

            if config.plot_interval > 0 and epoch % config.plot_interval == 0:
                step_count = len(infos_for_plot)
                end_reason = infos[-1].get("terminated_reason", "unknown")
                print(f"[Plot Debug] 准备画图: 包含 {step_count} 步，终止原因: {end_reason}")
                # ====== 2. 防止单步崩溃导致空图 ======
                if step_count < 3:
                    print(" -> [警告] 步数太少，跳过画图以防空白！这说明 Agent 在起点就挂了。")
                else:
                    # ====== 3. 正常画图 ======
                    curve = curve_from_infos(infos)
                    plot_episode(curve, f"{run_dir}/cycle_{cycle}_epoch_{epoch}.png")
             
            all_metrics.append(metrics)

    return all_metrics
