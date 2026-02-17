"""Trainer implementing Algorithm 2 with adaptive differential surrogate (project-consistent)."""
from __future__ import annotations

from dataclasses import dataclass
from logging import config
from typing import Dict, List

import numpy as np
import matplotlib.pyplot as plt
from sympy import frac
from ...rewards.paper_reward import reward_from_info
from ...envs.base_env import BasePackEnv
from ...evaluation.episode_rollout import rollout_env, rollout_surrogate
from ...evaluation.reports import summarize_episode
from ...rewards.paper_reward import PaperRewardConfig
from ...surrogate.dataset import build_dataset
from ...surrogate.gp_combined import CombinedSurrogate
from ...surrogate.gp_differential import DifferentialSurrogate
from ...surrogate.gp_static import StaticSurrogate
from ...utils.logging import log_metrics
from ...envs.observables import curve_from_infos
from ...utils.plotting import plot_episode
from ..actor_critic_ddpg import DDPGAgent  # 和静态 trainer 保持一致

def validate_with_combined_model(env, agent, static_surrogate, combined_model, hold_steps=1):
    """
    闭环验证：对比真实环境、纯静态模型、以及封装好的 Combined 代理模型。
    """
    print("开始闭环验证：对比静态模型与 Combined 修正模型...")

    # --- 1. 真实环境运行 (基准) ---
    real_traj = {'SOC': [], 'V': [], 'T': [], 'I': [], 'full_states': []}
    state_real, _ = env.reset(seed=123)
    done_real = False
    
    while not done_real:
        raw_action = float(agent.act(state_real)[0])
        real_traj['SOC'].append(state_real[0])
        real_traj['V'].append(state_real[2])
        real_traj['T'].append(state_real[4])
        real_traj['I'].append(raw_action)
        real_traj['full_states'].append(state_real.copy())

        state_real, _, done_real, truncated, _ = env.step(np.array([raw_action]))
        if done_real or truncated: break

    # --- 2. 纯静态代理模型闭环 (红色虚线) ---
    static_traj = {'SOC': [], 'V': [], 'T': [], 'I': []}
    state_init = real_traj['full_states'][0].copy()
    curr_static_state = state_init.copy()
    
    for _ in range(len(real_traj['SOC'])):
        a = float(agent.act(curr_static_state)[0])
        static_traj['SOC'].append(curr_static_state[0])
        static_traj['V'].append(curr_static_state[2])
        static_traj['T'].append(curr_static_state[4])
        static_traj['I'].append(a)

        # 仅使用静态模型增量进行演化
        d_static, _ = static_surrogate.predict(curr_static_state, np.array([a]))
        curr_static_state[:6] += d_static
        curr_static_state[6] = a 
        if curr_static_state[0] >= 1.05: break

    # --- 3. Combined 代理模型闭环 (蓝色实线) ---
    # 调用你封装好的 combined.predict，内部已包含 static + diff
    combined_traj = {'SOC': [], 'V': [], 'T': [], 'I': []}
    curr_comb_state = state_init.copy()
    
    print("正在计算 Combined 模型轨迹...")
    for _ in range(len(real_traj['SOC'])):
        a = float(agent.act(curr_comb_state)[0])
        combined_traj['SOC'].append(curr_comb_state[0])
        combined_traj['V'].append(curr_comb_state[2])
        combined_traj['T'].append(curr_comb_state[4])
        combined_traj['I'].append(a)

        # 直接调用你的组合模型接口
        d_combined, _ = combined_model.predict(curr_comb_state, np.array([a]))
        
        # 演化状态
        curr_comb_state[:6] += d_combined
        curr_comb_state[6] = a
        
        # 边界保护
        if curr_comb_state[0] >= 1.05: break

    # --- 4. 绘图比较 ---
    fig, axes = plt.subplots(4, 1, figsize=(10, 16), sharex=True)
    time_step = hold_steps * 10.0
    keys = ['SOC', 'V', 'T', 'I']
    ylabels = ['SOC', 'Voltage (V)', 'Temperature (K)', 'Current (A)']
    
    for i in range(4):
        ax = axes[i]
        key = keys[i]
        
        # 1. 真实 (黑实线)
        t_real = np.arange(len(real_traj[key])) * time_step
        ax.plot(t_real, real_traj[key], 'k-', label='Real Simulator', linewidth=2.0)
        
        # 2. 纯静态 (红虚线)
        t_static = np.arange(len(static_traj[key])) * time_step
        ax.plot(t_static, static_traj[key], 'r--', label='Static Surrogate', alpha=0.7)
        
        # 3. Combined 修正 (蓝实线)
        t_comb = np.arange(len(combined_traj[key])) * time_step
        ax.plot(t_comb, combined_traj[key], 'b-', label='Combined (Static + Diff)', linewidth=1.5)
            
        ax.set_ylabel(ylabels[i])
        ax.grid(True, linestyle=':', alpha=0.6)
        if i == 0: 
            ax.legend(loc='upper left')

    axes[-1].set_xlabel('Time (seconds)')
    plt.tight_layout()
    plt.show()

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


@dataclass
class AdaptiveConfig:
    """Configuration for adaptive training cycles (aligned with Cycle0Config style)."""

    cycles: int
    real_episodes_per_cycle: int
    surrogate_epochs: int
    policy_epochs: int

    # ====== NEW: keep same structure as cycle0 ======
    policy_rollouts_per_epoch: int = 3
    updates_per_epoch: int = 500
    plot_interval: int = 1

    # exploration noise on surrogate rollouts (absolute A)
    noise_sigma_start: float = 2.0
    noise_sigma_end: float = 0.2


def train_adaptive_cycles(
    env: BasePackEnv,
    agent: DDPGAgent,                 # ✅ 新增：用 cycle0 的 agent 来 act/update
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

        for ep in range(config.real_episodes_per_cycle):
            # 重置环境，获取初始状态
            state, info = env.reset()
            done = False
            prev_info = info
            
            # 计算噪声衰减
            frac = ep / max(1, config.real_episodes_per_cycle - 1)
            sigma_real = (1 - frac) * config.noise_sigma_start + frac * config.noise_sigma_end

            while not done:
                # 1. 选择动作并添加噪声
                a_det = float(agent.act(state)[0])
                noise = float(np.random.normal(0.0, sigma_real))
                a_clipped = np.array([np.clip(a_det + noise, low, high)], dtype=np.float32)

                # 记录更新 GP 所需的起点状态
                start_state = state.copy()
                
                # 2. 与真实物理环境交互 [cite: 171]
                next_state, _, terminated, truncated, next_info = env.step(a_clipped)
                
                # 3. 计算奖励 (用于强化学习训练)
                reward = reward_from_info(prev_info, next_info, reward_cfg, env.v_max, env.t_max)
                done = terminated or truncated

                # 4. 【关键：同步更新 Agent】
                # 在真实采样阶段同步更新神经网络参数 
                # agent.observe(state, a_clipped, reward, next_state, done)
                # if len(agent.buffer) > agent.config.batch_size:
                #     agent.update() # 边采集边学习，适应当前老化状态 [cite: 171]

                # 5. 【关键：收集差分模型所需数据】
                # 保存 (s, a, Δs_real)
                # 这里的 delta6 对应真实物理世界的演化结果 [cite: 150]
                delta_real6 = next_state[:6] - start_state[:6]
                transitions.append((start_state, a_clipped, delta_real6))

                # 状态更替
                state = next_state
                prev_info = next_info

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

        # 测试差分模型的效果
        validate_with_combined_model(env, agent, static_surrogate, combined, hold_steps=1)
        # ------------------------------------------------------------
        # (C) Policy training on COMBINED surrogate (same as cycle0 style)
        # ------------------------------------------------------------
        for epoch in range(config.policy_epochs):
            # 噪声衰减（和静态 trainer 一样的思想）:contentReference[oaicite:3]{index=3}
            progress = epoch / max(1, config.policy_epochs - 1)
            current_sigma = max(
                config.noise_sigma_end,
                config.noise_sigma_start
                - (config.noise_sigma_start - config.noise_sigma_end) * progress,
            )

            def policy_train(s: np.ndarray) -> np.ndarray:
                a_det = float(agent.act(s)[0])
                noise = np.random.normal(0.0, current_sigma)

                # 防止 0A 截断导致动作分布畸变（静态 trainer 的技巧）:contentReference[oaicite:4]{index=4}
                if a_det + noise > high:
                    noise = -abs(noise)
                elif a_det + noise < low:
                    noise = abs(noise)

                a = float(np.clip(a_det + noise, low, high))
                return np.array([a], dtype=np.float32)
            
            # 本 epoch 的统计容器（注意：每个 epoch 清空）=====
            epoch_rewards: List[float] = []
            best_reward = -1e18
            best_infos = None
            last_infos = None
            # 每个 epoch 做多条 surrogate rollout（同构 cycle0 的 policy_rollouts_per_epoch）:contentReference[oaicite:5]{index=5}
            for rollout_idx in range(config.policy_rollouts_per_epoch):
                # 起点：第一个从 reset，其它从真实数据分布抽样
                if rollout_idx == 0 or dataset_hat.states.shape[0] == 0:
                    state0, _ = env.reset()
                else:
                    state0 = dataset_hat.states[np.random.randint(0, dataset_hat.states.shape[0])].copy()

                total_reward, infos = rollout_surrogate(
                    state=state0,
                    surrogate=combined.predict,
                    policy=policy_train,
                    horizon=env.max_steps,
                    reward_cfg=reward_cfg,
                    dt=env.dt,
                    v_max=env.v_max,
                    t_max=env.t_max,
                )

                last_infos = infos
                epoch_rewards.append(float(total_reward))
                if float(total_reward) > best_reward:
                    best_reward = float(total_reward)
                    best_infos = infos

                # --- 用 surrogate infos 喂给 agent（与你静态 trainer 同口径）---
                for t in range(len(infos) - 1):
                    curr_info = infos[t]
                    next_info = infos[t + 1]

                    s = obs_from_info(curr_info)
                    s_next = obs_from_info(next_info)

                    a_val = float(next_info["I"])              # 动作 a_t
                    a = np.array([a_val], dtype=np.float32)
                    r = float(next_info["reward"])             # reward 属于执行 a_t 后的结果

                    is_violation = bool(next_info.get("violation", False))
                    is_last = (t == len(infos) - 2)
                    done = is_violation or is_last

                    agent.observe(s, a, r, s_next, done)
            epoch_a_loss: List[float] = []
            epoch_c_loss: List[float] = []
            for _ in range(config.updates_per_epoch):
                if len(agent.buffer) > agent.config.batch_size:
                    a_loss, c_loss = agent.update()
                    # 你的 update() 不会返回 None（不足 batch 时返回 0.0），这里直接记录即可
                    epoch_a_loss.append(float(a_loss))
                    epoch_c_loss.append(float(c_loss))

            use_infos = best_infos if best_infos is not None else last_infos
            assert use_infos is not None, "No rollout infos collected in this epoch."

            # ---- logging / plotting（同构 cycle0）----
            metrics = summarize_episode(use_infos)
            metrics.update(
                {
                    "epoch": epoch,
                    "cycle": cycle,
                    "phase": "adaptive",
                    "sigma": float(current_sigma),
                    "reward_mean": float(np.mean(epoch_rewards)) if epoch_rewards else 0.0,
                    "reward_last": float(epoch_rewards[-1]) if epoch_rewards else 0.0,
                    "reward_best": float(best_reward) if epoch_rewards else 0.0,
                    "a_loss": float(np.mean(epoch_a_loss)) if epoch_a_loss else 0.0,
                    "c_loss": float(np.mean(epoch_c_loss)) if epoch_c_loss else 0.0,
                    "buffer_size": int(len(agent.buffer)),
                }
            )
            log_metrics(f"{run_dir}/metrics.jsonl", metrics)

            if config.plot_interval > 0 and epoch % config.plot_interval == 0:
                curve = curve_from_infos(use_infos)
                plot_episode(curve, f"{run_dir}/cycle_{cycle}_epoch_{epoch}_best.png")

            all_metrics.append(metrics)
         # 测试差分模型的效果
        validate_with_combined_model(env, agent, static_surrogate, combined, hold_steps=1)
    return all_metrics
