"""Trainer implementing Algorithm 1 (Cycle0) with static surrogate."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Tuple
import csv
import torch
import numpy as np
import pandas as pd
import pybamm
import time
from scipy.interpolate import interp1d

from torch import clip

from ...envs.base_env import BasePackEnv
from ...evaluation.episode_rollout import rollout_env, rollout_surrogate
from ...evaluation.reports import summarize_episode
from ...rewards.paper_reward import PaperRewardConfig, compute_paper_reward
from ...surrogate.dataset import build_dataset
from ...surrogate.gp_static import StaticSurrogate
from ..noise import GaussianNoise
from ...utils.logging import log_metrics
from ...envs.observables import curve_from_infos
from ...utils.plotting import plot_episode
import matplotlib.pyplot as plt


def _is_on_policy_agent(agent: Any) -> bool:
    return bool(getattr(agent, "is_on_policy", False))


def _agent_buffer_len(agent: Any) -> int:
    buffer_obj = getattr(agent, "buffer", None)
    return len(buffer_obj) if buffer_obj is not None else 0


def _agent_ready_to_update(agent: Any) -> bool:
    batch_size = int(getattr(getattr(agent, "config", object()), "batch_size", 1))
    return _agent_buffer_len(agent) >= batch_size


def validate_full_charge_comparison(env, agent, surrogate, hold_steps=1, align_interval=50):
    """
    完全修复版：包含电流(I)的完整记录和绘图
    """
    print(f"开始闭环验证（包含 {align_interval} 步对齐参考）...")

    # --- 1. 真实环境闭环运行 ---
    # 初始化包含 'I'
    real_traj = {'SOC': [], 'V': [], 'T': [], 'I': [], 'full_states': []}
    state_real, info = env.reset(seed=123)
    done_real = False
    
    print("正在记录真实环境轨迹...")
    while not done_real:
        raw_action = float(agent.act(state_real)[0])
        action = np.array([raw_action], dtype=np.float32)
        
        # 记录
        real_traj['SOC'].append(state_real[0])
        real_traj['V'].append(state_real[2])
        real_traj['T'].append(state_real[4])
        real_traj['I'].append(raw_action) # 记录真实电流
        real_traj['full_states'].append(state_real.copy())

        for _ in range(hold_steps):
            state_real, reward, done_real, truncated, info = env.step(action)
            if done_real or truncated: break
        if done_real or truncated: break

    # --- 2. 纯代理模型闭环运行 (修复：增加电流记录) ---
    pred_traj = {'SOC': [], 'V': [], 'T': [], 'I': []} # <--- 关键修复：初始化 'I'
    state_init, _ = env.reset(seed=123)
    curr_pred_state = state_init.copy()
    
    print("正在记录纯代理模型轨迹...")
    # 使用真实轨迹长度作为参考
    for _ in range(len(real_traj['SOC'])):
        # 1. 决策
        raw_action_pred = float(agent.act(curr_pred_state)[0])
        
        # 2. 记录 (现在包含 I 了)
        pred_traj['SOC'].append(curr_pred_state[0])
        pred_traj['V'].append(curr_pred_state[2])
        pred_traj['T'].append(curr_pred_state[4])
        pred_traj['I'].append(raw_action_pred) # <--- 关键修复：记录预测电流

        # 3. 演化
        pred_delta, _ = surrogate.predict(curr_pred_state, np.array([raw_action_pred]))
        curr_pred_state[:6] += pred_delta
        curr_pred_state[6] = raw_action_pred 
        
        # 简单的物理边界保护，防止画图报错
        if curr_pred_state[0] >= 1.05 or curr_pred_state[2] >= 4.5: 
            break

    # --- 3. 强制对齐演化 (修复：增加电流记录) ---
    aligned_traj = {'SOC': [], 'V': [], 'T': [], 'I': []} # <--- 关键修复
    curr_align_state = state_init.copy()
    
    print(f"正在记录每 {align_interval} 步对齐一次的轨迹...")
    for i in range(len(real_traj['SOC'])):
        # 1. 决策
        raw_act_align = float(agent.act(curr_align_state)[0])

        # 2. 记录
        aligned_traj['SOC'].append(curr_align_state[0])
        aligned_traj['V'].append(curr_align_state[2])
        aligned_traj['T'].append(curr_align_state[4])
        aligned_traj['I'].append(raw_act_align) # <--- 关键修复

        # 3. 演化
        delta, _ = surrogate.predict(curr_align_state, np.array([raw_act_align]))
        curr_align_state[:6] += delta
        curr_align_state[6] = raw_act_align

        # 4. 强制对齐
        if (i + 1) % align_interval == 0 and (i + 1) < len(real_traj['full_states']):
            curr_align_state = real_traj['full_states'][i+1].copy()

    # --- 4. 结果可视化 ---
    fig, axes = plt.subplots(4, 1, figsize=(10, 15), sharex=True)
    time_step = hold_steps * 10.0

    keys = ['SOC', 'V', 'T', 'I']
    ylabels = ['SOC', 'Voltage (V)', 'Temperature (K)', 'Current (A)']
    
    for i in range(4):
        ax = axes[i]
        key = keys[i]
        
        # 1. 真实 (黑实线)
        t_real = np.arange(len(real_traj[key])) * time_step
        ax.plot(t_real, real_traj[key], 'k-', label='Real Simulator', linewidth=2.0)
        
        # 2. 纯代理 (红虚线)
        t_pred = np.arange(len(pred_traj[key])) * time_step
        ax.plot(t_pred, pred_traj[key], 'r--', label='Pure Surrogate (Drift)', alpha=0.7)
        
        # 3. 对齐参考 (绿点线)
        t_align = np.arange(len(aligned_traj[key])) * time_step
        ax.plot(t_align, aligned_traj[key], 'g:', label=f'Aligned (Every {align_interval} steps)', linewidth=1.5)
            
        ax.set_ylabel(ylabels[i])
        ax.grid(True, linestyle=':', alpha=0.6)
        
        if i == 0: 
            ax.legend(loc='upper left')

    axes[-1].set_xlabel('Time (seconds)')
    plt.tight_layout()
    plt.show()

def validate_surrogate_rollout(env, agent, surrogate, dataset, steps=5000, hold_steps=1):
    """
    对比 真实环境 vs 代理模型 的多步演化
    """
    print("开始验证代理模型滚动预测精度...")
    
    # 1. 从环境获取一个初始状态
    state, info = env.reset()

    #使用大电流测试
    test_current = -8.0 
    action = np.array([test_current], dtype=np.float32)
    
    print(f"测试条件: 电流 {test_current}A, 每步时长 {hold_steps*10}s")

    # 记录真实轨迹和预测轨迹
    real_traj = {'V': [], 'T': [], 'SOC': []}
    pred_traj = {'V': [], 'T': [], 'SOC': []}
    
    # 记录初始点
    # state 索引: 0:SOC, 2:V, 4:T
    real_traj['SOC'].append(state[0])
    real_traj['V'].append(state[2])   
    real_traj['T'].append(state[4])   
    
    pred_traj['SOC'].append(state[0])
    pred_traj['V'].append(state[2])
    pred_traj['T'].append(state[4])

    # 代理模型的当前状态初始化为真实初始状态
    curr_pred_state = state.copy()
    
    # 2. 运行 N 步
    for i in range(steps):
        # --- A. 真实环境演化 ---
        # 真实环境必须连续跑 hold_steps 步，才能和代理模型的 1 步对齐
        for _ in range(hold_steps):
            next_state_real, _, done, _, next_info_real = env.step(action)
            if done: break

        # --- B. 代理模型演化 ---
        # 代理模型本身就是基于 50s 间隔训练的，所以只跑 1 步
        pred_delta, pred_std = surrogate.predict(curr_pred_state, action)
        
        # 更新预测状态
        next_state_pred = curr_pred_state.copy()
        next_state_pred[:6] += pred_delta # 更新前6维
        next_state_pred[6] = float(action[0]) # 更新 I_prev
        
        # --- 记录数据 ---
        real_traj['SOC'].append(next_state_real[0])
        real_traj['V'].append(next_state_real[2]) 
        real_traj['T'].append(next_state_real[4]) 
        
        pred_traj['SOC'].append(next_state_pred[0])
        pred_traj['V'].append(next_state_pred[2])
        pred_traj['T'].append(next_state_pred[4])
        
        # 这一步很关键：下一轮预测基于上一轮的【预测值】，而不是真实值
        curr_pred_state = next_state_pred
        # env.step()会自动更新
        # state = next_state_real
        
        if done:
            print(f"真实环境在第 {i} 个大步提前结束 (触发约束)")
            break

    # 3. 画图
    fig, axes = plt.subplots(3, 1, figsize=(8, 10))
    
    # 计算时间轴 (秒)
    time_axis = np.arange(len(real_traj['SOC'])) * (hold_steps * 10.0)
    
    ax = axes[0]
    ax.plot(time_axis, real_traj['SOC'], 'k-', label='Real Simulator', linewidth=2)
    ax.plot(time_axis, pred_traj['SOC'], 'r--', label='Surrogate Model', linewidth=2)
    ax.set_ylabel('SOC')
    ax.legend()
    ax.set_title(f'Validation @ {test_current}A (Time Aligned)')

    ax = axes[1]
    ax.plot(time_axis, real_traj['V'], 'k-', linewidth=2)
    ax.plot(time_axis, pred_traj['V'], 'r--', linewidth=2)
    ax.set_ylabel('Voltage (V)')

    ax = axes[2]
    ax.plot(time_axis, real_traj['T'], 'k-', linewidth=2)
    ax.plot(time_axis, pred_traj['T'], 'r--', linewidth=2)
    ax.set_ylabel('Temperature (K)')
    ax.set_xlabel('Time (seconds)')
    
    plt.tight_layout()
    plt.show()

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

def save_transitions_with_episode_to_csv(
    transitions: List[Tuple[np.ndarray, np.ndarray, np.ndarray]],
    episode_ids: List[int],
    filepath: str,
) -> None:
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

    hold_steps: int = 1              # dt=10s
    v_soft_max: float = 4.17   # 或 env.v_max - 0.03
    t_soft_max: float = 309.5  # 或 env.t_max - 1.5

def get_chen2020_ocv_func():
    # 1. 加载参数
    param = pybamm.ParameterValues("Chen2020")

    # 2. 从你提供的列表中锁定的准确键名
    U_p = param["Positive electrode OCP [V]"]
    U_n = param["Negative electrode OCP [V]"]

    # 获取锂离子计量比限制 (用于定义 SOC 0% 到 100%)
    # 如果下述四个键名报错，通常是因为 PyBaMM 版本中对极组容量限制的定义不同
    # 你可以先尝试以下标准 Chen2020 键名：
    try:
        sto_n_0 = param["Lower stoichiometric limit in negative electrode"]
        sto_n_1 = param["Upper stoichiometric limit in negative electrode"]
        sto_p_0 = param["Upper stoichiometric limit in positive electrode"]
        sto_p_1 = param["Lower stoichiometric limit in positive electrode"]
    except KeyError:
        # 如果报错，请直接手动指定 Chen2020 的典型计量比（LG M50 电池）
        sto_n_0, sto_n_1 = 0.0279, 0.9014
        sto_p_0, sto_p_1 = 0.9077, 0.2661

    # 3. 构建 SOC 映射
    soc_range = np.linspace(0, 1, 100)
    ocv_values = []

    for soc in soc_range:
        curr_sto_n = sto_n_0 + soc * (sto_n_1 - sto_n_0)
        curr_sto_p = sto_p_0 - soc * (sto_p_0 - sto_p_1)

        # OCV = U_p(sto_p) - U_n(sto_n)
        v = param.evaluate(U_p(pybamm.Scalar(curr_sto_p))) - \
            param.evaluate(U_n(pybamm.Scalar(curr_sto_n)))
        ocv_values.append(float(v))

    return interp1d(soc_range, ocv_values, kind='linear', fill_value="extrapolate")
"""
真实环境采样的关键逻辑 
"""
def collect_real_data(
    env: BasePackEnv,
    reward_cfg: PaperRewardConfig,
    agent: Any,
    config: Cycle0Config,
) ->  Tuple[List[tuple[np.ndarray, np.ndarray, np.ndarray]], List[int]]:
    # 1.准备工作
    low = float(agent.config.action_low)   # -30
    high = float(agent.config.action_high) # 0
    transitions: List[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
    transition_episode_ids: List[int] = []

    # 软约束参数
    v_soft = float(getattr(config, "v_soft_max", env.v_max - 0.03))  # 比硬约束低一点，比如 4.17
    t_soft = float(getattr(config, "t_soft_max", env.t_max - 1.5))   # 比硬约束低一点，比如 318.5K

    get_ocv = get_chen2020_ocv_func()
    def physics_limit_action(action_from_agent: float, info: dict) -> float:
        """
        取代旧的 guard_action，基于状态估计计算物理安全电流阈值 [cite: 231, 235]
        """
        soc_curr = float(info.get("SOC_cell_max", info.get("SOC_pack")))
        v_curr = float(info.get("V_cell_max"))

        # 1. 计算当前 OCV
        u_ocv = float(get_ocv(soc_curr))

        # 2. 设定物理边界：截止电压 4.2V，估算内阻 0.025 Ohm
        v_limit = 4.2 
        r_internal = 0.025 

        # 3. 计算物理安全电流边界 I_bound (根据论文公式 6-13) [cite: 238]
        # I_bound = (U_cut - U_ocv) / R
        i_bound_single = max(0.0, (v_limit - u_ocv) / r_internal)

        # 4. 适配 3p6s 电池组总电流 (充电为负值)
        i_bound_pack = i_bound_single * 3.0
        # print(f"[DEBUG] SOC={soc_curr:.4f}, OCV={u_ocv:.3f}V, I_bound_single={i_bound_single:.2f}A, I_bound_pack={i_bound_pack:.2f}A")

        # 5. 动作裁剪：Agent 可以尝试物理极限内的任何电流，但禁止导致瞬时超压 [cite: 243, 272]
        safe_action = np.clip(action_from_agent, -i_bound_pack, 0.0)

        return float(safe_action)
    
    print(f"开始 Model-Free 预训练 (Warm-up) 共 {config.real_episodes} 回合...")

    for ep in range(config.real_episodes):
        # 2. 噪声衰减 (模仿训练过程)
        frac = ep / max(1, config.real_episodes - 1)
        # 预训练阶段可以使用较大的噪声，鼓励探索
        sigma = ((1 - frac) * config.noise_sigma_start + frac * config.noise_sigma_end) * (high - low)
        noise = GaussianNoise(sigma=float(sigma))

        # state, info = env.reset()
        state, info = env.reset(options={"soc_low": 0.1, "soc_high": 0.9})
        done = False

        # 统计本回合奖励
        ep_reward = 0.0
        current_soc = float(info.get("SOC_pack", state[0]))
        while not done:
            # --- A. 选择动作 ---
            raw_action = float(agent.act(state)[0])
            if _is_on_policy_agent(agent):
                a_noisy = raw_action
            else:
                a_noisy = raw_action + noise.sample()
            a_clipped = float(np.clip(a_noisy, low, high))
            #    B.安全守卫（用上一时刻 info）
            safe_action = physics_limit_action(a_clipped, info)
            safe_action = float(np.clip(safe_action, low, high))

            # 准备执行的动作
            action_to_exec = np.array([safe_action], dtype=np.float32)

            #   C.环境交互
            start_state = state.copy()
            accumulated_reward = 0.0

            # 执行 hold steps (模拟宏观步长) (Time = t -> t + N*dt) 10s
            for _ in range(config.hold_steps):
                # 记录 step 前的 SOC
                prev_soc = current_soc
                # === [新增计时逻辑] 2. 精确记录单步仿真耗时 ===
                # t0 = time.perf_counter()  # 记录开始时间
                next_state, _, terminated, truncated, next_info = env.step(action_to_exec)
                # print(f"仿真环境运行一次 (env.step) 所需时间: {(time.perf_counter() - t0) * 1000:.2f} 毫秒")
                # 更新当前的 SOC, V, T 等信息
                current_soc = float(next_info["SOC_pack"])
                v_max = float(next_info["V_cell_max"])
                t_max = float(next_info["T_cell_max"])
                # 尝试获取 std_soc，如果没有则设为 0
                std_soc = float(next_info.get("std_SOC", 0.0)) 
                # 获取真实的电流
                I_exec = float(next_info.get("I_pack_true", safe_action))
                if abs(I_exec - safe_action) > 1e-3:
                    print(f"[WARN] current mismatch: safe={safe_action}, exec={I_exec}")

                # 手动计算这一小步的奖励
                # 注意：这里调用你外部定义的 compute_paper_reward
                r_step, _, _, _, _, _, _ = compute_paper_reward(
                    soc_prev=prev_soc,
                    soc_next=current_soc,
                    v_max_next=v_max,
                    t_max_next=t_max,
                    std_soc_next=std_soc,
                    action_current=I_exec, # 传入实际执行的电流
                    v_limit=env.v_max,
                    t_limit=env.t_max,
                    config=reward_cfg 
                )
                # 累积奖励
                accumulated_reward += r_step

                # 状态更新
                state = next_state
                info = next_info
               
                # 如果中途挂了（过压/时间到），立即停止，保留当前的 state 用于计算 Delta
                if terminated or truncated:
                    done = True
                    break
            # --- 3. 训练 Agent ---
            # 存入的是：(0s状态, 10s动作, 10s总奖励, 10s状态)
            agent.observe(start_state, action_to_exec, accumulated_reward, state, done)
            # 只有当 buffer 数据够了才 update
            if (not _is_on_policy_agent(agent)) and _agent_ready_to_update(agent):
                agent.update()
            # # --- 4. 收集数据给 GP ---
            # Delta = State(t + 50s) - State(t)
            # 这样算出来的 温度 Delta 会比之前大 5 倍以上！
            final_delta = state[:6] - start_state[:6]
            # 存入 dataset 的是：在 start_state 下，执行 action，导致了 final_delta 的变化
            transitions.append((start_state.copy(), action_to_exec.copy(), final_delta.copy()))
            transition_episode_ids.append(ep + 1)
            ep_reward += accumulated_reward
        # 打印日志
        if (ep + 1) % 5 == 0:
            print(
                f"[Warmup] Ep {ep+1} | R: {ep_reward:.2f} | "
                f"SOC: {info['SOC_pack']:.4f} | Vmax: {info['V_cell_max']:.4f} | "
                f"Buf: {_agent_buffer_len(agent)}"
            )
    if _is_on_policy_agent(agent) and _agent_ready_to_update(agent):
        agent.update()
    agent.save(str("runs//cycle0//policy_start.pt"))
    return transitions,  transition_episode_ids

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

def verify_dataset_coverage(transitions):
    """
    验证 transitions 数据集的覆盖情况
    transitions 结构: [(start_state, action_to_exec, final_delta), ...]
    其中 start_state[0] 是 SOC
    """
    # 1. 提取数据
    # 假设 start_state 的第一个元素是 SOC，action_to_exec 是一个包含电流的 array
    soc_values = np.array([t[0][0] for t in transitions])
    actions = np.array([t[1][0] for t in transitions])

    # 2. 基础统计
    high_soc_mask = soc_values > 0.6
    num_high_soc = np.sum(high_soc_mask)
    total_samples = len(transitions)

    print("-" * 30)
    print(f"数据集总量: {total_samples}")
    print(f"SOC > 0.6 的样本数: {num_high_soc} (占比: {num_high_soc/total_samples*100:.2f}%)")

    if num_high_soc > 0:
        high_soc_actions = actions[high_soc_mask]
        print(f"SOC > 0.6 时的电流区间: [{np.min(high_soc_actions):.2f}A, {np.max(high_soc_actions):.2f}A]")
        print(f"SOC > 0.6 时的平均电流: {np.mean(high_soc_actions):.2f}A")

        # 统计大电流样本数（例如电流幅值 > 10A，假设充电电流为负）
        large_current_high_soc = np.sum(high_soc_actions < -10.0)
        print(f"SOC > 0.6 且电流 > 10A 的样本数: {large_current_high_soc}")
    else:
        print("警告：数据集中完全没有 SOC > 0.6 的数据！")

    # 3. 可视化分析
    plt.figure(figsize=(15, 6))

    # 图 1: SOC 分布直方图
    plt.subplot(1, 2, 1)
    plt.hist(soc_values, bins=50, color='skyblue', edgecolor='black', alpha=0.7)
    plt.axvline(0.6, color='red', linestyle='--', label='SOC=0.6 边界')
    plt.title('SOC 数据分布密度')
    plt.xlabel('SOC')
    plt.ylabel('频数')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # 图 2: 电流 vs SOC 散点图 (这是验证“有效性”的关键)
    plt.subplot(1, 2, 2)
    plt.scatter(soc_values, actions, alpha=0.4, s=15, c='darkblue')
    plt.axvline(0.6, color='red', linestyle='--', label='SOC=0.6 边界')
    plt.title('电流动作在不同 SOC 下的覆盖情况')
    plt.xlabel('SOC')
    plt.ylabel('电流 Action (A)')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('dataset_coverage_check.png')
    plt.show()

def load_transitions_from_csv(filepath: str) -> List[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    df = pd.read_csv(filepath)
    
    # 自动识别列
    state_cols = [c for c in df.columns if c.startswith('s_')]
    action_cols = ['action']
    delta_cols = [c for c in df.columns if c.startswith('d_')]
    
    transitions = []
    for _, row in df.iterrows():
        state = row[state_cols].values.astype(np.float32)
        action = row[action_cols].values.astype(np.float32)
        delta = row[delta_cols].values.astype(np.float32)
        transitions.append((state, action, delta))
        
    print(f"[OK] Loaded {len(transitions)} transitions from: {filepath}")
    return transitions

def train_cycle0(
    env: BasePackEnv,
    agent: Any,
    reward_cfg: PaperRewardConfig,
    surrogate: StaticSurrogate,
    config: Cycle0Config,
    run_dir: str,
) -> Dict[str, float]:
    
    """Run Cycle0 training pipeline."""
    # 1) 真实环境采集
    # transitions1 = collect_real_data(env, reward_cfg, agent, config)
    transitions, transition_episode_ids = collect_real_data(env, reward_cfg, agent, config)
    # verify_dataset_coverage(transitions)
    # save_transitions_to_csv(transitions, "dataset.csv")
    # save_transitions_with_episode_to_csv(transitions, transition_episode_ids, "dataset_with_episode.csv")
    
    # actions = np.array([a[0] for _, a, _ in transitions], dtype=float)
    # print("Action stats: mean", actions.mean(), "std", actions.std(), "min", actions.min(), "max", actions.max())
    # csv_path = "C:/Users/lht64/Desktop/fast-charging-EV/dataset.csv"
   
    # transitions2 = load_transitions_from_csv(csv_path)
    # transitions = transitions1 + transitions2

    # ckpt_path = "C:/Users/lht64/Desktop/fast-charging-EV/runs/cycle0/policy_start.pt"
    # agent.load(str(ckpt_path))
    # 2) 静态代理模型训练
    dataset = build_dataset(transitions)
    print(f"开始训练静态代理模型，共 {config.surrogate_epochs} epochs...")
    t_train_start = time.perf_counter() # <--- 记录训练开始时间

    surrogate.fit(dataset, epochs=config.surrogate_epochs)

    t_train_end = time.perf_counter()   # <--- 记录训练结束时间
    print(f"静态代理模型训练完成！耗时: {t_train_end - t_train_start:.2f} 秒")
    # surrogate.fit(dataset, epochs=config.surrogate_epochs)
    print("静态代理模型训练完成。")
    
    # 3) 测试 surrogate 训练效果
    # validate_surrogate_rollout(env, agent, surrogate, dataset)
    # 3) N-step 误差评估（用 agent 跑真实轨迹，然后 surrogate 复现）
    # validate_full_charge_comparison(env, agent, surrogate)
    # 4) 用静态代理训练 RL 策略
    low = float(agent.config.action_low)
    high = float(agent.config.action_high)
    actor_losses = []
    critic_losses = []
   
    print(f"训练开始前 Buffer 大小: {_agent_buffer_len(agent)}")

    for epoch in range(config.policy_epochs):
        

        # --- 1. 动态计算噪声标准差 (Sigma Decay) ---
        # 随着训练进行，噪声从大变小
        # 进度 progress: 0.0 -> 1.0
        progress = epoch / max(1, config.policy_epochs - 1)

        # 设定噪声范围：
        # start=2.0A (约10%): 依然保持一定的探索能力
        # end=0.1A (极小): 后期几乎就是确定性策略，为了稳定收敛
        sigma_start = 2.0
        sigma_end = 0.1
        current_sigma = max(sigma_end, sigma_start - (sigma_start - sigma_end) * progress)
        
        def policy_train(s: np.ndarray) -> np.ndarray:
            """ 
            定义带噪声的策略:不能用高斯：actor 初始输出接近 0
            噪声如果为正 → a_noisy > 0 → clip → 0 50% 的动作都变成 0A
            噪声为负 → 才会出现负电流"""
            # A. 获取 Agent 的建议动作 (这是经过 Warm-up 训练的聪明动作)
            # 此时 Agent 可能输出 -10A 左右
            a_det = float(agent.act(s)[0])
         
            # B. 生成高斯噪声
            if _is_on_policy_agent(agent):
                noise = 0.0
            else:
                noise = np.random.normal(0, current_sigma)

            # C. 【可选优化】防止 0A 截断的非对称噪声技巧
            # 如果当前动作已经很接近 0 (比如 > -2A)，且噪声是正的，这会导致结果 > 0 被截断
            # 我们强制反转噪声方向，让它往负方向探索
            if a_det + noise > high:
                noise = -abs(noise)
            # 如果 (动作+噪声) 低于下界 -30，强制反转，向正方向探索
            elif a_det + noise < low:
                noise = abs(noise)
            # D. 叠加并裁剪
            a_noisy = a_det + noise
            a_final = float(np.clip(a_noisy, low, high))
            return np.array([a_final], dtype=np.float32)
        
        def policy_cc(s: np.ndarray) -> np.ndarray:
                return np.array([-10.0], dtype=np.float32)
        
        # 临时列表记录本 epoch 的 loss
        epoch_a_loss = []
        epoch_c_loss = []
        for rollout_idx in range(config.policy_rollouts_per_epoch):
            # 起点：第一个从 reset，后面从真实数据分布抽样
            if rollout_idx == 0:
                state0, _ = env.reset()
            else:
                state0 = dataset.states[np.random.randint(0, dataset.states.shape[0])].copy()
            
            total_reward, infos = rollout_surrogate(
                state=state0,
                surrogate=surrogate.predict,
                policy=policy_train,
                horizon=env.max_steps,
                reward_cfg=reward_cfg,
                dt=env.dt * config.hold_steps,
                v_max=env.v_max,
                t_max=env.t_max,
            )
            # print(
            #     "[CC] R", round(total_reward, 2),
            #     "SOC_end", round(infos[-1]["SOC_pack"], 4),
            #     "Vmax", round(max(i["V_cell_max"] for i in infos), 4),
            #     "Tmax", round(max(i["T_cell_max"] for i in infos), 2),
            #     "I_mean", round(np.mean([i["I"] for i in infos[:-1]]), 3),
            #     "violations:", sum(1 for x in infos if x.get("violation", False)),
            #     "reason:", infos[-1]["terminated_reason"],
            # )
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
                metrics.update({"epoch": epoch, "phase": "cycle0", "reward": total_reward})
                log_metrics(f"{run_dir}/metrics.jsonl", metrics)

                if config.plot_interval > 0 and epoch % config.plot_interval == 0:
                    curve = curve_from_infos(infos)
                    plot_episode(curve, f"{run_dir}/episode_{epoch}.png")

            zero_cnt = 0
            total_cnt = 0
            a_min, a_max = 1e9, -1e9
            a_sum, a_sq = 0.0, 0.0

            for t in range(len(infos) - 1):
                # Current Info (包含 State t 和 Action t)
                curr_info  = infos[t]
                # Next Info (包含 State t+1, 和 Transition Reward)
                next_info = infos[t + 1]

                # 1) State & Next State
                s = obs_from_info(curr_info)
                s_next = obs_from_info(next_info)

                # 2) Action 
                a_val = float(next_info["I"])
                a = np.array([a_val], dtype=np.float32)

                # 3) Reward 
                # Reward 是动作产生的结果，所以存在 next_info 里
                r = float(next_info["reward"])

                # 4) Done 信号 【重要修改】
                # 如果 next_info 标记了 violation，那么这一步就是 Done
                is_violation = next_info["violation"]
                is_last_step_in_list = (t == len(infos) - 2)
                done = False
                if is_violation:
                    done = True # 撞墙死
                elif is_last_step_in_list:
                    done = True # 时间到或SOC满导致结束


                agent.observe(s, a, r, s_next, done)

                # --- 统计 actions 分布 ---
                total_cnt += 1
                if abs(a_val) < 1e-8:
                    zero_cnt += 1
                a_min = min(a_min, a_val)
                a_max = max(a_max, a_val)
                a_sum += a_val
                a_sq += a_val * a_val

            # --- for t 循环结束后（每个 rollout 打印一次）---
            mean = a_sum / max(1, total_cnt)
            var = a_sq / max(1, total_cnt) - mean**2
            std = (var if var > 0 else 0.0) ** 0.5
            print(f"[BUF] actions: zero_ratio={zero_cnt/total_cnt:.3f} mean={mean:.3f} std={std:.3f} min={a_min:.3f} max={a_max:.3f}")

            if _is_on_policy_agent(agent) and _agent_ready_to_update(agent):
                loss_a, loss_c = agent.update()
                epoch_a_loss.append(loss_a)
                epoch_c_loss.append(loss_c)

        # --- 【修改点 3】: 更新并记录 Loss ---
        # 确保 Agent 在每个 epoch 结束时进行多次更新
        for _ in range(config.updates_per_epoch):
            # 只有当 buffer 够大时才 update
            if (not _is_on_policy_agent(agent)) and _agent_ready_to_update(agent):
                # 假设 update 返回 (actor_loss, critic_loss)
                # 如果你的 update 没有返回值，需要去修改 DDPGAgent.update
                loss_a, loss_c = agent.update()
                epoch_a_loss.append(loss_a)
                epoch_c_loss.append(loss_c)
        
        # 打印平均 Loss
        if epoch_a_loss:
            avg_a = np.mean(epoch_a_loss)
            avg_c = np.mean(epoch_c_loss)
            print(f"       -> Loss | Actor: {avg_a:.4f} | Critic: {avg_c:.4f}")
            actor_losses.append(avg_a)
            critic_losses.append(avg_c)
        # for _ in range(config.updates_per_epoch):
        #     agent.update()

    return {"transitions": len(transitions)}
