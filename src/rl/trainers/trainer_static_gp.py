"""Trainer implementing Algorithm 1 (Cycle0) with static surrogate."""
from __future__ import annotations

from dataclasses import dataclass
import os
from typing import Dict, List, Tuple
import csv

import numpy as np
from torch import clip
import torch
import pybamm
from scipy.interpolate import interp1d
from ...envs.base_env import BasePackEnv
from ...evaluation.episode_rollout import rollout_env, rollout_surrogate
from ...evaluation.reports import summarize_episode
from ...rewards.paper_reward import PaperRewardConfig, compute_minimal_reward, compute_paper_reward
from ...surrogate.dataset import build_dataset
from ...surrogate.gp_static import StaticSurrogate
from ..actor_critic_ddpg import DDPGAgent
from ..noise import GaussianNoise
from ...utils.logging import log_metrics
from ...envs.observables import curve_from_infos
from ...utils.plotting import plot_episode
import matplotlib.pyplot as plt


def plot_reward_breakdown(infos: List[Dict], filename: str, v_limit: float):
    """
    绘制单回合内奖励各项的详细拆解图，用于诊断 Agent 行为。
    """
    # 1. 提取数据
    # infos 列表长度为 N+1 (包含初始状态)，但动作和奖励只有 N 个
    # 我们取前 N 个 info 中的奖励记录
    steps = range(len(infos) - 1)
    
    # 初始化数据容器
    data = {
        "r_soc": [], "r_time": [], "r_v": [], "r_t": [], 
        "r_const": [], "r_action": [], "total_step_reward": [],
        "SOC": [], "I": [], "std_SOC": [], "V_max": []
    }
    
    for i in steps:
        info = infos[i]
        # 获取各项奖励 (如果字典里没有，默认为 0)
        data["r_soc"].append(info.get("r_soc", 0.0))
        data["r_time"].append(info.get("r_time", 0.0))
        data["r_v"].append(info.get("r_v", 0.0))
        data["r_t"].append(info.get("r_t", 0.0))
        data["r_const"].append(info.get("r_const", 0.0))
        data["r_action"].append(info.get("r_action", 0.0))
        # 注意：rollout_surrogate 里 step_reward 是存在 info["reward"] 还是累加的？
        # 这里我们重新把分项加一遍以确保准确展示分项之和
        step_sum = (data["r_soc"][-1] + data["r_time"][-1] + data["r_v"][-1] + 
                    data["r_t"][-1] + data["r_const"][-1] + data["r_action"][-1])
        data["total_step_reward"].append(step_sum)
        
        # 物理状态
        data["SOC"].append(info["SOC_pack"])
        data["I"].append(info.get("I", 0.0)) # 或者是 info["i_abs"] * sign
        data["std_SOC"].append(info.get("std_SOC", 0.0))
        data["V_max"].append(info["V_cell_max"])

    # 2. 绘图
    fig, axes = plt.subplots(3, 1, figsize=(12, 14), sharex=True)
    
    # --- 子图 1: 物理状态 (SOC, 电流, 电压) ---
    ax1 = axes[0]
    ax1.plot(steps, data["SOC"], color="blue", label="SOC", linewidth=2)
    ax1.set_ylabel("SOC", color="blue")
    ax1.tick_params(axis='y', labelcolor="blue")
    ax1.grid(True, alpha=0.3)
    
    ax1_r = ax1.twinx()
    ax1_r.plot(steps, data["I"], color="green", linestyle="--", label="Current (A)")
    ax1_r.set_ylabel("Current (A)", color="green")
    ax1_r.tick_params(axis='y', labelcolor="green")
    ax1.set_title("Physical State Evolution")
    
    # --- 子图 2: 关键瓶颈指标 (std_SOC, V_max) ---
    # 专门用来盯着看是不是这两个指标导致了惩罚
    ax2 = axes[1]
    ax2.plot(steps, data["std_SOC"], color="orange", label="Cell Spread (std_SOC)", linewidth=2)
    ax2.axhline(0, color="grey", lw=0.5)
    ax2.set_ylabel("std_SOC", color="orange")
    
    ax2_r = ax2.twinx()
    ax2_r.plot(steps, data["V_max"], color="red", label="V_max")
    ax2_r.axhline(v_limit, color="red", linestyle=":", label="V Limit")
    ax2_r.set_ylabel("Voltage (V)", color="red")
    ax2.set_title("Constraints: Cell Spread & Voltage")
    ax2.legend(loc="upper left")
    ax2_r.legend(loc="upper right")
    
    # --- 子图 3: 奖励分项堆叠/对比图 (核心诊断) ---
    ax3 = axes[2]
    # 绘制总奖励背景
    ax3.fill_between(steps, data["total_step_reward"], 0, color="gray", alpha=0.1, label="Total Step Reward")
    
    # 绘制各分项
    ax3.plot(steps, data["r_soc"], label="Reward: SOC (+)", color="green", linewidth=1.5)
    ax3.plot(steps, data["r_const"], label="Penalty: Consistency (-)", color="orange", linewidth=1.5)
    ax3.plot(steps, data["r_v"], label="Penalty: Voltage (-)", color="red", linewidth=1.5)
    ax3.plot(steps, data["r_action"], label="Penalty: Action (-)", color="purple", linestyle=":", linewidth=1)
    ax3.plot(steps, data["r_time"], label="Penalty: Time (-)", color="black", linestyle="--", linewidth=1)
    
    ax3.set_ylabel("Reward Value")
    ax3.set_xlabel("Steps")
    ax3.set_title("Reward Decomposition (Why did it stop?)")
    ax3.legend(loc="lower left", ncol=3, fontsize='small')
    ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()
    print(f"[Diag] Reward breakdown saved to {filename}")

def summarize_reward_terms(infos, v_limit, idle_thr=1.0):
    import numpy as np

    # 对齐：reward/分项通常存在 infos[t+1]，从 1..end
    keys = ["reward", "r_soc", "r_time", "r_v", "r_t", "r_const", "r_action", "delta_soc", "v_soft"]
    arr = {k: [] for k in keys}
    I_list = []

    for t in range(len(infos) - 1):
        nxt = infos[t + 1]
        cur = infos[t]
        I_list.append(float(cur["I"]))
        for k in keys:
            if k in nxt:
                arr[k].append(nxt[k])

    # 转 numpy
    for k in arr:
        arr[k] = np.asarray(arr[k], dtype=float) if len(arr[k]) else np.asarray([], dtype=float)
    I = np.asarray(I_list, dtype=float)

    # delta_soc 统计
    ds = arr["delta_soc"]
    if len(ds):
        ds_stats = {
            "mean": float(ds.mean()),
            "median": float(np.median(ds)),
            "p10": float(np.quantile(ds, 0.10)),
            "p90": float(np.quantile(ds, 0.90)),
            "min": float(ds.min()),
            "max": float(ds.max()),
        }
    else:
        ds_stats = {}

    # 软限触发统计
    v_soft = arr["v_soft"] > 0.5 if len(arr["v_soft"]) else np.zeros_like(I, dtype=bool)
    soft_ratio = float(v_soft.mean()) if len(v_soft) else 0.0

    # 电流统计（总体/软限区间）
    def stat_I(mask=None):
        x = I if mask is None else I[mask]
        if len(x) == 0:
            return {"mean": None, "std": None, "min": None, "max": None, "idle_ratio": None}
        return {
            "mean": float(x.mean()),
            "std": float(x.std()),
            "min": float(x.min()),
            "max": float(x.max()),
            "idle_ratio": float((np.abs(x) < idle_thr).mean()),
        }

    I_all = stat_I()
    I_soft = stat_I(v_soft[:len(I)])  # 对齐长度

    # reward 分项均值和累积
    term_stats = {}
    for k in ["reward", "r_soc", "r_time", "r_v", "r_t", "r_const", "r_action"]:
        x = arr[k]
        if len(x):
            term_stats[k] = {
                "mean": float(x.mean()),
                "sum": float(x.sum()),
                "min": float(x.min()),
                "max": float(x.max()),
            }

    return {
        "delta_soc": ds_stats,
        "soft_ratio": soft_ratio,
        "I_all": I_all,
        "I_soft": I_soft,
        "terms": term_stats,
    }

# 保存模型的函数
def save_agent_model(agent: DDPGAgent, filepath="trained_agent_model.pth"):
    torch.save({
        "actor": agent.actor.state_dict(),  # 保存 actor 网络
        "critic": agent.critic.state_dict(),  # 保存 critic 网络
        "state_norm": agent.state_norm,  # 保存归一化参数
    }, filepath)  # 保存文件路径
    print(f"代理模型已保存到: {filepath}")


# 加载已保存的模型
def load_agent_model(agent: DDPGAgent, filepath="trained_agent_model.pth"):
    checkpoint = torch.load(filepath)
    agent.actor.load_state_dict(checkpoint["actor"])  # 加载 actor 网络
    agent.critic.load_state_dict(checkpoint["critic"])  # 加载 critic 网络
    agent.state_norm = checkpoint["state_norm"]  # 恢复归一化参数
    print(f"代理模型已加载: {filepath}")


def validate_surrogate_rollout(env, agent, surrogate, dataset, 
                               steps=10, 
                               hold_steps=1, 
                               test_currents=[-10.0, -12.0, -5.0]): # 1. 修改参数，传入电流列表
    """
    对比 真实环境 vs 代理模型 的多步演化 (支持多电流测试)
    """
    print(f"开始验证代理模型滚动预测精度，测试电流列表: {test_currents}")
    
    # 2. 将绘图初始化移到循环外部，以便在同一张图上绘制
    fig, axes = plt.subplots(3, 1, figsize=(10, 12))
    
    # 生成颜色映射，区分不同电流
    colors = plt.cm.viridis(np.linspace(0, 1, len(test_currents)))

    # --- 开始循环测试不同电流 ---
    for idx, test_current in enumerate(test_currents):
        print(f"\n>>> 正在测试电流: {test_current}A ...")
        
        # 3. 从环境获取一个初始状态 (每次循环都重置)
        state, info = env.reset()
        
        # 设置当前循环的测试电流
        action = np.array([test_current], dtype=np.float32)
        
        # 记录真实轨迹和预测轨迹
        real_traj = {'V': [], 'T': [], 'SOC': []}
        pred_traj = {'V': [], 'T': [], 'SOC': []}
        
        # 记录初始点
        real_traj['SOC'].append(state[0])
        real_traj['V'].append(state[2])   
        real_traj['T'].append(state[4])   
        
        pred_traj['SOC'].append(state[0])
        pred_traj['V'].append(state[2])
        pred_traj['T'].append(state[4])

        # 代理模型的当前状态初始化为真实初始状态
        curr_pred_state = state.copy()
        
        # 4. 运行 N 步 (核心逻辑保持不变)
        done_early = False
        for i in range(steps):
            # --- A. 真实环境演化 ---
            for _ in range(hold_steps):
                next_state_real, _, done, _, _ = env.step(action)
                if done:
                    done_early = True
                    break # 跳出 hold_steps 循环

            # --- B. 代理模型演化 ---
            pred_delta, pred_std = surrogate.predict(curr_pred_state, action)
            
            # 更新预测状态
            next_state_pred = curr_pred_state.copy()
            next_state_pred[:6] += pred_delta 
            next_state_pred[6] = float(action[0]) 
            
            # --- 记录数据 ---
            # 注意：如果真实环境done了，next_state_real可能是不完整的，但为了绘图我们只记录有效的
            if not done_early or (done_early and i == 0): # 简单处理：如果第一步就done了可能需要特殊处理
                real_traj['SOC'].append(next_state_real[0])
                real_traj['V'].append(next_state_real[2]) 
                real_traj['T'].append(next_state_real[4]) 
            
            pred_traj['SOC'].append(next_state_pred[0])
            pred_traj['V'].append(next_state_pred[2])
            pred_traj['T'].append(next_state_pred[4])
            
            # 递归更新
            curr_pred_state = next_state_pred
            
            if done_early:
                print(f"  - 真实环境在第 {i} 个大步触发约束提前结束")
                break

        # 5. 绘图 (在循环内绘制当前电流的曲线)
        # 动态计算当前轨迹的时间轴
        time_axis = np.arange(len(real_traj['SOC'])) * (hold_steps * 10.0)
        c = colors[idx] # 当前线条颜色
        
        # SOC 子图
        ax = axes[0]
        ax.plot(time_axis, real_traj['SOC'], linestyle='-', color=c, label=f'Real {test_current}A')
        ax.plot(time_axis, pred_traj['SOC'], linestyle='--', color=c, label=f'Pred {test_current}A')
        
        # 电压 子图
        ax = axes[1]
        ax.plot(time_axis, real_traj['V'], linestyle='-', color=c)
        ax.plot(time_axis, pred_traj['V'], linestyle='--', color=c)
        
        # 温度 子图
        ax = axes[2]
        ax.plot(time_axis, real_traj['T'], linestyle='-', color=c)
        ax.plot(time_axis, pred_traj['T'], linestyle='--', color=c)

    # 6. 设置图例和标签 (循环外)
    axes[0].set_ylabel('SOC')
    axes[0].set_title(f'Surrogate Rollout Validation (Hold={hold_steps*10}s)')
    axes[0].legend(ncol=2, fontsize='small') # 图例分两列显示以免太长
    
    axes[1].set_ylabel('Voltage (V)')
    
    axes[2].set_ylabel('Temperature (K)')
    axes[2].set_xlabel('Time (seconds)')
    
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
@dataclass
class Cycle0Config:
    """Configuration for cycle0 training."""

    real_episodes: int  # 跑真实环境，收集 (s, a, Δs)
    surrogate_epochs: int # 用这些数据训练静态代理模型的 epoch 数
    policy_epochs: int #用 surrogate 做 rollout，产生 “伪经验” 来训练 RL policy（DDPG）
    policy_rollouts_per_epoch: int = 5  # 每个 epoch 用 surrogate rollout 的条数
    updates_per_epoch: int = 20  # 每个 epoch 更新 policy 的次数
    plot_interval: int = 1  # 每隔多少个 epoch 保存一次轨迹图（1=每次）
    # ====== collect_real_data 探索参数 ======
    eps_random_start: float = 0.85     # 你的环境很容易撞 Vmax，随机比例建议更高
    eps_random_end: float = 0.25

    # 这里写“比例”，真正 sigma 会乘 (high-low)
    noise_sigma_start: float = 0.20
    noise_sigma_end: float = 0.05

    hold_steps: int = 1               # dt=10s，hold 5步=50s，更像阶梯恒流
    v_soft_max: float = 4.17   # 或 env.v_max - 0.03
    t_soft_max: float = 309  # 或 env.t_max - 1.5

"""
真实环境采样的关键逻辑 
"""
def collect_real_data(
    env: BasePackEnv,
    reward_cfg: PaperRewardConfig,
    agent: DDPGAgent,
    config: Cycle0Config,
) -> List[tuple[np.ndarray, np.ndarray, np.ndarray]]:
    # 1.准备工作
    low = float(agent.config.action_low)   # -20
    high = float(agent.config.action_high) # 0
    transitions: List[tuple[np.ndarray, np.ndarray, np.ndarray]] = []

    # 软约束参数
    # v_soft = float(getattr(config, "v_soft_max", env.v_max - 0.03))  # 比硬约束低一点，比如 4.17
    # t_soft = float(getattr(config, "t_soft_max", env.t_max - 1.5))   # 比硬约束低一点，比如 318.5K

    # 定义Guard：当接近边界时，把电流往 0A 拉
    # def guard_action(action: float, info: dict) -> float:
    #     v = float(info.get("V_cell_max", -1e9))
    #     t = float(info.get("T_cell_max", -1e9))
    #     viol = bool(info.get("violation", False))

    #     # 已经违规：立即大幅减小电流（靠近0）
    #     if viol:
    #         return float(np.clip(action + 5.0, low, high))  # prev_action是负数，加5就是减小幅值

    #     # 接近电压或温度软阈值：渐进减小电流
    #     if v >= v_soft or t >= t_soft:
    #         return float(np.clip(action + 2.0, low, high))

    #     return action
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
        v_limit = 4.22 
        r_internal = 0.025 
        
        # 3. 计算物理安全电流边界 I_bound (根据论文公式 6-13) [cite: 238]
        # I_bound = (U_cut - U_ocv) / R
        i_bound_single = max(0.0, (v_limit - u_ocv) / r_internal)
        
        # 4. 适配 3p6s 电池组总电流 (充电为负值)
        i_bound_pack = i_bound_single * 3.0
        
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
        # 仅在 Warmup 阶段使用
        # state, info = env.reset(options={"soc_low": 0.1, "soc_high": 0.9})
          # 前 30% 的回合强制从低电量开始，因为快充主要发生在低电量
        # if ep < config.real_episodes * 0.1:
        #     options = {"soc_low": 0.05, "soc_high": 0.3}
        # else:
        # options = {"soc_low": 0.1, "soc_high": 0.9}
        # state, info = env.reset(options=options)
      
        if np.random.rand() > 0.5:
            options = {"soc_low": 0.5, "soc_high": 0.8} 
        else:
            options = {"soc_low": 0.1, "soc_high": 0.5}

        state, info = env.reset(options=options)
        done = False

        # 统计本回合奖励
        ep_reward = 0.0
        current_soc = float(info.get("SOC_pack", state[0]))
        while not done:
            # --- A. 选择动作 ---
            raw_action = float(agent.act(state)[0])
            # 策略 2: 注入大尺度噪声或强制随机。
            # 20% 的时间里，完全不听 Agent 的，强制输出大电流测试物理极限
            if np.random.rand() < 0.2:
                a_noisy = np.random.uniform(low, high) # 强制全局随机探索
            else:
                a_noisy = raw_action + noise.sample()
                
            a_clipped = float(np.clip(a_noisy, low, high))
            
            # 策略 3: 应用物理限制，但给限制留一点“溢出”空间
            # 这样能采集到“刚超一点压”和“刚好不超压”的对比数据，对代理模型极其重要
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

                next_state, _, terminated, truncated, next_info = env.step(action_to_exec)
                
                # 更新当前的 SOC, V, T 等信息
                current_soc = float(next_info["SOC_pack"])
                v_max = float(next_info["V_cell_max"])
                t_max = float(next_info["T_cell_max"])
                # 尝试获取 std_soc，如果没有则设为 0
                std_soc = float(next_info.get("std_SOC", 0.0)) 
                # 获取真实的电流
                I_exec = float(next_info.get("I_pack_true", safe_action))

                # 手动计算这一小步的奖励
                # 注意：这里调用你外部定义的 compute_paper_reward
                # r_step, _ = compute_paper_reward(
                #     soc_prev=prev_soc,
                #     soc_next=current_soc,
                #     v_max_next=v_max,
                #     t_max_next=t_max,
                #     std_soc_next=std_soc,
                #     action_current=I_exec, # 传入实际执行的电流
                #     v_limit=env.v_max,
                #     t_limit=env.t_max,
                #     config=reward_cfg 
                # )
                r_step = compute_minimal_reward(
                    soc_prev=prev_soc,
                    soc_next=current_soc,
                    v_max=v_max,
                    t_max=t_max,
                    v_limit=env.v_max,
                    t_limit=env.t_max,
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
            # 存入的是：(0s状态, 50s动作, 50s总奖励, 50s状态)
            agent.observe(start_state, action_to_exec, accumulated_reward, state, done)
            # 只有当 buffer 数据够了才 update
            if len(agent.buffer) > agent.config.batch_size:
                agent.update()

            # # --- 4. 收集数据给 GP ---
            # Delta = State(t + 50s) - State(t)
            # 这样算出来的 温度 Delta 会比之前大 5 倍以上！
            final_delta = state[:6] - start_state[:6]
            # 存入 dataset 的是：在 start_state 下，执行 action，导致了 final_delta 的变化
            transitions.append((start_state.copy(), action_to_exec.copy(), final_delta.copy()))

            ep_reward += accumulated_reward
        # 打印日志
        if (ep + 1) % 5 == 0:
            print(
                f"[Warmup] Ep {ep+1} | R: {ep_reward:.2f} | "
                f"SOC: {info['SOC_pack']:.4f} | Vmax: {info['V_cell_max']:.4f} | "
                f"Buf: {len(agent.buffer)}"
            )

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

def train_cycle0(
    env: BasePackEnv,
    agent: DDPGAgent,
    reward_cfg: PaperRewardConfig,
    surrogate: StaticSurrogate,
    config: Cycle0Config,
    run_dir: str,
) -> Dict[str, float]:
    
    """Run Cycle0 training pipeline."""
    # 1) 真实环境采集
    transitions = collect_real_data(env, reward_cfg, agent, config)
    verify_dataset_coverage(transitions)
    save_transitions_to_csv(transitions, "dataset.csv")

    actions = np.array([a[0] for _, a, _ in transitions], dtype=float)
    print("Action stats: mean", actions.mean(), "std", actions.std(), "min", actions.min(), "max", actions.max())

    # 2) 静态代理模型训练
    dataset = build_dataset(transitions)
    # t_idx = -2 
    # v_idx = 2   # 假设第2列是 V_max

    # print(f"--- 数据统计量诊断 ---")
    # print(f"【温度 Delta】 均值: {dataset.d_mean[t_idx]:.8f}, 标准差: {dataset.d_std[t_idx]:.8f}")
    # print(f"【电压 Delta】 均值: {dataset.d_mean[v_idx]:.8f}, 标准差: {dataset.d_std[v_idx]:.8f}")
    # print(f"【温度 Delta】 最大值: {np.max(dataset.deltas[:, t_idx]):.8f}, 最小值: {np.min(dataset.deltas[:, t_idx]):.8f}")

    # # 关键检查点
    # if dataset.d_std[t_idx] <= 1.1e-6:
    #     print("❌ 致命错误：温度标准差接近 1e-6 (Clip值)。")
    #     print("   原因：采样间隔太短，或者数据里全是静置，温度根本没变。")
    #     print("   后果：模型认为温度永远不变，归一化失效。")
    # else:
    #     print("✅ 统计量看起来有波动。")

    surrogate.fit(dataset, epochs=config.surrogate_epochs)
    print("静态代理模型训练完成。")
    
    # 3) 测试 surrogate 训练效果
    validate_surrogate_rollout(env, agent, surrogate, dataset)
    
    # 4) 用静态代理训练 RL 策略
    low = float(agent.config.action_low)
    high = float(agent.config.action_high)
    actor_losses = []
    critic_losses = []
   
    print(f"训练开始前 Buffer 大小: {len(agent.buffer)}")

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
            noise = np.random.normal(0, current_sigma)

            # C. 【可选优化】防止 0A 截断的非对称噪声技巧
            # 如果当前动作已经很接近 0 (比如 > -2A)，且噪声是正的，这会导致结果 > 0 被截断
            # 我们强制反转噪声方向，让它往负方向探索
            if a_det + noise > high:
                noise = -abs(noise)
            # 如果 (动作+噪声) 低于下界 -20，强制反转，向正方向探索
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
            if rollout_idx < 3:
                state0, _ = env.reset()
            else:
                state0 = dataset.states[np.random.randint(0, dataset.states.shape[0])].copy()
            
            total_reward, infos = rollout_surrogate(
                state=state0,
                surrogate=surrogate.predict,
                policy=policy_train,
                horizon=env.max_steps,
                reward_cfg=reward_cfg,
                dt = env.dt * config.hold_steps,  # 代理模型的时间步长,
                v_max=env.v_max,
                t_max=env.t_max,
            )
            
            if rollout_idx == 0:
                diag = summarize_reward_terms(infos, v_limit=env.v_max, idle_thr=1.0)
                print("[DELTA_SOC]", diag["delta_soc"])
                print(f"[VSOFT] soft_ratio={diag['soft_ratio']:.3f}")
                print("[I_ALL]", diag["I_all"])
                print("[I_SOFT]", diag["I_soft"])
                print("[TERMS_MEAN]",
                    {k: round(v["mean"], 6) for k, v in diag["terms"].items()})
                print("[TERMS_SUM ]",
                    {k: round(v["sum"],  3) for k, v in diag["terms"].items()})
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
                    # =========== 新增代码开始 ===========
                    # 只有在绘图周期，且 rollout_idx == 0 (只画第一个 rollout) 时调用
                    diagnosis_filename = f"{run_dir}/reward_diag_epoch_{epoch}.png"
                    plot_reward_breakdown(infos, diagnosis_filename, v_limit=env.v_max)
                    # =========== 新增代码结束 ===========

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

                # 存入 Buffer
                # if epoch == 0 and rollout_idx == 0 and t < 15:
                #     print("[CHK] t", t, "a", a_val, "s_Iprev", s[-1], "snext_Iprev", s_next[-1], "r", r, "done", done)

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
    

        # --- 【修改点 3】: 更新并记录 Loss ---
        # 确保 Agent 在每个 epoch 结束时进行多次更新
        for _ in range(config.updates_per_epoch):
            # 只有当 buffer 够大时才 update
            if len(agent.buffer) > agent.config.batch_size:
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
