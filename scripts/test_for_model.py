"""Train Cycle0 static surrogate and actor-critic policy."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, Dict, List, Tuple
import csv
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pybamm
from scipy.interpolate import interp1d
from pathlib import Path
import argparse
import sys


# 获取当前脚本的绝对路径
current_dir = Path(__file__).resolve().parent
# 获取项目根目录 (即 scripts 的上一级)
root_dir = current_dir.parent
# 将根目录添加到 Python 搜索路径中
sys.path.append(str(root_dir))

from src.envs.liionpack_spme_pack_env import build_pack_env
from src.rewards.paper_reward import PaperRewardConfig
from src.rl.actor_critic_ddpg import DDPGAgent, DDPGConfig
from src.rl.trainers.trainer_static_gp import Cycle0Config
from src.surrogate.gp_static import StaticSurrogate
from src.utils.config import load_config
from src.utils.seeds import set_global_seed
from src.utils.logging import ensure_dir
from src.envs.base_env import BasePackEnv
from src.surrogate.dataset import build_dataset




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

        t_diff_max = curr_pred_state[4] - 298.15
        # 构造代理模型需要的 state 输入 (7维原始 + 1维温差 = 8维)
        state_aug_for_surrogate = np.append(curr_pred_state, t_diff_max)
        # 3. 演化
        pred_delta, _ = surrogate.predict(state_aug_for_surrogate, np.array([raw_action_pred]))
        d4_neural = pred_delta[4]
        d5_neural = pred_delta[5]
        
        # 3. 核心物理干预： Newton's Cooling Law (散热补丁)
        # T_amb 是环境温度，假设为 298.15
        # k 是散热系数，这是一个超参数。
        # 如果温度还是不降，就调大 k (如 0.005, 0.01)
        k_cooling = 0.03  
        T_amb = 298.15
        
        # 获取当前 obs 中的温度 (假设 obs 的索引 4 是当前温度)
        current_T = curr_pred_state[4]
        
        # 计算由于温差产生的散热量（一定是负值，除非电池比环境冷）
        cooling_effect = -k_cooling * (current_T - T_amb)
        
        # 4. 叠加散热项
        # 神经网络学到的是“产热规律”，物理公式补充“散热规律”
        pred_delta[4] = d4_neural + cooling_effect
        pred_delta[5] = d5_neural + cooling_effect
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
        
        t_diff_max = curr_pred_state[4] - 298.15
        # 构造代理模型需要的 state 输入 (7维原始 + 1维温差 = 8维)
        state_aug_for_surrogate = np.append(curr_pred_state, t_diff_max)
        # 3. 演化
        delta, _ = surrogate.predict(state_aug_for_surrogate, np.array([raw_act_align]))
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

def validate_surrogate_rollout(env, agent, surrogate, dataset, steps=20, hold_steps=1):
    """
    对比 真实环境 vs 代理模型 的多步演化
    """
    print("开始验证代理模型滚动预测精度...")
    
    # 1. 从环境获取一个初始状态
    state, info = env.reset()

    #使用大电流测试
    test_current = -15.0 
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

    hold_steps: int = 1              # dt=10s，hold 5步=50s，更像阶梯恒流
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
    agent: DDPGAgent,
    reward_cfg: PaperRewardConfig,
    surrogate: StaticSurrogate,
    config: Cycle0Config,
    run_dir: str,
) -> Dict[str, float]:
    
    """Run Cycle0 training pipeline."""
    # 1) 真实环境采集

    
    csv_path = "C:\\Users\\85721\\Desktop\\fast-charging-EV\\dataset_final_physics.csv"
    transitions = load_transitions_from_csv(csv_path)
   

    ckpt_path = "C:\\Users\\85721\\Desktop\\fast-charging-EV\\runs\\cycle0\\agent_ckpt.pt"
    
    agent.load(str(ckpt_path))
    # 2) 静态代理模型训练
    dataset = build_dataset(transitions)
 
    surrogate.fit(dataset, epochs=config.surrogate_epochs)

    print("静态代理模型训练完成。")
    
    # 3) 测试 surrogate 训练效果
    # validate_surrogate_rollout(env, agent, surrogate, dataset)
    # 3) N-step 误差评估（用 agent 跑真实轨迹，然后 surrogate 复现）
    validate_full_charge_comparison(env, agent, surrogate)
    

parser = argparse.ArgumentParser()
parser.add_argument("--config", required=True)
args = parser.parse_args()

config = load_config(args.config).data
set_global_seed(config.get("seed"))

env = build_pack_env(config["env"])
reward_cfg = PaperRewardConfig(**config["reward"])
rl_cfg = DDPGConfig(
    gamma=config["rl"]["gamma"],
    actor_lr=config["rl"]["actor_lr"],
    critic_lr=config["rl"]["critic_lr"],
    tau=config["rl"]["tau"],
    batch_size=config["rl"]["batch_size"],
    buffer_size=config["rl"]["buffer_size"],
    action_low=config["rl"]["action_low"],
    action_high=config["rl"]["action_high"],
)
agent = DDPGAgent(state_dim=env.observation_space.shape[0], action_dim=1, config=rl_cfg)

surrogate = StaticSurrogate(
    input_dim=env.observation_space.shape[0] + 2 ,  # state(7) + action(1)
    output_dim=env.observation_space.shape[0] -1,  # 预测 delta 的前 6 维（不预测 Iprev）
    hidden_sizes=config["surrogate"]["hidden_sizes"],
    ensemble_size=config["surrogate"]["ensemble_size"],
    lr=config["surrogate"]["learning_rate"],
)

run_dir = ensure_dir(Path(config["logging"]["runs_dir"]) / "cycle0")
cycle0_cfg = Cycle0Config(**config["trainer"]["cycle0"])

train_cycle0(env, agent, reward_cfg, surrogate, cycle0_cfg, str(run_dir))

