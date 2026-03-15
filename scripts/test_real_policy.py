"""Test the trained Actor policy in the real Liionpack environment."""
from __future__ import annotations

import argparse
import sys
import os
from pathlib import Path
import numpy as np
import torch
import matplotlib.pyplot as plt

# 1. 路径设置 (保持与训练脚本一致)
current_dir = Path(__file__).resolve().parent
root_dir = current_dir.parent
sys.path.append(str(root_dir))

from src.envs.liionpack_spme_pack_env import build_pack_env
from src.rl.agent_factory import build_agent_from_config
from src.utils.config import load_config
from src.utils.seeds import set_global_seed

# 设置设备
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def guard_action(action: float, info: dict, low: float, high: float) -> float:
    """生产环境必备的安全守卫 (与训练时逻辑保持一致)"""
    v = float(info.get("V_cell_max", -1e9))
    t = float(info.get("T_cell_max", -1e9))
    viol = bool(info.get("violation", False))
    
    # 硬编码一些软约束参数，或者从 config 传入
    v_soft = 4.17
    t_soft = 318.15 - 1.5 

    # 1. 已经违规：大幅回撤
    if viol:
        return float(np.clip(action + 5.0, low, high))
    
    # 2. 接近边界：温和回撤
    if v >= v_soft or t >= t_soft:
        return float(np.clip(action + 2.0, low, high))
        
    return action

def plot_results(logs, dt, save_path=None):
    """绘制测试结果曲线"""
    steps = len(logs["soc"])
    time = np.arange(steps) * dt
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    
    # SOC
    axes[0, 0].plot(time, logs["soc"], label="SOC", color='green')
    axes[0, 0].set_title("Pack SOC")
    axes[0, 0].set_ylabel("SOC")
    axes[0, 0].set_xlabel("Time [s]")
    axes[0, 0].grid(True)
    
    # Voltage
    axes[0, 1].plot(time, logs["voltage"], label="Max V", color='blue')
    axes[0, 1].axhline(4.2, color='red', linestyle='--', label="Limit 4.2V")
    axes[0, 1].set_title("Max Cell Voltage")
    axes[0, 1].set_ylabel("Voltage [V]")
    axes[0, 1].set_xlabel("Time [s]")
    axes[0, 1].legend()
    axes[0, 1].grid(True)
    
    # Current
    axes[1, 0].plot(time, logs["current"], label="Current", color='orange')
    axes[1, 0].set_title("Pack Current")
    axes[1, 0].set_ylabel("Current [A]")
    axes[1, 0].set_xlabel("Time [s]")
    axes[1, 0].grid(True)
    
    # Reward
    axes[1, 1].plot(time, logs["reward"], label="Reward", color='purple')
    axes[1, 1].set_title("Step Reward")
    axes[1, 1].set_ylabel("Reward")
    axes[1, 1].set_xlabel("Time [s]")
    axes[1, 1].grid(True)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
        print(f"Plot saved to {save_path}")
    plt.show()

def main() -> None:
    parser = argparse.ArgumentParser()
    # 需要传入原本的 config 文件以构建环境
    parser.add_argument("--config", required=True, help="Path to the config.yaml used for training")
    # 自动扫描 runs/adaptive/adaptive_cyclexx/agent_ckpt.pt
    parser.add_argument(
        "--runs-root",
        default="runs/adaptive",
        help="Root directory that contains adaptive_cycle*/agent_ckpt.pt",
    )
    parser.add_argument(
        "--start-cycle",
        type=int,
        default=1,
        help="Starting cycle index, inclusive",
    )
    parser.add_argument(
        "--end-cycle",
        type=int,
        default=50,
        help="Ending cycle index, inclusive",
    )
    # 可选：是否禁用安全守卫 (纯测神经网络性能)
    parser.add_argument("--no-guard", action="store_true", help="Disable safety guard")
    args = parser.parse_args()

    # 1. 加载配置
    print(f"Loading config from {args.config}...")
    config = load_config(args.config).data
    set_global_seed(config.get("seed", 42))

    # 2. 构建真实环境 (LiionPack)
    print("Building environment...")
    env = build_pack_env(config["env"])
    env.set_aging_stage(4)
    # 3. 初始化 Agent (结构必须与训练时一致)
    print("Initializing Agent...")
    # 注意：action_dim 这里硬编码为 1，与你训练脚本一致
    agent = build_agent_from_config(
        state_dim=env.observation_space.shape[0],
        action_dim=1,
        rl_config=config["rl"],
    )

    runs_root = Path(args.runs_root)
    
    # 动作范围
    low = config["rl"]["action_low"]
    high = config["rl"]["action_high"]

    for cycle in range(args.start_cycle, args.end_cycle + 1):
        policy_path = runs_root / f"adaptive_cycle{cycle}" / "agent_ckpt.pt"
        if not policy_path.exists():
            print(f"⚠️ Skip cycle {cycle}: policy file not found: {policy_path}")
            continue

        # 4. 加载权重
        print(f"\n=== Cycle {cycle} ===")
        print(f"Loading policy weights from {policy_path}...")
        try:
            agent.load(str(policy_path), map_location=str(DEVICE))
            print("✅ Policy and Normalizer loaded successfully.")
        except Exception as e:
            print(f"❌ Failed to load policy for cycle {cycle}: {e}")
            continue

        agent.actor.eval()  # 切换到评估模式

        # 5. 开始测试循环
        print("Starting inference...")
        state, info = env.reset()
        done = False

        logs = {"soc": [], "voltage": [], "current": [], "reward": []}
        total_reward = 0.0
        steps = 0

        while not done:
            # A. 神经网络预测 (确定性策略)
            # agent.act 内部已经包含归一化处理(如果有)和 no_grad
            action = agent.act(state)
            raw_action = float(action[0])

            # B. 安全守卫 (可选)
            if not args.no_guard:
                final_action = guard_action(raw_action, info, low, high)
            else:
                final_action = raw_action
                # 依然需要 clip 防止超出物理极限
                final_action = float(np.clip(final_action, low, high))

            # 包装成 array
            action_vec = np.array([final_action], dtype=np.float32)

            # C. 环境执行
            next_state, reward, terminated, truncated, next_info = env.step(action_vec)

            # D. 记录数据
            # 注意：这里我们取 next_info 里的真实数据
            logs["soc"].append(next_info.get("SOC_pack", 0.0))
            logs["voltage"].append(next_info.get("V_cell_max", 0.0))
            # 记录真实执行的电流 (I_pack_true)，而不是我们设定的电流，看看是否有偏差
            logs["current"].append(next_info.get("I_pack_true", final_action))
            logs["reward"].append(reward)

            state = next_state
            info = next_info
            total_reward += reward
            steps += 1

            if terminated or truncated:
                done = True
                reason = info.get("terminated_reason", "Unknown")
                print(f"Episode finished at step {steps}. Reason: {reason}")

        print(f"Test Complete. Total Reward: {total_reward:.2f}, Final SOC: {logs['soc'][-1]:.4f}")

        # 6. 画图
        save_img_path = policy_path.parent / "test_result.png"
        plot_results(logs, dt=config["env"]["dt"], save_path=save_img_path)

if __name__ == "__main__":
    main()