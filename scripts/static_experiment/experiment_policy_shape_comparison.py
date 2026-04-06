"""静态代理模型有效性分析实验二：验证50个episode+静态代理模型训练的有效性
1.在真实物理仿真环境下训练650个episode得到agent
2.在50个真实物理仿真环境训练后200个episode在静态代理模型上继续训练得到agent（直接加载runs/cycle0/agent_ckpt.pt）
3.对比两个agent在真实物理仿真环境中的行为轨迹（施加电流、SOC、端电压、电池温度随时间的变化）
"""
from __future__ import annotations

import argparse
import time
import sys
from pathlib import Path
import matplotlib.pyplot as plt

current_dir = Path(__file__).resolve().parent
if str(current_dir) not in sys.path:
    sys.path.append(str(current_dir))

from common import (
    build_env_agent_reward,
    evaluate_policy_trajectory,
    run_real_training_collect_style,
)


def plot_trajectories(real_traj, mix_traj, output: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
    panels = [
        ("current_a", "施加电流 (A)"),
        ("soc", "电池 SOC (-)"),
        ("voltage_v", "端电压 (V)"),
        ("temperature_k", "电池温度 (K)"),
    ]
    for ax, (key, label) in zip(axes.ravel(), panels):
        ax.plot(real_traj["time_s"], real_traj[key], label="真实环境训练 650 episodes", linewidth=2)
        ax.plot(mix_traj["time_s"], mix_traj[key], "--", label="加载混合训练 agent_ckpt", linewidth=2)
        ax.set_ylabel(label)
        ax.grid(alpha=0.3)
    axes[1, 0].set_xlabel("时间 (seconds)")
    axes[1, 1].set_xlabel("时间 (seconds)")
    axes[0, 0].legend()
    plt.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output, dpi=220)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/pack_3p6s_spme.yaml"))
    parser.add_argument("--output-dir", type=Path, default=Path("runs/static_experiment/exp1_policy_shape"))
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--real-episodes", type=int, default=650)
    parser.add_argument("--agent-ckpt", type=Path, default=Path("runs/cycle0/agent_ckpt.pt"))
    parser.add_argument("--save-real-agent-ckpt", type=Path, default=None)
    args = parser.parse_args()

    cfg, env_real, agent_real, reward_cfg = build_env_agent_reward(args.config, args.seed)
    _ = cfg
    # === 开始计时 ===
    print(f"开始在真实物理仿真环境中训练 {args.real_episodes} 个 episodes...")
    start_time = time.time()
    
    run_real_training_collect_style(env_real, agent_real, reward_cfg, episodes=args.real_episodes)
    
    # === 结束计时 ===
    end_time = time.time()
    elapsed_seconds = end_time - start_time
    elapsed_minutes = elapsed_seconds / 60.0
    elapsed_hours = elapsed_minutes / 60.0
    
    print(f"[耗时统计] 真实环境训练 {args.real_episodes} episodes 共耗时: "
          f"{elapsed_seconds:.2f} 秒 (约 {elapsed_minutes:.2f} 分钟 / {elapsed_hours:.2f} 小时)")
    # run_real_training_collect_style(env_real, agent_real, reward_cfg, episodes=args.real_episodes)
    save_real_ckpt = args.save_real_agent_ckpt or (args.output_dir / "real_trained_agent_ckpt.pt")
    save_real_ckpt.parent.mkdir(parents=True, exist_ok=True)
    agent_real.save(str(save_real_ckpt))
    real_traj = evaluate_policy_trajectory(env_real, agent_real, seed=args.seed)

    _, env_mix, agent_mix, reward_cfg_mix = build_env_agent_reward(args.config, args.seed)
    _ = reward_cfg_mix
    agent_mix.load(str(args.agent_ckpt), map_location="cpu")
    mix_traj = evaluate_policy_trajectory(env_mix, agent_mix, seed=args.seed)

    plot_trajectories(real_traj, mix_traj, args.output_dir / "policy_shape_comparison.png")
    print(f"[Done] 保存到: {args.output_dir}")
    print(f"[Saved] 真实环境训练 agent: {save_real_ckpt}")


if __name__ == "__main__":
    main()