"""静态代理模型有效性分析实验三：
证明无模型和本文提出的基于模型的强化学习方法都在前300个 episode 内收敛到了相似的最优解
通过训练过程动态曲线进行了严密的论证：

记录了 650 个 episode 的完整训练过程：
1.累积回报：展示了智能体在每个 episode 中获得的总奖励。
2.充电时间：展示了智能体完成充电任务所需的时间。
3.电压违规程度：展示了智能体在充电过程中违反电压安全约束的程度。
4.温度违规程度：展示了智能体在充电过程中违反温度安全约束的程度。

"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd

current_dir = Path(__file__).resolve().parent
if str(current_dir) not in sys.path:
    sys.path.append(str(current_dir))

from common import (
    build_env_agent_reward,
    fit_static_surrogate,
    run_real_training_collect_style,
    run_surrogate_training,
    save_metrics_csv,
)


def plot_metric_curves(df_real: pd.DataFrame, df_mix: pd.DataFrame, output: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
    settings = [
        ("total_reward", "累积回报"),
        ("charge_time_s", "充电时间 (s)"),
        ("voltage_violation", "电压违规程度 (V)") ,
        ("temperature_violation", "温度违规程度 (K)"),
    ]
    for ax, (col, title) in zip(axes.ravel(), settings):
        ax.plot(df_real["episode"], df_real[col], label="真实基线", linewidth=1.8)
        ax.plot(df_mix["episode"], df_mix[col], label="混合方案", linewidth=1.8, linestyle="--")
        ax.set_title(title)
        ax.grid(alpha=0.3)
    axes[1, 0].set_xlabel("Episode Number")
    axes[1, 1].set_xlabel("Episode Number")
    axes[0, 0].legend()
    plt.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output, dpi=220)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/pack_3p6s_spme.yaml"))
    parser.add_argument("--output-dir", type=Path, default=Path("runs/static_experiment/exp2_training_process_1000"))
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--episodes", type=int, default=1000)
    parser.add_argument("--mix-real-warmup", type=int, default=50)
    parser.add_argument("--surrogate-epochs", type=int, default=25)
    args = parser.parse_args()

    cfg, env_real, agent_real, reward_cfg = build_env_agent_reward(args.config, args.seed)
    real_metrics, _ = run_real_training_collect_style(env_real, agent_real, reward_cfg, episodes=args.episodes)
    save_metrics_csv(real_metrics, args.output_dir / "real_baseline_metrics.csv")

    _, env_mix, agent_mix, reward_cfg_mix = build_env_agent_reward(args.config, args.seed)
    warm_metrics, transitions = run_real_training_collect_style(env_mix, agent_mix, reward_cfg_mix, episodes=args.mix_real_warmup)
    surrogate = fit_static_surrogate(env_mix, transitions, cfg, epochs=args.surrogate_epochs)
    remain = max(0, args.episodes - args.mix_real_warmup)
    mix_sur_metrics = run_surrogate_training(env_mix, agent_mix, surrogate, reward_cfg_mix, episodes=remain, rollouts_per_episode=3, updates_per_episode=50)

    # episode 编号连续化
    for idx, m in enumerate(mix_sur_metrics, start=args.mix_real_warmup + 1):
        m.episode = idx
    mix_metrics = warm_metrics + mix_sur_metrics
    save_metrics_csv(mix_metrics, args.output_dir / "hybrid_metrics.csv")

    plot_metric_curves(pd.DataFrame([m.__dict__ for m in real_metrics]), pd.DataFrame([m.__dict__ for m in mix_metrics]), args.output_dir / "training_process_1000_compare.png")
    print(f"[Done] 保存到: {args.output_dir}")


if __name__ == "__main__":
    main()