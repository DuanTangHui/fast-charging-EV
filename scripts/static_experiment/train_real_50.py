"""独立脚本：在真实物理仿真环境下，使用 50 个不同的随机种子分别训练 650 个 episode，并单独保存模型。"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# 确保脚本能找到项目根目录或 common 模块
current_dir = Path(__file__).resolve().parent
if str(current_dir) not in sys.path:
    sys.path.append(str(current_dir))

from common import (
    build_env_agent_reward,
    run_real_training_collect_style,
)

def main() -> None:
    parser = argparse.ArgumentParser(description="多随机种子真实环境批量训练脚本")
    parser.add_argument("--config", type=Path, default=Path("configs/pack_3p6s_spme.yaml"), help="配置文件路径")
    parser.add_argument("--output-dir", type=Path, default=Path("runs/real_agents_batch"), help="模型统一保存目录")
    parser.add_argument("--real-episodes", type=int, default=650, help="每个种子在真实环境中训练的 episode 数量")
    parser.add_argument("--num-seeds", type=int, default=50, help="需要跑的随机种子总数")
    parser.add_argument("--start-seed", type=int, default=36, help="起始随机种子")
    args = parser.parse_args()

    # 创建输出目录
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print("=== 开始批量训练真实环境模型 ===")
    print(f"总种子数: {args.num_seeds}, 起始种子: {args.start_seed}, 每个种子 Episodes: {args.real_episodes}")
    print(f"保存路径: {args.output_dir}")

    # 循环遍历指定的随机种子数量
    for i in range(args.num_seeds):
        current_seed = args.start_seed + i
        print(f"\n{'-'*50}")
        print(f"[{i+1}/{args.num_seeds}] 正在初始化并训练 Seed: {current_seed}")
        print(f"{'-'*50}")

        # 1. 为当前种子构建完全独立的 Environment, Agent 和 Reward
        # 注：build_env_agent_reward 内部会处理该种子的全局随机性设置
        cfg, env, agent, reward_cfg = build_env_agent_reward(args.config, current_seed)

        # 2. 在真实环境中执行训练 (650 个 episodes)
        run_real_training_collect_style(env, agent, reward_cfg, episodes=args.real_episodes)

        # 3. 动态命名并保存当前种子训练出的 Agent
        save_path = args.output_dir / f"real_agent_seed_{current_seed}.pt"
        agent.save(str(save_path))
        print(f"[Success] Seed {current_seed} 训练完成！模型已保存至: {save_path}")

    print(f"\n=== 恭喜！所有 {args.num_seeds} 个种子的训练已全部完成！ ===")
    print(f"所有模型文件均位于: {args.output_dir}")

if __name__ == "__main__":
    main()