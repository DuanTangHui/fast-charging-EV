"""Train Cycle0 static surrogate and actor-critic policy."""
from __future__ import annotations

import argparse
import sys
import os
from pathlib import Path

# 获取当前脚本的绝对路径
current_dir = Path(__file__).resolve().parent
# 获取项目根目录 (即 scripts 的上一级)
root_dir = current_dir.parent
# 将根目录添加到 Python 搜索路径中
sys.path.append(str(root_dir))

import torch

from src.envs.liionpack_spme_pack_env import build_pack_env
from src.rewards.paper_reward import PaperRewardConfig
from src.rl.agent_factory import build_agent_from_config
from src.rl.trainers.trainer_static_gp import Cycle0Config, train_cycle0
from src.surrogate.gp_static import StaticSurrogate
from src.utils.config import load_config
from src.utils.seeds import set_global_seed
from src.utils.logging import ensure_dir

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to config file")
    # 增加两个参数，方便你灵活控制跑多少个种子，以及从哪个数字开始
    parser.add_argument("--num_seeds", type=int, default=50, help="Number of random seeds to run")
    parser.add_argument("--start_seed", type=int, default=30, help="Starting seed number")
    args = parser.parse_args()

    # 1. 加载一次配置
    config = load_config(args.config).data
    base_runs_dir = Path(config["logging"]["runs_dir"])
    cycle0_cfg = Cycle0Config(**config["trainer"]["cycle0"])

    # 2. 开始循环 100 个种子
    for i in range(args.num_seeds):
        # 计算当前循环使用的具体种子值 (例如：0, 1, 2... 99)
        current_seed = args.start_seed + i
        print(f"\n{'='*40}")
        print(f"Starting training for Seed: {current_seed} ({i+1}/{args.num_seeds})")
        print(f"{'='*40}")

        # 设置全局随机种子
        set_global_seed(current_seed)

        # ⚠️ 必须在循环内部重新构建 Env, Agent 和 Surrogate，保证每次都是从零开始
        env = build_pack_env(config["env"])
        reward_cfg = PaperRewardConfig(**config["reward"])
        agent = build_agent_from_config(
            state_dim=env.observation_space.shape[0],
            action_dim=1,
            rl_config=config["rl"],
        )

        surrogate = StaticSurrogate(
            input_dim=env.observation_space.shape[0] + 1 ,  # state(7) + action(1)
            output_dim=env.observation_space.shape[0] - 1,  # 预测 delta 的前 6 维
            hidden_sizes=config["surrogate"]["hidden_sizes"],
            ensemble_size=config["surrogate"]["ensemble_size"],
            lr=config["surrogate"]["learning_rate"],
        )

        # 3. 为当前种子创建一个独立的输出目录
        # 例如: logs/runs/cycle0_seed_0, logs/runs/cycle0_seed_1 ...
        run_dir = ensure_dir(base_runs_dir / f"cycle0_seed_{current_seed}")

        # 4. 执行训练
        train_cycle0(env, agent, reward_cfg, surrogate, cycle0_cfg, str(run_dir))

        # 5. 保存模型 (自动存入对应的 seed 文件夹)
        agent.save(str(run_dir / "agent_ckpt.pt"))
        torch.save(surrogate, run_dir / "static_surrogate.pt")
        
        print(f"Finished training for Seed: {current_seed}. Results saved in {run_dir}")

if __name__ == "__main__":
    main()