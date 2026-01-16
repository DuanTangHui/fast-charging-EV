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
from src.rl.actor_critic_ddpg import DDPGAgent, DDPGConfig
from src.rl.trainers.trainer_static_gp import Cycle0Config, train_cycle0
from src.surrogate.gp_static import StaticSurrogate
from src.utils.config import load_config
from src.utils.seeds import set_global_seed
from src.utils.logging import ensure_dir


def main() -> None:
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
        input_dim=env.observation_space.shape[0] + 1,
        output_dim=env.observation_space.shape[0] -1,  # 去掉 SOH 维度
        hidden_sizes=config["surrogate"]["hidden_sizes"],
        ensemble_size=config["surrogate"]["ensemble_size"],
        lr=config["surrogate"]["learning_rate"],
    )

    run_dir = ensure_dir(Path(config["logging"]["runs_dir"]) / "cycle0")
    cycle0_cfg = Cycle0Config(**config["trainer"]["cycle0"])

    train_cycle0(env, agent, reward_cfg, surrogate, cycle0_cfg, str(run_dir))

    torch.save(agent.actor.state_dict(), run_dir / "policy.pt")
    torch.save(surrogate, run_dir / "static_surrogate.pt")


if __name__ == "__main__":
    main()
