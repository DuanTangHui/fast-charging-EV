"""Run adaptive cycles with differential surrogate updates."""
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
from src.rl.trainers.trainer_adaptive_gp import AdaptiveConfig, train_adaptive_cycles
from src.surrogate.dataset import build_dataset
from src.surrogate.gp_combined import CombinedSurrogate
from src.surrogate.gp_differential import DifferentialSurrogate
from src.surrogate.gp_static import StaticSurrogate
from src.utils.config import load_config
from src.utils.logging import ensure_dir
from src.utils.seeds import set_global_seed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    config = load_config(args.config).data
    set_global_seed(config.get("seed"))

    env = build_pack_env(config["env"])
    reward_cfg = PaperRewardConfig(**config["reward"])

    static_ckpt = Path(config["logging"]["runs_dir"]) / "cycle0" / "static_surrogate.pt"
    if static_ckpt.exists():
        static_surrogate = torch.load(static_ckpt)
    else:
        static_surrogate = StaticSurrogate(
            input_dim=env.observation_space.shape[0] + 1,
            output_dim=env.observation_space.shape[0],
            hidden_sizes=config["surrogate"]["hidden_sizes"],
            ensemble_size=config["surrogate"]["ensemble_size"],
            lr=config["surrogate"]["learning_rate"],
        )
        transitions = []
        for _ in range(2):
            state, info = env.reset()
            done = False
            while not done:
                action = env.action_space.sample()
                next_state, _, terminated, truncated, _ = env.step(action)
                transitions.append((state, action, next_state - state))
                state = next_state
                done = terminated or truncated
        dataset = build_dataset(transitions)
        static_surrogate.fit(dataset, epochs=5)
    diff_surrogate = DifferentialSurrogate(
        input_dim=env.observation_space.shape[0] + 1,
        output_dim=env.observation_space.shape[0],
        hidden_sizes=config["surrogate"]["hidden_sizes"],
        ensemble_size=config["surrogate"]["ensemble_size"],
        lr=config["surrogate"]["learning_rate"],
    )
    combined = CombinedSurrogate(static_surrogate, diff_surrogate)

    run_dir = ensure_dir(Path(config["logging"]["runs_dir"]) / "adaptive")
    adaptive_cfg = AdaptiveConfig(**config["trainer"]["adaptive"])

    train_adaptive_cycles(
        env,
        static_surrogate,
        diff_surrogate,
        combined,
        reward_cfg,
        adaptive_cfg,
        str(run_dir),
        soh_enabled=config["soh_prior"]["enabled"],
        lambda_prior=config["soh_prior"]["lambda_prior"],
        theta_dim=config["soh_prior"]["theta_dim"],
        dummy_soh=config["soh_prior"]["dummy_soh"],
    )

    torch.save(diff_surrogate, run_dir / "diff_surrogate.pt")


if __name__ == "__main__":
    main()
