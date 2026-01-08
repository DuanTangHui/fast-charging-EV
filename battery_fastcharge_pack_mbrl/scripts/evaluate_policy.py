"""Evaluate a saved policy on the environment."""
from __future__ import annotations

import argparse
from pathlib import Path

import torch

from src.envs.liionpack_spme_pack_env import build_pack_env
from src.evaluation.episode_rollout import rollout_env
from src.evaluation.reports import summarize_episode
from src.rewards.paper_reward import PaperRewardConfig
from src.rl.actor_critic_ddpg import Actor
from src.utils.config import load_config
from src.utils.logging import ensure_dir, log_metrics
from src.utils.plotting import plot_episode
from src.envs.observables import curve_from_infos


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--ckpt", required=True)
    args = parser.parse_args()

    config = load_config(args.config).data
    env = build_pack_env(config["env"])
    reward_cfg = PaperRewardConfig(**config["reward"])

    actor = Actor(
        state_dim=env.observation_space.shape[0],
        action_dim=1,
        hidden_sizes=[128, 128],
        action_low=config["rl"]["action_low"],
        action_high=config["rl"]["action_high"],
    )
    actor.load_state_dict(torch.load(args.ckpt, map_location="cpu"))

    def policy(state):
        with torch.no_grad():
            return actor(torch.tensor(state, dtype=torch.float32).unsqueeze(0)).numpy()[0]

    total_reward, infos = rollout_env(env, policy, reward_cfg)
    metrics = summarize_episode(infos)
    metrics["reward"] = total_reward

    run_dir = ensure_dir(Path(config["logging"]["runs_dir"]) / "evaluation")
    log_metrics(run_dir / "metrics.jsonl", metrics)
    curve = curve_from_infos(infos)
    plot_episode(curve, run_dir / "evaluation.png")


if __name__ == "__main__":
    main()
