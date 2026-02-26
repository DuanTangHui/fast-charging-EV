from __future__ import annotations

from typing import Any

from .actor_critic_ddpg import DDPGAgent, DDPGConfig
from .actor_critic_td3 import TD3Agent, TD3Config
from .actor_critic_ppo import PPOAgent, PPOConfig


def build_agent_from_config(state_dim: int, action_dim: int, rl_config: dict[str, Any]):
    algo = str(rl_config.get("algorithm", "ddpg")).lower()
    common = dict(
        gamma=rl_config["gamma"],
        actor_lr=rl_config["actor_lr"],
        critic_lr=rl_config["critic_lr"],
        tau=rl_config.get("tau", 0.005),
        batch_size=rl_config["batch_size"],
        buffer_size=rl_config["buffer_size"],
        action_low=rl_config["action_low"],
        action_high=rl_config["action_high"],
    )

    if algo == "ddpg":
        return DDPGAgent(state_dim, action_dim, DDPGConfig(**common))
    if algo == "td3":
        extra = rl_config.get("td3", {})
        return TD3Agent(state_dim, action_dim, TD3Config(**common, **extra))
    if algo == "ppo":
        extra = rl_config.get("ppo", {})
        return PPOAgent(state_dim, action_dim, PPOConfig(**common, **extra))

    raise ValueError(f"Unsupported RL algorithm: {algo}")
