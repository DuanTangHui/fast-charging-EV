"""DDPG-style actor-critic implementation."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np
import torch
from torch import nn

from .replay_buffer import ReplayBuffer


class Actor(nn.Module):
    """Actor network mapping states to actions."""

    def __init__(self, state_dim: int, action_dim: int, hidden_sizes: list[int], action_low: float, action_high: float) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        last_dim = state_dim
        for size in hidden_sizes:
            layers.append(nn.Linear(last_dim, size))
            layers.append(nn.ReLU())
            last_dim = size
        layers.append(nn.Linear(last_dim, action_dim))
        self.net = nn.Sequential(*layers)
        self.action_low = action_low
        self.action_high = action_high

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        action = torch.tanh(self.net(state))
        scale = (self.action_high - self.action_low) / 2.0
        bias = (self.action_high + self.action_low) / 2.0
        return action * scale + bias


class Critic(nn.Module):
    """Critic network mapping state-action to value."""

    def __init__(self, state_dim: int, action_dim: int, hidden_sizes: list[int]) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        last_dim = state_dim + action_dim
        for size in hidden_sizes:
            layers.append(nn.Linear(last_dim, size))
            layers.append(nn.ReLU())
            last_dim = size
        layers.append(nn.Linear(last_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([state, action], dim=-1))


@dataclass
class DDPGConfig:
    """Configuration for DDPG."""

    gamma: float
    actor_lr: float
    critic_lr: float
    tau: float
    batch_size: int
    buffer_size: int
    action_low: float
    action_high: float


class DDPGAgent:
    """DDPG agent with target networks."""

    def __init__(self, state_dim: int, action_dim: int, config: DDPGConfig) -> None:
        self.actor = Actor(state_dim, action_dim, [128, 128], config.action_low, config.action_high)
        self.critic = Critic(state_dim, action_dim, [128, 128])
        self.target_actor = Actor(state_dim, action_dim, [128, 128], config.action_low, config.action_high)
        self.target_critic = Critic(state_dim, action_dim, [128, 128])
        self.target_actor.load_state_dict(self.actor.state_dict())
        self.target_critic.load_state_dict(self.critic.state_dict())
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=config.actor_lr)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=config.critic_lr)
        self.config = config
        self.buffer = ReplayBuffer(config.buffer_size)

    def act(self, state: np.ndarray) -> np.ndarray:
        """Select action from actor network."""

        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            action = self.actor(state_tensor).numpy()[0]
        return action

    def update(self) -> Tuple[float, float]:
        """Update networks from replay buffer."""

        if len(self.buffer) < self.config.batch_size:
            return 0.0, 0.0
        states, actions, rewards, next_states, dones = self.buffer.sample(self.config.batch_size)
        states_t = torch.tensor(states, dtype=torch.float32)
        actions_t = torch.tensor(actions, dtype=torch.float32)
        rewards_t = torch.tensor(rewards, dtype=torch.float32).unsqueeze(-1)
        next_states_t = torch.tensor(next_states, dtype=torch.float32)
        dones_t = torch.tensor(dones, dtype=torch.float32).unsqueeze(-1)

        with torch.no_grad():
            next_actions = self.target_actor(next_states_t)
            target_q = self.target_critic(next_states_t, next_actions)
            y = rewards_t + self.config.gamma * (1 - dones_t) * target_q

        q_val = self.critic(states_t, actions_t)
        critic_loss = torch.mean((q_val - y) ** 2)
        self.critic_opt.zero_grad()
        critic_loss.backward()
        self.critic_opt.step()

        actor_loss = -self.critic(states_t, self.actor(states_t)).mean()
        self.actor_opt.zero_grad()
        actor_loss.backward()
        self.actor_opt.step()

        self._soft_update(self.actor, self.target_actor)
        self._soft_update(self.critic, self.target_critic)
        return float(actor_loss.item()), float(critic_loss.item())

    def _soft_update(self, source: nn.Module, target: nn.Module) -> None:
        for target_param, source_param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(
                target_param.data * (1.0 - self.config.tau) + source_param.data * self.config.tau
            )
