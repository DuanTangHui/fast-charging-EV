"""PPO implementation with continuous actions."""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import torch
from torch import nn

from .actor_critic_ddpg import RunningNormalizer, DEVICE


class PPOActor(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden_sizes: list[int], action_low: float, action_high: float) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        last_dim = state_dim
        for size in hidden_sizes:
            layers.extend([nn.Linear(last_dim, size), nn.ReLU()])
            last_dim = size
        layers.append(nn.Linear(last_dim, action_dim))
        self.mu_net = nn.Sequential(*layers)
        self.log_std = nn.Parameter(torch.zeros(action_dim))
        self.action_low = action_low
        self.action_high = action_high

    def dist(self, state: torch.Tensor) -> torch.distributions.Normal:
        mu = torch.tanh(self.mu_net(state))
        scale = (self.action_high - self.action_low) / 2.0
        bias = (self.action_high + self.action_low) / 2.0
        mu = mu * scale + bias
        std = torch.exp(self.log_std).clamp(min=1e-4, max=2.0)
        return torch.distributions.Normal(mu, std)


class ValueNet(nn.Module):
    def __init__(self, state_dim: int, hidden_sizes: list[int]) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        last_dim = state_dim
        for size in hidden_sizes:
            layers.extend([nn.Linear(last_dim, size), nn.ReLU()])
            last_dim = size
        layers.append(nn.Linear(last_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.net(state)


@dataclass
class PPOConfig:
    gamma: float
    actor_lr: float
    critic_lr: float
    tau: float
    batch_size: int
    buffer_size: int
    action_low: float
    action_high: float
    clip_ratio: float = 0.2
    ppo_epochs: int = 4
    gae_lambda: float = 0.95
    entropy_coef: float = 0.0


class PPOAgent:
    def __init__(self, state_dim: int, action_dim: int, config: PPOConfig) -> None:
        self.actor = PPOActor(state_dim, action_dim, [128, 128], config.action_low, config.action_high).to(DEVICE)
        self.critic = ValueNet(state_dim, [128, 128]).to(DEVICE)
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=config.actor_lr)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=config.critic_lr)
        self.config = config
        self.state_norm = RunningNormalizer(state_dim)
        self.is_on_policy = True
        self.buffer: List[dict] = []

    def act(self, state: np.ndarray) -> np.ndarray:
        s = np.asarray(state, dtype=np.float32).reshape(-1)
        s_norm = self.state_norm.normalize(s)
        s_t = torch.tensor(s_norm, dtype=torch.float32, device=DEVICE).unsqueeze(0)
        with torch.no_grad():
            dist = self.actor.dist(s_t)
            action = dist.sample()
        return action.squeeze(0).cpu().numpy().astype(np.float32)

    def observe(self, state: np.ndarray, action: np.ndarray, reward: float, next_state: np.ndarray, done: bool) -> None:
        self.state_norm.update(state)
        self.state_norm.update(next_state)
        s = torch.tensor(self.state_norm.normalize(state), dtype=torch.float32, device=DEVICE).unsqueeze(0)
        a = torch.tensor(action, dtype=torch.float32, device=DEVICE).unsqueeze(0)
        with torch.no_grad():
            dist = self.actor.dist(s)
            logp = dist.log_prob(a).sum(dim=-1).cpu().item()
            value = self.critic(s).cpu().item()
        self.buffer.append({
            "state": np.asarray(state, dtype=np.float32),
            "action": np.asarray(action, dtype=np.float32),
            "reward": float(reward),
            "done": float(done),
            "old_logp": float(logp),
            "value": float(value),
        })
        if len(self.buffer) > self.config.buffer_size:
            self.buffer = self.buffer[-self.config.buffer_size :]

    def update(self) -> Tuple[float, float]:
        if len(self.buffer) < self.config.batch_size:
            return 0.0, 0.0

        traj = self.buffer[-self.config.batch_size :]
        states = np.stack([t["state"] for t in traj])
        actions = np.stack([t["action"] for t in traj])
        rewards = np.array([t["reward"] for t in traj], dtype=np.float32)
        dones = np.array([t["done"] for t in traj], dtype=np.float32)
        old_logp = np.array([t["old_logp"] for t in traj], dtype=np.float32)
        values = np.array([t["value"] for t in traj], dtype=np.float32)

        returns = np.zeros_like(rewards)
        adv = np.zeros_like(rewards)
        last_gae = 0.0
        next_value = 0.0
        for t in reversed(range(len(rewards))):
            non_terminal = 1.0 - dones[t]
            delta = rewards[t] + self.config.gamma * next_value * non_terminal - values[t]
            last_gae = delta + self.config.gamma * self.config.gae_lambda * non_terminal * last_gae
            adv[t] = last_gae
            returns[t] = adv[t] + values[t]
            next_value = values[t]

        adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        s_t = torch.tensor(self.state_norm.normalize(states), dtype=torch.float32, device=DEVICE)
        a_t = torch.tensor(actions, dtype=torch.float32, device=DEVICE)
        old_logp_t = torch.tensor(old_logp, dtype=torch.float32, device=DEVICE)
        adv_t = torch.tensor(adv, dtype=torch.float32, device=DEVICE)
        ret_t = torch.tensor(returns, dtype=torch.float32, device=DEVICE).unsqueeze(-1)

        actor_loss_v = 0.0
        critic_loss_v = 0.0
        for _ in range(self.config.ppo_epochs):
            dist = self.actor.dist(s_t)
            logp = dist.log_prob(a_t).sum(dim=-1)
            ratio = torch.exp(logp - old_logp_t)
            clipped_ratio = torch.clamp(ratio, 1.0 - self.config.clip_ratio, 1.0 + self.config.clip_ratio)
            entropy = dist.entropy().sum(dim=-1).mean()
            actor_loss = -(torch.min(ratio * adv_t, clipped_ratio * adv_t).mean() + self.config.entropy_coef * entropy)

            value = self.critic(s_t)
            critic_loss = nn.functional.mse_loss(value, ret_t)

            self.actor_opt.zero_grad()
            actor_loss.backward()
            self.actor_opt.step()

            self.critic_opt.zero_grad()
            critic_loss.backward()
            self.critic_opt.step()

            actor_loss_v = float(actor_loss.item())
            critic_loss_v = float(critic_loss.item())

        self.buffer = []
        return actor_loss_v, critic_loss_v

    def save(self, path: str) -> None:
        torch.save({
            "actor": self.actor.state_dict(),
            "critic": self.critic.state_dict(),
            "actor_opt": self.actor_opt.state_dict(),
            "critic_opt": self.critic_opt.state_dict(),
            "state_norm": vars(self.state_norm),
            "buffer": self.buffer,
        }, path)

    def load(self, path: str, map_location: str | None = "cpu") -> None:
        ckpt = torch.load(path, map_location=map_location, weights_only=False)
        self.actor.load_state_dict(ckpt["actor"])
        self.critic.load_state_dict(ckpt["critic"])
        self.actor_opt.load_state_dict(ckpt["actor_opt"])
        self.critic_opt.load_state_dict(ckpt["critic_opt"])
        sn = ckpt["state_norm"]
        self.state_norm.count = int(sn["count"])
        self.state_norm.mean = np.array(sn["mean"], dtype=np.float64)
        self.state_norm.m2 = np.array(sn["m2"], dtype=np.float64)
        self.buffer = ckpt.get("buffer", [])
