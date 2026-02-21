"""TD3-style actor-critic implementation."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np
import torch
from torch import nn

from .actor_critic_ddpg import Actor, Critic, RunningNormalizer, DEVICE
from .replay_buffer import ReplayBuffer


@dataclass
class TD3Config:
    gamma: float
    actor_lr: float
    critic_lr: float
    tau: float
    batch_size: int
    buffer_size: int
    action_low: float
    action_high: float
    policy_noise: float = 0.2
    noise_clip: float = 0.5
    policy_delay: int = 2


class TD3Agent:
    def __init__(self, state_dim: int, action_dim: int, config: TD3Config) -> None:
        self.actor = Actor(state_dim, action_dim, [128, 128], config.action_low, config.action_high).to(DEVICE)
        self.actor_target = Actor(state_dim, action_dim, [128, 128], config.action_low, config.action_high).to(DEVICE)

        self.critic1 = Critic(state_dim, action_dim, [128, 128]).to(DEVICE)
        self.critic2 = Critic(state_dim, action_dim, [128, 128]).to(DEVICE)
        self.critic1_target = Critic(state_dim, action_dim, [128, 128]).to(DEVICE)
        self.critic2_target = Critic(state_dim, action_dim, [128, 128]).to(DEVICE)

        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic1_target.load_state_dict(self.critic1.state_dict())
        self.critic2_target.load_state_dict(self.critic2.state_dict())

        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=config.actor_lr)
        self.critic1_opt = torch.optim.Adam(self.critic1.parameters(), lr=config.critic_lr)
        self.critic2_opt = torch.optim.Adam(self.critic2.parameters(), lr=config.critic_lr)

        self.config = config
        self.buffer = ReplayBuffer(config.buffer_size)
        self.state_norm = RunningNormalizer(state_dim)
        self.is_on_policy = False
        self._update_step = 0

    def act(self, state: np.ndarray) -> np.ndarray:
        state = np.asarray(state, dtype=np.float32).reshape(-1)
        state_norm = self.state_norm.normalize(state)
        state_t = torch.tensor(state_norm, dtype=torch.float32, device=DEVICE).unsqueeze(0)
        with torch.no_grad():
            action = self.actor(state_t).cpu().numpy()[0]
        return action.astype(np.float32)

    def observe(self, state: np.ndarray, action: np.ndarray, reward: float, next_state: np.ndarray, done: bool) -> None:
        self.state_norm.update(state)
        self.state_norm.update(next_state)
        self.buffer.push(state, action, reward, next_state, done)

    def update(self) -> Tuple[float, float]:
        if len(self.buffer) < self.config.batch_size:
            return 0.0, 0.0

        self._update_step += 1
        states, actions, rewards, next_states, dones = self.buffer.sample(self.config.batch_size)

        states_t = torch.tensor(self.state_norm.normalize(states), dtype=torch.float32, device=DEVICE)
        actions_t = torch.tensor(actions, dtype=torch.float32, device=DEVICE)
        rewards_t = torch.tensor(rewards, dtype=torch.float32, device=DEVICE).unsqueeze(-1)
        next_states_t = torch.tensor(self.state_norm.normalize(next_states), dtype=torch.float32, device=DEVICE)
        dones_t = torch.tensor(dones, dtype=torch.float32, device=DEVICE).unsqueeze(-1)

        with torch.no_grad():
            noise = (torch.randn_like(actions_t) * self.config.policy_noise).clamp(-self.config.noise_clip, self.config.noise_clip)
            next_actions = self.actor_target(next_states_t) + noise
            next_actions = next_actions.clamp(self.config.action_low, self.config.action_high)

            q1_next = self.critic1_target(next_states_t, next_actions)
            q2_next = self.critic2_target(next_states_t, next_actions)
            q_next = torch.minimum(q1_next, q2_next)
            target_q = rewards_t + self.config.gamma * (1.0 - dones_t) * q_next

        q1 = self.critic1(states_t, actions_t)
        q2 = self.critic2(states_t, actions_t)
        critic_loss = nn.functional.mse_loss(q1, target_q) + nn.functional.mse_loss(q2, target_q)

        self.critic1_opt.zero_grad()
        self.critic2_opt.zero_grad()
        critic_loss.backward()
        self.critic1_opt.step()
        self.critic2_opt.step()

        actor_loss_value = 0.0
        if self._update_step % self.config.policy_delay == 0:
            actor_loss = -self.critic1(states_t, self.actor(states_t)).mean()
            self.actor_opt.zero_grad()
            actor_loss.backward()
            self.actor_opt.step()
            self._soft_update(self.actor, self.actor_target)
            self._soft_update(self.critic1, self.critic1_target)
            self._soft_update(self.critic2, self.critic2_target)
            actor_loss_value = float(actor_loss.item())

        return actor_loss_value, float(critic_loss.item())

    def _soft_update(self, source: nn.Module, target: nn.Module) -> None:
        for target_param, source_param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - self.config.tau) + source_param.data * self.config.tau)

    def save(self, path: str) -> None:
        ckpt = {
            "actor": self.actor.state_dict(),
            "actor_target": self.actor_target.state_dict(),
            "critic1": self.critic1.state_dict(),
            "critic2": self.critic2.state_dict(),
            "critic1_target": self.critic1_target.state_dict(),
            "critic2_target": self.critic2_target.state_dict(),
            "actor_opt": self.actor_opt.state_dict(),
            "critic1_opt": self.critic1_opt.state_dict(),
            "critic2_opt": self.critic2_opt.state_dict(),
            "state_norm": vars(self.state_norm),
            "replay": self.buffer.state_dict(),
            "update_step": self._update_step,
        }
        torch.save(ckpt, path)

    def load(self, path: str, map_location: str | None = "cpu") -> None:
        ckpt = torch.load(path, map_location=map_location)
        self.actor.load_state_dict(ckpt["actor"])
        self.actor_target.load_state_dict(ckpt["actor_target"])
        self.critic1.load_state_dict(ckpt["critic1"])
        self.critic2.load_state_dict(ckpt["critic2"])
        self.critic1_target.load_state_dict(ckpt["critic1_target"])
        self.critic2_target.load_state_dict(ckpt["critic2_target"])
        self.actor_opt.load_state_dict(ckpt["actor_opt"])
        self.critic1_opt.load_state_dict(ckpt["critic1_opt"])
        self.critic2_opt.load_state_dict(ckpt["critic2_opt"])

        sn = ckpt["state_norm"]
        self.state_norm.count = int(sn["count"])
        self.state_norm.mean = np.array(sn["mean"], dtype=np.float64)
        self.state_norm.m2 = np.array(sn["m2"], dtype=np.float64)

        if "replay" in ckpt and ckpt["replay"] is not None:
            self.buffer.load_state_dict(ckpt["replay"])
        self._update_step = int(ckpt.get("update_step", 0))
