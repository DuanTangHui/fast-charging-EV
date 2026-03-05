"""DDPG-style actor-critic implementation."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np
import torch
from torch import nn

from .replay_buffer import ReplayBuffer

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class RunningNormalizer:
    """
    状态归一化：在线估计 mean/std，用于将状态标准化输入网络。

    注意：
    - early 阶段 count 很小，直接返回原值（避免除零/不稳定）
    - 支持输入 shape:
        - (dim,) 单条状态
        - (batch, dim) 批量状态
    """

    def __init__(self, dim: int, min_std: float = 1e-3) -> None:
        self.dim = dim
        self.min_std = min_std
        self.count = 0
        self.mean = np.zeros(dim, dtype=np.float64)
        self.m2 = np.zeros(dim, dtype=np.float64)

    def update(self, x: np.ndarray) -> None:
        x_arr = np.asarray(x, dtype=np.float64).reshape(-1)
        if x_arr.shape[0] != self.dim:
            raise ValueError(f"Expected state dim {self.dim}, got {x_arr.shape[0]}")
        self.count += 1
        delta = x_arr - self.mean
        self.mean += delta / self.count
        delta2 = x_arr - self.mean
        self.m2 += delta * delta2

    def update_batch(self, batch: np.ndarray) -> None:
        for item in np.asarray(batch):
            self.update(item)

    def normalize(self, x: np.ndarray) -> np.ndarray:
        if self.count < 2:
            return np.asarray(x, dtype=np.float32)
        var = self.m2 / (self.count - 1)
        std = np.sqrt(np.maximum(var, self.min_std ** 2))
        return ((np.asarray(x, dtype=np.float32) - self.mean.astype(np.float32)) / std.astype(np.float32))

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
    """  DDPG 智能体：actor/critic + target 网络 + 经验回放 + 状态归一化。"""

    def __init__(self, state_dim: int, action_dim: int, config: DDPGConfig) -> None:
         # 网络
        self.actor = Actor(state_dim, action_dim, [128, 128], config.action_low, config.action_high).to(DEVICE)
        self.critic = Critic(state_dim, action_dim, [128, 128]).to(DEVICE)
        self.target_actor = Actor(state_dim, action_dim, [128, 128], config.action_low, config.action_high).to(DEVICE)
        self.target_critic = Critic(state_dim, action_dim, [128, 128]).to(DEVICE)
        # target 初始化
        self.target_actor.load_state_dict(self.actor.state_dict())
        self.target_critic.load_state_dict(self.critic.state_dict())

        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=config.actor_lr)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=config.critic_lr)

        self.config = config
        # 回放与归一化
        self.buffer = ReplayBuffer(config.buffer_size)
        self.state_norm = RunningNormalizer(state_dim)
        self.is_on_policy = False

    def act(self, state: np.ndarray) -> np.ndarray:
        """ 根据当前状态选择动作（只做归一化，不更新统计量）。
        统计量更新放在 observe()，避免“先验泄漏”与重复更新。"""

        state = np.asarray(state, dtype=np.float32).reshape(-1)
        state_norm = self.state_norm.normalize(state)
        state_tensor = torch.tensor(state_norm, dtype=torch.float32,
                            device=DEVICE).unsqueeze(0)
        with torch.no_grad():
            action = self.actor(state_tensor).cpu().numpy()[0]
        return action.astype(np.float32)
    
    def observe(self, state: np.ndarray, action: np.ndarray, reward: float, next_state: np.ndarray, done: bool) -> None:
        """
        存储 transition，并更新归一化统计量。
        推荐：同时更新 state 和 next_state，使均值/方差更稳定。
        """
        self.state_norm.update(state)
        self.state_norm.update(next_state)
        self.buffer.push(state, action, reward, next_state, done)

    def update(self) -> Tuple[float, float]:
        """从 replay buffer 采样，更新 actor/critic，并软更新 target 网络。"""

        if len(self.buffer) < self.config.batch_size:
            return 0.0, 0.0
        states, actions, rewards, next_states, dones = self.buffer.sample(self.config.batch_size)
        states_norm = self.state_norm.normalize(states)
        next_states_norm = self.state_norm.normalize(next_states)
        states_t = torch.tensor(states_norm, dtype=torch.float32, device=DEVICE)
        actions_t = torch.tensor(actions, dtype=torch.float32, device=DEVICE)
        rewards_t = torch.tensor(rewards, dtype=torch.float32,
                                device=DEVICE).unsqueeze(-1)
        next_states_t = torch.tensor(next_states_norm, dtype=torch.float32,
                                    device=DEVICE)
        dones_t = torch.tensor(dones, dtype=torch.float32,
                            device=DEVICE).unsqueeze(-1)
         # -------- critic 目标：y = r + gamma*(1-done)*Q'(s', a') --------
        with torch.no_grad():
            next_actions = self.target_actor(next_states_t)
            target_q = self.target_critic(next_states_t, next_actions)
            y = rewards_t + self.config.gamma * (1 - dones_t) * target_q
        # 更新critic
        q_val = self.critic(states_t, actions_t)
        critic_loss = torch.mean((q_val - y) ** 2)

        self.critic_opt.zero_grad()
        critic_loss.backward()
        self.critic_opt.step()

        # 更新actor   
        actor_loss = -self.critic(states_t, self.actor(states_t)).mean()

        self.actor_opt.zero_grad()
        actor_loss.backward()
        self.actor_opt.step()

        # 软更新目标网络target = (1-tau)*target + tau*source 所以软更新，目标网络更新很慢
        self._soft_update(self.actor, self.target_actor)
        self._soft_update(self.critic, self.target_critic)

        return float(actor_loss.item()), float(critic_loss.item())

    def _soft_update(self, source: nn.Module, target: nn.Module) -> None:
        for target_param, source_param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(
                target_param.data * (1.0 - self.config.tau) + source_param.data * self.config.tau
            )

    def save(self, path: str) -> None:
        """Save a full checkpoint for continuing training (Algorithm 2)."""
        ckpt = {
            "actor": self.actor.state_dict(),
            "critic": self.critic.state_dict(),
            "target_actor": self.target_actor.state_dict(),
            "target_critic": self.target_critic.state_dict(),
            "actor_opt": self.actor_opt.state_dict(),
            "critic_opt": self.critic_opt.state_dict(),
            "config": self.config,
            "state_norm": {
                "dim": self.state_norm.dim,
                "min_std": self.state_norm.min_std,
                "count": self.state_norm.count,
                "mean": self.state_norm.mean,
                "m2": self.state_norm.m2,
            },
            "replay": self.buffer.state_dict(),  # 依赖你在 replay_buffer.py 加的方法
        }
        torch.save(ckpt, path)

    def load(self, path: str, map_location: str | None = "cpu") -> None:
        """Load a full checkpoint for continuing training (Algorithm 2)."""
        ckpt = torch.load(path, map_location=map_location, weights_only=False)

        self.actor.load_state_dict(ckpt["actor"])
        self.critic.load_state_dict(ckpt["critic"])
        self.target_actor.load_state_dict(ckpt["target_actor"])
        self.target_critic.load_state_dict(ckpt["target_critic"])
        self.actor_opt.load_state_dict(ckpt["actor_opt"])
        self.critic_opt.load_state_dict(ckpt["critic_opt"])

        sn = ckpt["state_norm"]
        # 直接恢复 state_norm 内部统计量
        self.state_norm.count = int(sn["count"])
        self.state_norm.mean = np.array(sn["mean"], dtype=np.float64)
        self.state_norm.m2 = np.array(sn["m2"], dtype=np.float64)

        # restore replay buffer
        if "replay" in ckpt and ckpt["replay"] is not None:
            self.buffer.load_state_dict(ckpt["replay"])