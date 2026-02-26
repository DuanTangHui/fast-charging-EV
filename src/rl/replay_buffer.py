"""Replay buffer for off-policy RL."""
from __future__ import annotations

from collections import deque
from typing import Deque, Tuple

import numpy as np


class ReplayBuffer:
    """Simple replay buffer."""

    def __init__(self, capacity: int) -> None:
        self.capacity = capacity
        self.buffer: Deque[Tuple[np.ndarray, np.ndarray, float, np.ndarray, bool]] = deque(maxlen=capacity)

    def push(self, state: np.ndarray, action: np.ndarray, reward: float, next_state: np.ndarray, done: bool) -> None:
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        idx = np.random.choice(len(self.buffer), size=batch_size, replace=False)
        states, actions, rewards, next_states, dones = zip(*(self.buffer[i] for i in idx))
        return (
            np.stack(states),
            np.stack(actions),
            np.array(rewards, dtype=np.float32),
            np.stack(next_states),
            np.array(dones, dtype=np.float32),
        )

    def state_dict(self) -> dict:
        # 将 deque 中的数据序列化成 list，里面每项是 (s, a, r, s_next, done)
        items = list(self.buffer)
        # 直接返回 list（其中 np.ndarray 本身可被 torch.save 序列化）
        return {
            "capacity": self.capacity,
            "items": items,
        }

    def load_state_dict(self, state: dict) -> None:
        self.capacity = int(state["capacity"])
        self.buffer = deque(maxlen=self.capacity)
        for (s, a, r, s_next, done) in state["items"]:
            # 确保类型正确（避免 pickle/torch load 后变成 list）
            s = np.asarray(s)
            a = np.asarray(a)
            s_next = np.asarray(s_next)
            r = float(r)
            done = bool(done)
            self.buffer.append((s, a, r, s_next, done))

    def __len__(self) -> int:
        return len(self.buffer)
