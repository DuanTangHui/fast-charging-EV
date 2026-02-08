"""
电池组快充环境的基础接口（BasePackEnv）与观测容器（PackObservation）。

设计目标：
- 用一个统一的 PackObservation 把“cell级数组”聚合成“pack级标量观测”
- BasePackEnv 仅定义 action_space / observation_space 与 reset/step 接口
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import gymnasium as gym
import numpy as np


@dataclass
class PackObservation:
    """
    电池组层面的观测（用于 RL 输入）。

    字段含义：
    - SOC_pack:      pack 平均 SOC（0~1）
    - std_SOC:       cell SOC 的标准差（反映一致性/不均衡）
    - V_cell_max:    cell 端电压最大值（安全约束关键）
    - dV:            max(V_cell) - min(V_cell)（电压不一致性指标）
    - T_cell_max:    cell 温度最大值（安全约束关键）
    - T_cell_min:    cell 温度最小值（用于构造 dT 或判断分布）
    - I_prev:        上一步施加的电流（真实执行电流）
    """

    SOC_pack: float
    std_SOC: float
    V_cell_max: float
    dV: float
    T_cell_max: float
    T_cell_min: float
    I_prev: float

    def as_array(self) -> np.ndarray:
        """
        转成神经网络可直接输入的向量（dtype=float32）。

        注意：向量顺序必须与 observation_space.shape 对应，
        并且和你的训练/代理模型输入一致。
        """
        return np.array(
            [
                self.SOC_pack,
                self.std_SOC,
                self.V_cell_max,
                self.dV,
                self.T_cell_max,
                self.T_cell_min,
                self.I_prev,
            ],
            dtype=np.float32,
        )


class BasePackEnv(gym.Env):
    """
    电池组快充环境的抽象基类（Gymnasium 风格）。

    约定：
    - 动作为 1 维连续值：充电电流 I（A）
    - 这里默认“负值为充电”，动作范围 [-30, 0]
      若你后续要扩展为更大电流上限或双向电流，可以改 action_space。
    - 观测维度固定为 7（对应 PackObservation.as_array）
    """

    def __init__(self, dt: float, max_steps: int, v_max: float, t_max: float) -> None:
        # 仿真时间步长（秒）
        self.dt = float(dt)
        # 一个 episode 的最大步数（horizon）
        self.max_steps = int(max_steps)
        # 安全上限：最大单体电压（V）
        self.v_max = float(v_max)
        # 安全上限：最大单体温度（K）
        self.t_max = float(t_max)

        # 动作空间：充电电流 I（单位 A）
        # 负数表示充电；0 表示不充
        self.action_space = gym.spaces.Box(
            low=np.array([-30.0], dtype=np.float32),
            high=np.array([0.0], dtype=np.float32),
            shape=(1,),
            dtype=np.float32,
        )

        # 观测空间：7维 pack-level 向量
        self.observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(7,),
            dtype=np.float32,
        )

    def reset(self, *, seed: int | None = None, options: Dict | None = None) -> Tuple[np.ndarray, Dict]:
        """子类必须实现：返回 (obs, info)。"""
        raise NotImplementedError

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """
        子类必须实现：返回 (obs, reward, terminated, truncated, info)。

        你当前工程里 reward 多数在 trainer 内部算，这里可以先返回 0.0，
        或者后续把 reward 逻辑迁移到 env 内部。
        """
        raise NotImplementedError
