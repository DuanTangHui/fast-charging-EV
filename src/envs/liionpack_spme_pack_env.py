"""Pack environment with liionpack/pybamm backend or toy fallback."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np

# --- 1. 导入检查 ---
try:
    import liionpack
    import pybamm
    _HAS_LIIONPACK = True
except ImportError:
    _HAS_LIIONPACK = False

from .aging_scenarios import AgingParams
from .base_env import BasePackEnv
from .observables import build_observation

@dataclass
class PackState:
    """Internal state."""
    soc: np.ndarray
    voltage: np.ndarray
    temperature: np.ndarray
    soh: np.ndarray
    i_prev: float
    t: float
    step: int

class LiionpackSPMEPackEnv(BasePackEnv):
    """Pack environment wrapper that actually uses Liionpack if available."""

    def __init__(
        self,
        dt: float,
        max_steps: int,
        v_max: float,
        t_max: float,
        pack_cells_p: int,
        pack_cells_s: int,
        terminate_on_violation: bool = True,
        use_physics_backend: bool = True, # 新增开关
    ) -> None:
        super().__init__(dt=dt, max_steps=max_steps, v_max=v_max, t_max=t_max)
        self.pack_cells_p = pack_cells_p
        self.pack_cells_s = pack_cells_s
        self.terminate_on_violation = terminate_on_violation
        
        # 决定是否使用真实物理引擎
        self.use_liionpack = _HAS_LIIONPACK and use_physics_backend
        
        self._rng = np.random.default_rng(0)
        self._aging = AgingParams(1.0, 1.0, 1.0)
        self._state: PackState | None = None
        
        # --- 2. Liionpack 初始化占位 ---
        self.netlist = None
        self.solver = None
        if self.use_liionpack:
            # 这里需要初始化 liionpack 的 netlist
            # 为了简单演示，我们打印一个日志，实际代码需要在这里 setup_circuit
            print(f"Initializing Liionpack backend ({pack_cells_p}p{pack_cells_s}s)...")
            self.netlist = liionpack.setup_circuit(
                Np=pack_cells_p, Ns=pack_cells_s, Rb=1e-4, Rc=1e-2, Ri=5e-2, V=3.6, I=0.0
            )
            # 定义参数 (Chemistry)
            self.parameter_values = pybamm.ParameterValues("Chen2020")
        
        # 用于 Toy 模型的固定异质性参数 (防止每一步都跳变)
        self._cell_resistance_bias = None 

    @property
    def n_cells(self) -> int:
        return self.pack_cells_p * self.pack_cells_s

    def set_aging(self, aging: AgingParams) -> None:
        self._aging = aging

    def reset(self, *, seed: int | None = None, options: Dict | None = None) -> Tuple[np.ndarray, Dict]:
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        
        # 生成初始 SOC 分布
        soc_init = self._rng.uniform(0.2, 0.3, size=self.n_cells)
        soh_init = np.ones(self.n_cells) * self._aging.capacity_scale
        
        if self.use_liionpack:
            # --- 3. Liionpack 重置逻辑 ---
            # Liionpack 通常不支持像 Toy 这样随意的 reset，需要重新构建 Solver
            self.solver = liionpack.Solver(
                netlist=self.netlist,
                parameter_values=self.parameter_values,
                dt=self.dt,
                # 传入初始 SOC
                initial_soc=np.mean(soc_init) # Liionpack 简化处理通常用均值启动
            )
            # 获取初始状态
            # 注意: 这里假设 solve 的第一步 output
            v_init = np.ones(self.n_cells) * 3.6 
            t_init = np.ones(self.n_cells) * 298.15
        else:
            # --- Toy 模型重置 ---
            # 关键修改：初始化电芯的“固有差异”，而不是在 step 里加噪声
            self._cell_resistance_bias = self._rng.normal(0, 0.005, size=self.n_cells)
            
            v_init = 3.5 + 0.1 * soc_init + self._rng.normal(0, 0.01, size=self.n_cells)
            t_init = np.ones(self.n_cells) * 298.15 + self._rng.normal(0, 0.5, size=self.n_cells)

        self._state = PackState(soc_init, v_init, t_init, soh_init, 0.0, 0.0, 0)
        
        obs = build_observation(soc_init, v_init, t_init, soh_init, 0.0).as_array()
        return obs, self._build_info(self._state, 0.0, False)

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        if self._state is None:
            raise RuntimeError("Environment must be reset before step.")
            
        # 限制电流动作
        current = float(np.clip(action[0], self.action_space.low[0], self.action_space.high[0]))
        state = self._state
        
        if self.use_liionpack:
            # --- 4. Liionpack 步进逻辑 ---
            # 调用 solver 前进一步
            # liionpack step 接口通常需要 input_data
            output = self.solver.step(current) 
            
            # 从 output 解析出电压、温度、SOC
            # 这里是伪代码，具体取决于 liionpack 版本返回的字典结构
            voltage = output["Terminal voltage [V]"]
            temperature = output["Volume-averaged cell temperature [K]"]
            soc = output["Measured SOC"] # 或者通过 capacity 计算
            soh = state.soh
            
        else:
            # --- 5. Toy 模型物理修正 ---
            dt_hours = self.dt / 3600.0
            capacity_Ah = 3.0 * self._aging.capacity_scale
            
            # SOC 更新 (库伦计数)
            soc = np.clip(state.soc + (current / capacity_Ah) * dt_hours, 0.0, 1.0)
            
            # 温度更新 (增加散热项)
            # Q_gen = I^2 * R (简化)
            # Q_cool = h * (T - T_amb)
            heat_gen = 0.05 * (current ** 2) * self._aging.resistance_scale * 1e-3 # 假设一些发热系数
            cooling = 0.01 * (state.temperature - 298.15) # 散热系数
            temperature = state.temperature + (heat_gen - cooling) * self.dt
            
            # 电压更新 (OCV + IR)
            # 使用固定 bias，而不是每次产生新的 rng.normal
            r_internal = 0.02 * self._aging.resistance_scale + self._cell_resistance_bias
            voltage = 3.0 + 1.2 * soc + (current * r_internal) 
            # 只有测量噪声才应该是在这一步加随机数，且幅度应该很小
            voltage += self._rng.normal(0, 0.001, size=self.n_cells)
            
            soh = state.soh

        # 更新状态
        next_state = PackState(soc, voltage, temperature, soh, current, state.t + self.dt, state.step + 1)
        self._state = next_state

        # 检查约束
        v_max_val = float(np.max(voltage))
        t_max_val = float(np.max(temperature))
        violation = v_max_val > self.v_max or t_max_val > self.t_max
        terminated = next_state.step >= self.max_steps or (self.terminate_on_violation and violation)
        
        obs = build_observation(soc, voltage, temperature, soh, current).as_array()
        info = self._build_info(next_state, current, violation)
        
        return obs, 0.0, terminated, False, info

    def _build_info(self, state: PackState, current: float, violation: bool) -> Dict:
        """Helper to build info dict."""
        return {
            "t": state.t,
            "I": current,
            "SOC_pack": float(np.mean(state.soc)),
            "V_cell_max": float(np.max(state.voltage)),
            "V_cell_min": float(np.min(state.voltage)),
            "T_cell_max": float(np.max(state.temperature)),
            "T_cell_min": float(np.min(state.temperature)),
            "dV": float(np.max(state.voltage) - np.min(state.voltage)),
            "dT": float(np.max(state.temperature) - np.min(state.temperature)),
            "terminated_reason": "violation" if violation else "time",
            "violation": violation,
        }