"""Pack environment with liionpack/pybamm backend or toy fallback."""
from __future__ import annotations
import numpy as np
import warnings


from dataclasses import dataclass
from typing import Dict, Optional, Tuple, List, Any
from .aging_scenarios import AgingParams
from .base_env import BasePackEnv
from .observables import build_observation

warnings.filterwarnings(
    "ignore",
    message=r"The 'lumped' thermal option with 'dimensionality' 0 now uses the parameters.*",
)

try:
    import liionpack as lp  # type: ignore  # noqa: F401
    import pybamm  # type: ignore  # noqa: F401
    _HAS_LIIONPACK = True
except Exception:
    lp = None  # type: ignore
    pybamm = None  # type: ignore
    _HAS_LIIONPACK = False



@dataclass
class PackState:
    """Internal state for the toy environment."""

    soc: np.ndarray
    voltage: np.ndarray
    temperature: np.ndarray
    soh: np.ndarray
    i_prev: float
    t: float
    step: int


class LiionpackSPMEPackEnv(BasePackEnv):
    """Pack environment wrapper with toy fallback when liionpack is unavailable."""

    def __init__(
        self,
        dt: float,
        max_steps: int,
        v_max: float,
        t_max: float,
        pack_cells_p: int,
        pack_cells_s: int,
        terminate_on_violation: bool = True,
        backend: str = "auto",
        
        parameter_set: str = "Chen2020",
        Rb: float = 1.5e-6,
        Rc: float = 1e-2,
        Ri_init: float = 5e-2,
        ocv_init: float = 4.0,
        nproc: int = 1,

        htc: float = 10.0,
    ) -> None:
        super().__init__(dt=dt, max_steps=max_steps, v_max=v_max, t_max=t_max)
        self.pack_cells_p = pack_cells_p
        self.pack_cells_s = pack_cells_s
        self.terminate_on_violation = terminate_on_violation
        self._rng = np.random.default_rng(0)
        self._aging = AgingParams(1.0, 1.0, 1.0)
        
        # ---------------- backend selection ----------------
        if backend == "auto":
            self._backend = "liionpack" if _HAS_LIIONPACK else "toy"
        else:
            self._backend = backend

        # ---------------- toy state ----------------
        self._state: Optional[PackState] = None

        # ---------------- liionpack state ----------------
        self._rm = None
        self._protocol: Optional[List[float]] = None
        self._terminal_voltage_guess: float = float(ocv_init * pack_cells_s)
        self._lp_step: int = 0
        self._lp_inputs: Dict[str, Any] = {}
        self._htc = float(htc)
        self._lp_inputs = {
            "Total heat transfer coefficient [W.m-2.K-1]": np.ones(self.n_cells) * self._htc
        }

        if self._backend == "liionpack":
            if not _HAS_LIIONPACK:
                raise RuntimeError("backend='liionpack' requested but liionpack/pybamm not available.")
            # 构建 3P6S 的电路拓扑 netlist
            self._netlist = lp.setup_circuit(
                Np=pack_cells_p,
                Ns=pack_cells_s,
                Rb=Rb,
                Rc=Rc,
                Ri=Ri_init,
                V=ocv_init,
                I=0.0,  # initial current guess, will be overwritten per step
            )
            # 加载 Chen2020 参数集
            self._parameter_values = pybamm.ParameterValues(parameter_set)

            # 用 experiment 的 period 来定义 liionpack stepping 的全局步长
            period_s = int(round(self.dt))
            if period_s <= 0:
                raise ValueError("dt must be >= 1 second for liionpack stepping.")
            self._experiment = pybamm.Experiment(
                operating_conditions=[f"Rest for {period_s} s"] * max_steps,
                period=f"{period_s} s",
            )

            # 选择 SPMe + lumped thermal 的仿真函数
            self._sim_func = lp.thermal_simulation

            # 创建求解管理器
            self._rm = lp.CasadiManager()
            self._nproc = int(nproc)

    @property
    def n_cells(self) -> int:
        return self.pack_cells_p * self.pack_cells_s
    # 将liionpack输出转换为cell向量 ：(T,n_cells)、(T,) --> (n_cells,)  
    def _extract_cell_vector(self, arr, n_cells: int, *, name: str) -> np.ndarray:
        """
        Convert liionpack step_output entry to a (n_cells,) float vector.

        step_output values are often shaped like:
        - (T, n_cells)
        - (T,) scalar over time
        - (T, 1)
        - scalar
        """
        x = np.asarray(arr, dtype=float)

        # take last time index if time-series
        if x.ndim >= 1:
            x_last = x[-1]
        else:
            x_last = x

        x_last = np.asarray(x_last, dtype=float)

        if x_last.ndim == 0:
            return np.ones(n_cells, dtype=float) * float(x_last)
        if x_last.ndim == 1:
            if x_last.size == n_cells:
                return x_last
            if x_last.size == 1:
                return np.ones(n_cells, dtype=float) * float(x_last[0])
            raise ValueError(f"{name} has unexpected size {x_last.size}, expected {n_cells} or 1.")
        if x_last.ndim == 2:
            # sometimes it's (n_cells, 1) or (1, n_cells)
            if x_last.shape == (n_cells, 1):
                return x_last[:, 0]
            if x_last.shape == (1, n_cells):
                return x_last[0, :]
            raise ValueError(f"{name} has unexpected shape {x_last.shape}.")
        raise ValueError(f"{name} has unsupported ndim={x_last.ndim}.")
    # 用“真实 cell current”做库仑计数更新 SOC
    def _update_soc_coulomb_counting(
        self,
        soc_prev: np.ndarray,
        i_cells: np.ndarray,
        dt_s: float,
        C_cell_Ah: float,
        *,
        soc_min: float = 0.0,
        soc_max: float = 1.0,
        sign_convention: str = "pybamm",  # "pybamm": I>0 discharge, I<0 charge
    ) -> Tuple[np.ndarray, float]:
        """
        Coulomb counting SOC update using *actual* cell currents from the model.

        Args:
            soc_prev: (n_cells,) previous SOC
            i_cells:  (n_cells,) actual cell currents [A] from step_output
            dt_s:     timestep [s]
            C_cell_Ah: nominal *cell* capacity [Ah] (NOT pack capacity)
            sign_convention:
                - "pybamm": I>0 discharge => SOC decreases; I<0 charge => SOC increases
                - "charge_positive": I>0 charge => SOC increases (use if your stack uses opposite sign)

        Returns:
            soc_new: (n_cells,) updated SOC
            soc_pack: mean SOC
        """
        dt_h = float(dt_s) / 3600.0
        C = float(C_cell_Ah)

        if C <= 0:
            raise ValueError(f"C_cell_Ah must be positive, got {C}.")

        if sign_convention == "pybamm":
            # discharge current reduces SOC
            delta = (-i_cells / C) * dt_h
        elif sign_convention == "charge_positive":
            delta = (i_cells / C) * dt_h
        else:
            raise ValueError(f"Unknown sign_convention={sign_convention}")

        soc_new = np.clip(soc_prev + delta, soc_min, soc_max)
        soc_pack = float(np.mean(soc_new))
        return soc_new, soc_pack
    
    def set_aging(self, aging: AgingParams) -> None:
        """Update aging parameters for subsequent resets."""
        self._aging = aging
     # ------------------------ liionpack helpers ------------------------
    
    def _lp_setup_on_reset(self, initial_soc: float) -> None:
        """
        Call manager.solve(..., setup_only=True) once per episode.
        """
        assert self._rm is not None
        # 我需要的输出变量
        self._requested_outputs = [
            "Terminal voltage [V]",  #单体端电压
            "Volume-averaged cell temperature [K]",  #每个 cell 的温度
        ]
        # 整个电池物理仿真图的“编译阶段”
        self._rm.solve(
            netlist=self._netlist,
            sim_func=self._sim_func,
            parameter_values=self._parameter_values,
            experiment=self._experiment,
            output_variables=self._requested_outputs,
            inputs=self._lp_inputs,
            nproc=self._nproc, # 可用cpu
            initial_soc=float(initial_soc),
            setup_only=True,  # important: only build internal structures once :contentReference[oaicite:6]{index=6}
        )

        # protocol buffer for ALL steps; we will overwrite protocol[k] each RL step
        # manager._step expects a flattened_protocol list. :contentReference[oaicite:7]{index=7}
        nsteps = int(self._rm.Nsteps)
        self._rm.protocol = np.zeros(nsteps, dtype=float) # 存放action：I的 每一步都记录下来
        self._rm.step = -1 #liionpack 内部当前时间索引
        self._lp_step = 0 # 你环境包装的当前 step索引
        
    def _lp_do_step(self, current: float) -> Dict[str, Any]:
        """
        Do one liionpack step and return extracted info.
        NOTE: extraction is best-effort because liionpack output containers can differ by version.
        """
        assert self._rm is not None 
        step = self._lp_step
        self._rm.protocol[step] = float(current)
        self._rm.step = step
        # vlims_ok boolean indicates if voltage limits are ok in liionpack stepping flow. :contentReference[oaicite:8]{index=8}
        vlims_ok = self._rm._step(step, updated_inputs={})
        self._lp_step += 1

        # ---- IMPORTANT ----
        # Different liionpack versions expose step outputs differently.
        # The quickest way: print(dir(self._rm)) once, inspect attributes, and map them here.
        # We'll provide placeholders so your env API stays consistent.
        info: Dict[str, Any] = {
            "vlims_ok": bool(vlims_ok),
            "lp_step": self._lp_step,
        }
        return info
    
    # ------------------------ gym.Env API ------------------------
    def reset(self, *, seed: int | None = None, options: Dict | None = None) -> Tuple[np.ndarray, Dict]:
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        if self._backend == "liionpack":
            # initial SoC: liionpack currently applies same initial_soc to all cells (doc note) :contentReference[oaicite:9]{index=9}
            initial_soc = float(self._rng.uniform(0.2, 0.3))
            self._lp_setup_on_reset(initial_soc)
            self._rm.protocol[0] = 0.0
            self._rm.step = 0
            _ = self._rm._step(0, updated_inputs={})
            self._lp_step = 0
            if hasattr(self._rm, "record_times") and len(self._rm.record_times) > 0:
                self._rm.record_times[0] = 0.0
            # For now build a “neutral” observation; you should replace with real extracted vectors
            so = self._rm.step_output()
            v_cells = np.asarray(so["Terminal voltage [V]"][-1, :], dtype=float)
            t_cells = np.asarray(so["Volume-averaged cell temperature [K]"][-1, :], dtype=float)
            v_pack = float(np.asarray(so["Pack terminal voltage [V]"])[-1])

            soc = np.ones(self.n_cells) * initial_soc
            soh = np.ones(self.n_cells) * self._aging.capacity_scale
            try:
                self._C_cell_Ah = float(self._parameter_values["Nominal cell capacity [A.h]"])
            except Exception:
                self._C_cell_Ah = 3.0  # 兜底，防止参数名不一致导
            # 让 self._state 有意义（后续 SOC proxy 用）
            self._state = PackState(soc, v_cells, t_cells, soh, 0.0, 0.0, 0)
            self._last_vmax = float(np.max(v_cells))
            obs = build_observation(soc, v_cells, t_cells, soh, 0.0).as_array()
            info = {
                "t": 0.0,
                "I": 0.0,
                "SOC_pack": float(np.mean(soc)),
                "V_cell_max": float(np.max(v_cells)),
                "V_cell_min": float(np.min(v_cells)),
                "T_cell_max": float(np.max(t_cells)),
                "T_cell_min": float(np.min(t_cells)),
                "V_pack": v_pack,
                "dV": float(v_cells.max() - v_cells.min()),
                "dT": float(t_cells.max() - t_cells.min()),
                "terminated_reason": "reset",
                "violation": False,
                "backend": "liionpack",
            }
            info.update({
                "I_pack_from_cells": 0.0,
                "I_cell_mean": 0.0,
                "I_cell_max": 0.0,
                "I_cell_min": 0.0,
            })
            return obs, info
        
        
        # ---- toy fallback (your original code) ----
        soc = self._rng.uniform(0.2, 0.3, size=self.n_cells)
        voltage = 3.5 + 0.1 * soc + self._rng.normal(0, 0.01, size=self.n_cells)
        temperature = 298 + self._rng.normal(0, 0.5, size=self.n_cells)
        soh = np.ones(self.n_cells) * self._aging.capacity_scale
        self._state = PackState(soc, voltage, temperature, soh, 0.0, 0.0, 0)
        obs = build_observation(soc, voltage, temperature, soh, 0.0).as_array()
        info = {
            "t": 0.0,
            "I": 0.0,
            "SOC_pack": float(np.mean(soc)),
            "V_cell_max": float(np.max(voltage)),
            "V_cell_min": float(np.min(voltage)),
            "T_cell_max": float(np.max(temperature)),
            "T_cell_min": float(np.min(temperature)),
            "dV": float(np.max(voltage) - np.min(voltage)),
            "dT": float(np.max(temperature) - np.min(temperature)),
            "terminated_reason": "reset",
            "violation": False,
            "backend": "toy",
        }
        return obs, info

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        # 获取policy的电流，裁剪到-20-0之间。
        current = float(np.clip(action[0], self.action_space.low[0], self.action_space.high[0]))
        if self._backend == "liionpack":
            # 1) step liionpack
            lp_info = self._lp_do_step(current)
       
            # 2) Extract real outputs from liionpack (USER outputs live in step_output, not rm.output)
            so = self._rm.step_output()  # provided by your version (you saw it in dir())
           
            v_pack = float(np.asarray(so["Pack terminal voltage [V]"])[-1])
            v_cells = np.asarray(so["Terminal voltage [V]"][-1, :], dtype=float)
            t_cells = np.asarray(so["Volume-averaged cell temperature [K]"][-1, :], dtype=float)
            dV_cells = float(v_cells.max() - v_cells.min())
            dT_cells = float(t_cells.max() - t_cells.min())
            # SOC: your liionpack outputs may not expose SOC directly.
            # For now, keep a simple coulomb-counting proxy so obs has a meaningful SOC signal.
            # ---- SOC via Coulomb Counting using *actual* cell currents ----
            if self._state is None:
                soc_prev = np.ones(self.n_cells, dtype=float) * 0.25
            else:
                soc_prev = self._state.soc

            # 读取模型实际 cell current（比用 action/current 更真实）
            i_cells = self._extract_cell_vector(so["Cell current [A]"], self.n_cells, name="Cell current [A]")
            # if self._lp_step < 10:
            #     print(f"[CHK] action_current={current:.3f}  mean(i_cells)={np.mean(i_cells):.3f}  "
            #         f"I_pack_est_s={(np.sum(i_cells)/self.pack_cells_s):.3f}")

            # cell capacity (Ah)
            C_cell_Ah = float(getattr(self, "_C_cell_Ah", None) or self._parameter_values["Nominal cell capacity [A.h]"])

            # 更新 SOC：默认 PyBaMM 约定 I>0 放电，I<0 充电
            soc, soc_pack = self._update_soc_coulomb_counting(
                soc_prev=soc_prev,
                i_cells=i_cells,
                dt_s=self.dt,
                C_cell_Ah=C_cell_Ah,
                sign_convention="pybamm",
            )

            soc_done = soc_pack >= 0.995  # 比 0.999 更稳
            if self._lp_step < 5:
            # 如果你动作是负值代表充电，SOC 应该不下降
                if current < -1e-6 and soc_pack < float(np.mean(soc_prev)) - 1e-6:
                    print("[WARN] SOC decreased during charging. Check sign convention (pybamm vs your action sign).")

            soh = np.ones(self.n_cells) * self._aging.capacity_scale
            
            I_sum = float(np.sum(i_cells))
            I_pack_est_s = I_sum / self.pack_cells_s          # ≈ pack current
            I_pack_est_p = float(self.pack_cells_p * np.mean(i_cells))  # 另一种等价估计
            # keep state for SOC proxy continuity
            self._state = PackState(soc, v_cells, t_cells, soh, current, self._lp_step * self.dt, self._lp_step)

            obs = build_observation(soc, v_cells, t_cells, soh, current).as_array()

            v_max = float(np.max(v_cells))
            v_min = float(np.min(v_cells))
            t_max = float(np.max(t_cells))
            t_min = float(np.min(t_cells))
            # 安全约束 （是否违约）
            violation = (v_max > self.v_max) or (t_max > self.t_max) or (not lp_info["vlims_ok"])
            # 终止条件  
            terminated = (
                (self._lp_step >= self.max_steps)
                or soc_done
                or (self.terminate_on_violation and violation)
            )
            truncated = False
            if violation:
                reason = "violation"
            elif soc_done:
                reason = "soc_full"
            elif self._lp_step >= self.max_steps:
                reason = "horizon"
            else:
                reason = "running"
            self._last_vmax = v_max

            info = {
                "t": float(self._lp_step * self.dt),                 # [s] 当前仿真时间（第 lp_step 步 * dt）
    
                # 电流相关（重要：区分“动作setpoint” vs “模型真实执行”）
                "I": current,                                        # [A] 动作电流（你传入/clip后的 setpoint）
                "I_pack_est": I_pack_est_s,   # 推荐用这个作为“实际执行电流”
                "I_pack_est_p": I_pack_est_p,             # 可选
                "I_cell_mean": float(np.mean(i_cells)),              # [A] cell 电流均值（理想均匀分流时≈I_pack/18）
                "I_cell_max": float(np.max(i_cells)),                # [A] cell 最大电流（看是否有分流异常）
                "I_cell_min": float(np.min(i_cells)),                # [A] cell 最小电流
                
                # SOC
                "SOC_pack": soc_pack,                      # [-] pack 平均 SOC（你用库仑计数得到的 cell SOC 均值）
                
                # 电压/温度（cell级统计 & pack电压）
                "V_cell_max": v_max,                                 # [V] 单体端电压最大值（安全约束关键）
                "V_cell_min": v_min,                                 # [V] 单体端电压最小值
                "T_cell_max": t_max,                                 # [K] 单体温度最大值（安全约束关键）
                "T_cell_min": t_min,                                 # [K] 单体温度最小值
                "V_pack": v_pack,                                    # [V] pack 端电压（通常≈串联电芯电压和）

                # pack 内不一致性指标（你 trainer 需要 dV/dT）
                "dV": dV_cells,                                      # [V] max(V_cell) - min(V_cell)
                "dT": dT_cells,                                      # [K] max(T_cell) - min(T_cell)

                # 终止与安全
                "terminated_reason": reason,                          # str 终止原因：violation / soc_full / horizon / running
                "violation": violation,                               # bool 是否违反约束（电压/温度/或 vlims_ok=False）
                "backend": "liionpack",                               # str 后端标记

                # liionpack内部信息
                **lp_info,                                            # e.g. vlims_ok, lp_step
            }
            # if self._lp_step < 10:
            #     print("raw_action=", action, " clipped_current=", current, " bounds=", self.action_space.low[0], self.action_space.high[0])
            self._last_info = info

            return obs, 0.0, terminated, truncated, info
        if self._state is None:
            raise RuntimeError("Environment must be reset before step.")
        state = self._state
        dt_hours = self.dt / 3600.0
        capacity = 3.0 * self._aging.capacity_scale
        soc = np.clip(state.soc + (current / capacity) * dt_hours, 0.0, 1.0)
        voltage = 3.0 + 1.2 * soc - 0.02 * current * self._aging.resistance_scale
        voltage += self._rng.normal(0, 0.005, size=self.n_cells)
        temperature = state.temperature + (0.05 * current * self._aging.thermal_scale)
        temperature += self._rng.normal(0, 0.1, size=self.n_cells)
        soh = state.soh

        state = PackState(soc, voltage, temperature, soh, current, state.t + self.dt, state.step + 1)
        self._state = state

        obs = build_observation(soc, voltage, temperature, soh, current).as_array()
        v_max = float(np.max(voltage))
        t_max = float(np.max(temperature))
        violation = v_max > self.v_max or t_max > self.t_max
        terminated = state.step >= self.max_steps or (self.terminate_on_violation and violation)
        truncated = False
        info = {
            "t": state.t,
            "I": current,
            "SOC_pack": float(np.mean(soc)),
            "V_cell_max": v_max,
            "V_cell_min": float(np.min(voltage)),
            "T_cell_max": t_max,
            "T_cell_min": float(np.min(temperature)),
            "dV": float(np.max(voltage) - np.min(voltage)),
            "dT": float(np.max(temperature) - np.min(temperature)),
            "terminated_reason": "violation" if violation else "time",
            "violation": violation,
            "backend": "toy",
        }
        return obs, 0.0, terminated, truncated, info


def build_pack_env(config: Dict) -> LiionpackSPMEPackEnv:
    """Construct the pack environment (liionpack or toy fallback)."""

    env = LiionpackSPMEPackEnv(
        dt=float(config["dt"]),
        max_steps=int(config["max_steps"]),
        v_max=float(config["v_max"]),
        t_max=float(config["t_max"]),
        pack_cells_p=int(config["pack_cells_p"]),
        pack_cells_s=int(config["pack_cells_s"]),
        terminate_on_violation=bool(config.get("terminate_on_violation", True)),
        backend=str(config.get("backend", "auto")),
        parameter_set=str(config.get("parameter_set", "Chen2020")),
        Rb=float(config.get("Rb", 1.5e-6)),
        Rc=float(config.get("Rc", 1e-2)),
        Ri_init=float(config.get("Ri_init", 5e-2)),
        ocv_init=float(config.get("ocv_init", 4.0)),
        nproc=int(config.get("nproc", 1)),
        htc=float(config.get("htc", 10.0)),
    )
    return env
