"""
只保留 liionpack/pybamm 后端的 3P6S 电池组环境（中文注释版）。

step_output keys（你已验证）包含：
- Pack current [A]             # pack 真实电流（标量）
- Pack terminal voltage [V]    # pack 端电压（标量）
- Cell current [A]             # 单体电流（向量）
- Terminal voltage [V]         # 单体端电压（向量）
- Volume-averaged cell temperature [K] # 单体温度（向量）
等

设计要点：
- reset: rm.solve(..., setup_only=True) 编译并分配 protocol
- step:  写入 protocol[step]=current，然后 rm._step(step) 推进一步
- SOC:   liionpack 不一定直接给 SOC，这里用库仑计数 proxy 更新（用 cell current 更新 cell SOC）
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .aging_scenarios import AgingParams
from .base_env import BasePackEnv
from .observables import build_observation

warnings.filterwarnings(
    "ignore",
    message=r"The 'lumped' thermal option with 'dimensionality' 0 now uses the parameters.*",
)

try:
    import liionpack as lp  # type: ignore
    import pybamm  # type: ignore
except Exception as e:
    raise RuntimeError("当前版本只支持 liionpack 后端，请先安装 liionpack 与 pybamm。") from e


@dataclass
class PackState:
    """内部状态：用于保存 SOC proxy 的连续性。"""
    soc: np.ndarray          # (n_cells,)
    voltage: np.ndarray      # (n_cells,)
    temperature: np.ndarray  # (n_cells,)
    soh: np.ndarray          # (n_cells,)
    i_prev: float
    t: float
    step: int

@dataclass
class HeterogeneityConfig:
    """电芯不一致性注入配置（reset 时生效）"""
    enable: bool = True

    # 初始 SOC 分布
    soc_mean_low: float = 0.20
    soc_mean_high: float = 0.30
    soc_sigma: float = 0.03          # cell SOC 标准差
    soc_clip_low: float = 0.05
    soc_clip_high: float = 0.95

    # 内阻 Ri（lognormal）
    ri_cv: float = 0.10              # 变异系数（std/mean）
    ri_clip_low: float = 0.6         # 限幅（相对倍数）
    ri_clip_high: float = 1.8

    # 换热系数 HTC（lognormal）
    htc_cv: float = 0.20
    htc_clip_low: float = 0.5
    htc_clip_high: float = 2.0

class LiionpackSPMEPackEnv(BasePackEnv):
    """liionpack + pybamm 的电池组环境封装（无 toy 分支）。"""

    def __init__(
        self,
        dt: float,
        max_steps: int,
        v_max: float,
        t_max: float,
        pack_cells_p: int,
        pack_cells_s: int,
        terminate_on_violation: bool = True,

        parameter_set: str = "Chen2020",
        Rb: float = 1.5e-6,
        Rc: float = 1e-2,
        Ri_init: float = 5e-2,
        ocv_init: float = 4.0,
        nproc: int = 1,
        htc: float = 10.0,

        # ===== 新增 =====
        soc_init_low: float = 0.2,
        soc_init_high: float = 0.3,
        soc_sigma_reset: float = 0.01,
    ) -> None:
        super().__init__(dt=dt, max_steps=max_steps, v_max=v_max, t_max=t_max)

        # ---------- pack 配置 ----------
        self.pack_cells_p = int(pack_cells_p)
        self.pack_cells_s = int(pack_cells_s)
        self.terminate_on_violation = bool(terminate_on_violation)

        # ---------- 随机数与老化参数 ----------
        self._rng = np.random.default_rng(0)
        self._aging = AgingParams(1.0, 1.0, 1.0)
        # ---------- reset 单体间的不一致性 ----------
        self._soc_init_low = float(soc_init_low)
        self._soc_init_high = float(soc_init_high)
        self._soc_sigma_reset = float(soc_sigma_reset)
        # ---------- liionpack 内部对象 ----------
        # 创建求解管理器器
        self._rm = lp.CasadiManager()
        self._nproc = int(nproc) # 可用cpu
        # 构建 3P6S 的电路拓扑 netlist
        self._netlist = lp.setup_circuit(
            Np=self.pack_cells_p,
            Ns=self.pack_cells_s,
            Rb=float(Rb),
            Rc=float(Rc),
            Ri=float(Ri_init),
            V=float(ocv_init),
            I=0.0,
        )
        # 加载chen2020参数集
        self._parameter_values = pybamm.ParameterValues(str(parameter_set))
        # 定义步长
        period_s = int(round(self.dt))
        if period_s <= 0:
            raise ValueError("dt 必须 >= 1 秒（liionpack stepping 要求）。")
        # 总时长
        self._experiment = pybamm.Experiment(
            operating_conditions=[f"Rest for {period_s} s"] * int(self.max_steps),
            period=f"{period_s} s",
        )
        # 选择 SPMe + lumped thermal 的仿真函数
        self._sim_func = lp.thermal_simulation

        # 输入：热交换系数（每个 cell 一个）
        self._lp_inputs: Dict[str, Any] = {
            "Total heat transfer coefficient [W.m-2.K-1]": np.ones(self.n_cells) * float(htc)
        }

        # ---------- episode 运行时状态 ----------
        self._lp_step = 0
        self._state: Optional[PackState] = None
        self._last_info: Dict[str, Any] = {}
        self._last_vmax: float = float("-inf")

        # ---------- 缓存参数 ----------
        self._C_cell_Ah: float = 3.0  # 默认兜底
        self._requested_outputs: List[str] = []

    @property
    def n_cells(self) -> int:
        return self.pack_cells_p * self.pack_cells_s

    def set_aging(self, aging: AgingParams) -> None:
        self._aging = aging

    # ============== step_output 解析工具 ==============
    # 将liionpack输出转换为cell向量 ： (T,n_cells)、(T,) --> (n_cells,)  
    def _extract_cell_vector(self, arr: Any, *, name: str) -> np.ndarray:
        """将 step_output 的条目转为 (n_cells,) 向量（取最后时刻）。"""
        x = np.asarray(arr, dtype=float)
        x_last = x[-1] if x.ndim >= 1 else x
        x_last = np.asarray(x_last, dtype=float)

        if x_last.ndim == 0:
            return np.ones(self.n_cells, dtype=float) * float(x_last)

        if x_last.ndim == 1:
            if x_last.size == self.n_cells:
                return x_last
            if x_last.size == 1:
                return np.ones(self.n_cells, dtype=float) * float(x_last[0])
            raise ValueError(f"{name} size={x_last.size}，期望 {self.n_cells} 或 1。")

        if x_last.ndim == 2:
            if x_last.shape == (self.n_cells, 1):
                return x_last[:, 0]
            if x_last.shape == (1, self.n_cells):
                return x_last[0, :]
            raise ValueError(f"{name} shape={x_last.shape} 不符合预期。")

        raise ValueError(f"{name} ndim={x_last.ndim} 不支持。")

    def _extract_scalar_last(self, arr: Any) -> float:
        """将 step_output 条目转为标量（取最后时刻）。"""
        x = np.asarray(arr, dtype=float)
        if x.ndim == 0:
            return float(x)
        return float(x[-1])

    # ============== SOC proxy（库仑计数） ：用“真实 cell current”做库仑计数更新 SOC==============
    def _update_soc_by_cell_currents(
        self,
        soc_prev: np.ndarray,
        i_cells: np.ndarray,
        dt_s: float,
        C_cell_Ah: float,
        *,
        sign_convention: str = "pybamm",  # pybamm: I>0 放电, I<0 充电
    ) -> Tuple[np.ndarray, float]:
        """用单体电流更新单体 SOC，并返回 pack 平均 SOC。"""
        dt_h = float(dt_s) / 3600.0
        C = float(C_cell_Ah)
        if C <= 0:
            raise ValueError(f"C_cell_Ah 必须 >0，当前 {C}。")

        if sign_convention == "pybamm":
            delta = (-i_cells / C) * dt_h
        elif sign_convention == "charge_positive":
            delta = (i_cells / C) * dt_h
        else:
            raise ValueError(f"未知 sign_convention={sign_convention!r}")

        soc = np.clip(soc_prev + delta, 0.0, 1.0)
        return soc, float(np.mean(soc))

    # ============== liionpack 编译（每个 episode 一次） ==============
    def _lp_setup_on_reset(self, initial_soc: float) -> None:
        """每个 episode reset 时执行一次：编译模型并分配 protocol。"""
        # 我需要的输出变量
        self._requested_outputs = [
            # "Current [A]",
            "Terminal voltage [V]",  # ✅ cell 端电压（PyBaMM变量）
            "Volume-averaged cell temperature [K]",  # ✅ cell 温度（PyBaMM变量）
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
            setup_only=True,
        )

        # 存放action：I的 每一步都记录下来
        nsteps = int(self._rm.Nsteps)
        self._rm.protocol = np.zeros(nsteps, dtype=float)

        self._rm.step = -1 #liionpack 内部当前时间索引
        self._lp_step = 0 # 你环境包装的当前 step索引

        # 缓存单体容量（用于 SOC proxy）
        try:
            self._C_cell_Ah = float(self._parameter_values["Nominal cell capacity [A.h]"])
        except Exception:
            self._C_cell_Ah = 3.0
          
    def _lp_do_step(self, current: float) -> Dict[str, Any]:
        """执行 liionpack 一步：写 protocol 并推进 _step。"""
        step = self._lp_step
        self._rm.protocol[step] = float(current)
        self._rm.step = step
        vlims_ok = self._rm._step(step, updated_inputs={})
        self._lp_step += 1
        return {"vlims_ok": bool(vlims_ok), "lp_step": self._lp_step}

    # ============== Gym API ==============
    def reset(self, *, seed: int | None = None, options: Dict | None = None) -> Tuple[np.ndarray, Dict]:
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        soc_mean = float(self._rng.uniform(self._soc_init_low, self._soc_init_high))
        initial_soc = soc_mean
        self._lp_setup_on_reset(initial_soc)

        # 预热一步（I=0），用于填充 step_output 缓冲
        self._rm.protocol[0] = 0.0
        self._rm.step = 0
        _ = self._rm._step(0, updated_inputs={})
        self._lp_step = 0

        so = self._rm.step_output()

        # 从 step_output 读取真实输出
        v_cells = np.asarray(so["Terminal voltage [V]"][-1, :], dtype=float)
        t_cells = np.asarray(so["Volume-averaged cell temperature [K]"][-1, :], dtype=float)
        v_pack = self._extract_scalar_last(so["Pack terminal voltage [V]"])

        # 初始 SOC / SOH
        soc_sigma = self._soc_sigma_reset # 你也可以放到 yaml 配置里
        soc = self._rng.normal(loc=soc_mean, scale=soc_sigma, size=self.n_cells).astype(float)
        soc = np.clip(soc, 0.05, 0.95)
        soh = np.ones(self.n_cells, dtype=float) * self._aging.capacity_scale

        self._state = PackState(soc, v_cells, t_cells, soh, 0.0, 0.0, 0)
        self._last_vmax = float(np.max(v_cells))

        obs = build_observation(soc, v_cells, t_cells, soh, 0.0).as_array()

        info = self._build_info(
            t=0.0,
            current=0.0,
            soc=soc,
            v_cells=v_cells,
            t_cells=t_cells,
            v_pack=v_pack,
            so=so,
            i_cells=None,
            violation=False,
            reason="reset",
            extra={"vlims_ok": True, "lp_step": 0},
        )
        self._last_info = info
        return obs, info

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        if self._state is None:
            raise RuntimeError("必须先 reset() 再 step().")

        # 动作电流裁剪（例如 [-20,0]）
        current = float(np.clip(action[0], self.action_space.low[0], self.action_space.high[0]))
        # 执行一步
        lp_info = self._lp_do_step(current)
        so = self._rm.step_output()

        # 真实输出
        # 真实执行的 pack 电流（环境状态的一部分）
        I_pack_true = self._extract_scalar_last(so["Pack current [A]"])
        v_cells = np.asarray(so["Terminal voltage [V]"][-1, :], dtype=float)
        t_cells = np.asarray(so["Volume-averaged cell temperature [K]"][-1, :], dtype=float)
        v_pack = self._extract_scalar_last(so["Pack terminal voltage [V]"])

        # cell current（用于 SOC proxy）
        i_cells = self._extract_cell_vector(so["Cell current [A]"], name="Cell current [A]")

        # SOC proxy 更新
        soc_prev = self._state.soc
        soc, soc_pack = self._update_soc_by_cell_currents(
            soc_prev=soc_prev,
            i_cells=i_cells,
            dt_s=self.dt,
            C_cell_Ah=self._C_cell_Ah,
            sign_convention="pybamm",
        )

        # 充电 current 为负，SOC 理论上应增加；若减少提示符号问题
        if self._lp_step < 5 and current < -1e-6 and soc_pack < float(np.mean(soc_prev)) - 1e-6:
            print("[WARN] 充电过程中 SOC 下降：请检查 sign_convention（pybamm vs 你的动作符号）。")

        # SOH：episode 常量
        soh = np.ones(self.n_cells, dtype=float) * self._aging.capacity_scale

        
        # 更新内部状态
        self._state = PackState(
            soc=soc,
            voltage=v_cells,
            temperature=t_cells,
            soh=soh,
            i_prev=I_pack_true,
            t=self._lp_step * self.dt,
            step=self._lp_step,
        )

        obs = build_observation(soc, v_cells, t_cells, soh, I_pack_true).as_array()

        # 安全约束与终止
        v_max = float(np.max(v_cells))
        t_max = float(np.max(t_cells))
        # 安全约束 （是否违约）
        violation = (v_max > self.v_max) or (t_max > self.t_max) or (not lp_info["vlims_ok"])

        soc_done = soc_pack >= 0.995
        horizon_done = self._lp_step >= self.max_steps
        # 终止条件  
        terminated = horizon_done or soc_done or (self.terminate_on_violation and violation)
        truncated = False

        if violation:
            reason = "violation"
        elif soc_done:
            reason = "soc_full"
        elif horizon_done:
            reason = "horizon"
        else:
            reason = "running"

        self._last_vmax = v_max

        info = self._build_info(
            t=float(self._lp_step * self.dt),
            current=current,
            soc=soc,
            v_cells=v_cells,
            t_cells=t_cells,
            v_pack=v_pack,
            so=so,
            i_cells=i_cells,
            violation=violation,
            reason=reason,
            extra=lp_info,
        )
        self._last_info = info
        # if self._lp_step < 10:
        #     print(
        #         f"I_set={current:.3f}  I_true={I_pack_true:.3f}  "
        #         f"ΔI={I_pack_true-current:+.3f}  Vmax={v_cells.max():.4f}  SOCmean={soc.mean():.4f}"
        #     )
        # if self._lp_step < 20:
        #     print("Vmin/Vmax:", float(v_cells.min()), float(v_cells.max()))

        return obs, 0.0, terminated, truncated, info

    # ============== info 统一构造 ==============
    def _build_info(
        self,
        *,
        t: float,
        current: float,
        soc: np.ndarray,
        v_cells: np.ndarray,
        t_cells: np.ndarray,
        v_pack: float,
        so: Dict[str, Any],
        i_cells: Optional[np.ndarray],
        violation: bool,
        reason: str,
        extra: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """统一 info 字段（包含真实 pack 电流 & 诊断 cell 电流统计）。"""
        v_max = float(np.max(v_cells))
        v_min = float(np.min(v_cells))
        t_max = float(np.max(t_cells))
        t_min = float(np.min(t_cells))

        # 真实 pack current（liionpack 输出，优先使用它）
        I_pack_true = self._extract_scalar_last(so["Pack current [A]"])

        info: Dict[str, Any] = {
            "t": float(t),

            # 动作 setpoint
            "I": float(current),
            "I_prev": float(current),

            # liionpack 真实执行
            "I_pack_true": float(I_pack_true),
            "V_pack": float(v_pack),

            # SOC 统计
            "SOC_pack": float(np.mean(soc)),
            "std_SOC": float(np.std(soc)),

            # 电压/温度统计
            "V_cell_max": v_max,
            "V_cell_min": v_min,
            "T_cell_max": t_max,
            "T_cell_min": t_min,
            "dV": float(v_max - v_min),
            "dT": float(t_max - t_min),

            # 终止与安全
            "terminated_reason": str(reason),
            "violation": bool(violation),
            "backend": "liionpack",
        }

        # cell current 诊断指标（看并联分流是否均匀、是否存在异常单体）
        if i_cells is not None:
            info.update(
                {
                    "I_cell_mean": float(np.mean(i_cells)),
                    "I_cell_max": float(np.max(i_cells)),
                    "I_cell_min": float(np.min(i_cells)),
                    "I_cell_std": float(np.std(i_cells)),
                }
            )
        else:
            info.update(
                {"I_cell_mean": 0.0, "I_cell_max": 0.0, "I_cell_min": 0.0, "I_cell_std": 0.0}
            )

        if extra:
            info.update(extra)
        return info


def build_pack_env(config: Dict[str, Any]) -> LiionpackSPMEPackEnv:
    """根据 config 构建环境（只支持 liionpack 后端）。"""
    return LiionpackSPMEPackEnv(
        dt=float(config["dt"]),
        max_steps=int(config["max_steps"]),
        v_max=float(config["v_max"]),
        t_max=float(config["t_max"]),
        pack_cells_p=int(config["pack_cells_p"]),
        pack_cells_s=int(config["pack_cells_s"]),
        terminate_on_violation=bool(config.get("terminate_on_violation", True)),
        parameter_set=str(config.get("parameter_set", "Chen2020")),
        Rb=float(config.get("Rb", 1.5e-6)),
        Rc=float(config.get("Rc", 1e-2)),
        Ri_init=float(config.get("Ri_init", 5e-2)),
        ocv_init=float(config.get("ocv_init", 4.0)),
        nproc=int(config.get("nproc", 1)),
        htc=float(config.get("htc", 10.0)),
         # ===== 新增 =====
        soc_init_low=float(config.get("soc_init_low", 0.2)),
        soc_init_high=float(config.get("soc_init_high", 0.3)),
        soc_sigma_reset=float(config.get("soc_sigma_reset", 0.01)),
    )
