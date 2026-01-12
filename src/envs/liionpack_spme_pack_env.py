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
            # build netlist: lp.setup_circuit uses Np, Ns, Rb, Rc, Ri, V, I ... :contentReference[oaicite:3]{index=3}
            self._netlist = lp.setup_circuit(
                Np=pack_cells_p,
                Ns=pack_cells_s,
                Rb=Rb,
                Rc=Rc,
                Ri=Ri_init,
                V=ocv_init,
                I=0.0,  # initial current guess, will be overwritten per step
            )
            self._parameter_values = pybamm.ParameterValues(parameter_set)

            # A minimal experiment ONLY used to define timestep length for manager setup.
            # liionpack uses experiment period as global timestep in stepping solver. :contentReference[oaicite:4]{index=4}
            period_s = int(round(self.dt))
            if period_s <= 0:
                raise ValueError("dt must be >= 1 second for liionpack stepping.")
            self._experiment = pybamm.Experiment(
                operating_conditions=[f"Rest for {period_s} s"] * max_steps,
                period=f"{period_s} s",
            )

            # Choose a sim_func. thermal_simulation is an SPMe + lumped thermal example in docs. :contentReference[oaicite:5]{index=5}
            self._sim_func = lp.thermal_simulation

            # Create manager
            self._rm = lp.CasadiManager()
            self._nproc = int(nproc)

    @property
    def n_cells(self) -> int:
        return self.pack_cells_p * self.pack_cells_s

    def set_aging(self, aging: AgingParams) -> None:
        """Update aging parameters for subsequent resets."""
        self._aging = aging
     # ------------------------ liionpack helpers ------------------------
    def _lp_setup_on_reset(self, initial_soc: float) -> None:
        """
        Call manager.solve(..., setup_only=True) once per episode.
        """
        assert self._rm is not None
        self._requested_outputs = [
            "Terminal voltage [V]",
            "Volume-averaged cell temperature [K]",
        ]
        self._rm.solve(
            netlist=self._netlist,
            sim_func=self._sim_func,
            parameter_values=self._parameter_values,
            experiment=self._experiment,
            output_variables=self._requested_outputs,
            inputs=self._lp_inputs,
            nproc=self._nproc,
            initial_soc=float(initial_soc),
            setup_only=True,  # important: only build internal structures once :contentReference[oaicite:6]{index=6}
        )

        # protocol buffer for ALL steps; we will overwrite protocol[k] each RL step
        # manager._step expects a flattened_protocol list. :contentReference[oaicite:7]{index=7}
        nsteps = int(self._rm.Nsteps)
        self._rm.protocol = np.zeros(nsteps, dtype=float)
        self._rm.step = -1
        self._lp_step = 0
        

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
        current = float(np.clip(action[0], self.action_space.low[0], self.action_space.high[0]))
        if self._backend == "liionpack":
           

            # 1) step liionpack
            lp_info = self._lp_do_step(current)

            # 2) TODO: extract soc/voltage/temp from manager into vectors
            #    For now keep placeholders so code runs end-to-end.
            #    You should implement _extract_* once you inspect rm outputs.
            # 2) Extract real outputs from liionpack (USER outputs live in step_output, not rm.output)
            so = self._rm.step_output()  # provided by your version (you saw it in dir())
           
            v_pack = float(np.asarray(so["Pack terminal voltage [V]"])[-1])

            v_cells = np.asarray(so["Terminal voltage [V]"][-1, :], dtype=float)
            t_cells = np.asarray(so["Volume-averaged cell temperature [K]"][-1, :], dtype=float)

            dV_cells = float(v_cells.max() - v_cells.min())
            dT_cells = float(t_cells.max() - t_cells.min())
            # SOC: your liionpack outputs may not expose SOC directly.
            # For now, keep a simple coulomb-counting proxy so obs has a meaningful SOC signal.
            if self._state is None:
                soc = np.ones(self.n_cells) * 0.25
            else:
                soc = self._state.soc

            dt_hours = self.dt / 3600.0
            capacity_Ah = 3.0 * self._aging.capacity_scale   # keep consistent with your toy setting for now
            soc = np.clip(soc + (-current / capacity_Ah) * dt_hours, 0.0, 1.0)
            soc_pack = float(np.mean(soc))
            soc_done = soc_pack >= 0.999  # 或 0.995

            soh = np.ones(self.n_cells) * self._aging.capacity_scale

            # keep state for SOC proxy continuity
            self._state = PackState(soc, v_cells, t_cells, soh, current, self._lp_step * self.dt, self._lp_step)

            obs = build_observation(soc, v_cells, t_cells, soh, current).as_array()

            v_max = float(np.max(v_cells))
            v_min = float(np.min(v_cells))
            t_max = float(np.max(t_cells))
            t_min = float(np.min(t_cells))
            violation = (v_max > self.v_max) or (t_max > self.t_max) or (not lp_info["vlims_ok"])
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
                "t": float(self._lp_step * self.dt),
                "I": current,
                "SOC_pack": float(np.mean(soc)),
                "V_cell_max": v_max,
                "V_cell_min": v_min,
                "T_cell_max": t_max,
                "T_cell_min": t_min,
                "V_pack": v_pack,
                "dV": dV_cells,     # trainer 需要这个键
                "dT": dT_cells,
                "terminated_reason": reason,
                "violation": violation,
                "backend": "liionpack",
                **lp_info,
            }
            if self._lp_step < 10:
                print("raw_action=", action, " clipped_current=", current, " bounds=", self.action_space.low[0], self.action_space.high[0])

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
