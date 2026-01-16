"""Trainer implementing Algorithm 2 with adaptive differential surrogate."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import numpy as np

from ...calibration.fast_calibrator import fast_calibrate
from ...envs.aging_scenarios import compute_aging_params
from ...envs.base_env import BasePackEnv
from ...evaluation.episode_rollout import rollout_env, rollout_surrogate
from ...evaluation.reports import summarize_episode
from ...rewards.paper_reward import PaperRewardConfig
from ...soh_prior.feature_extraction import extract_features
from ...soh_prior.soh2param_mapper import SOHToParamMapper
from ...soh_prior.stapinn_predictor import DummyPredictor
from ...surrogate.dataset import build_dataset
from ...surrogate.gp_combined import CombinedSurrogate
from ...surrogate.gp_differential import DifferentialSurrogate
from ...surrogate.gp_static import StaticSurrogate
from ...utils.logging import log_metrics
from ...envs.observables import curve_from_infos
from ...utils.plotting import plot_episode


@dataclass
class AdaptiveConfig:
    """Configuration for adaptive training cycles."""

    cycles: int
    real_episodes_per_cycle: int
    surrogate_epochs: int
    policy_epochs: int


def train_adaptive_cycles(
    env: BasePackEnv,
    static_surrogate: StaticSurrogate,
    diff_surrogate: DifferentialSurrogate,
    combined: CombinedSurrogate,
    reward_cfg: PaperRewardConfig,
    config: AdaptiveConfig,
    run_dir: str,
    soh_enabled: bool,
    lambda_prior: float,
    theta_dim: int,
    dummy_soh: float,
) -> List[Dict[str, float]]:
    """Run adaptive cycles with differential updates."""

    predictor = DummyPredictor(fixed_soh=dummy_soh)
    mapper = SOHToParamMapper(theta_dim=theta_dim)
    all_metrics: List[Dict[str, float]] = []

    for cycle in range(1, config.cycles + 1):
        segments = []
        transitions: List[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
        for _ in range(config.real_episodes_per_cycle):
            def guarded_random(state: np.ndarray) -> np.ndarray:
                # 默认随机
                a = np.random.uniform(-20.0, 0.0, size=(1,)).astype(np.float32)

                # 从 env 里拿上一步 info
                env = guarded_random.env_ref
                last = getattr(env, "_last_info", None)
               
                if last is None:
                    return a

                v = float(last.get("V_cell_max", 0.0))

                # 电压守门（给 dt=30s 留缓冲）
                if v > 4.195:
                    return np.array([0.0], dtype=np.float32)
                if v > 4.18:
                    return np.array([-2.0], dtype=np.float32)

                return a
            guarded_random.env_ref = env
            total_reward, infos = rollout_env(env, guarded_random, reward_cfg)
            segments.append(
                {
                    "soc": np.array([info["SOC_pack"] for info in infos]),
                    "voltage": np.array([info["V_cell_max"] for info in infos]),
                    "temperature": np.array([info["T_cell_max"] for info in infos]),
                    "time": np.array([info["t"] for info in infos]),
                }
            )
            for i in range(len(infos) - 1):
                s = np.array(
                    [
                        infos[i]["SOC_pack"],
                        infos[i]["V_cell_max"],
                        infos[i]["V_cell_min"],
                        infos[i]["dV"],
                        infos[i]["T_cell_max"],
                        infos[i]["T_cell_min"],
                        infos[i]["dT"],
                        0.0,
                        0.0,
                        0.0,
                        1.0,
                        infos[i]["I"],
                    ],
                    dtype=np.float32,
                )
                s_next = np.array(
                    [
                        infos[i + 1]["SOC_pack"],
                        infos[i + 1]["V_cell_max"],
                        infos[i + 1]["V_cell_min"],
                        infos[i + 1]["dV"],
                        infos[i + 1]["T_cell_max"],
                        infos[i + 1]["T_cell_min"],
                        infos[i + 1]["dT"],
                        0.0,
                        0.0,
                        0.0,
                        1.0,
                        infos[i + 1]["I"],
                    ],
                    dtype=np.float32,
                )
                transitions.append((s, np.array([infos[i]["I"]], dtype=np.float32), s_next - s))
        
      
        states = np.stack([t[0] for t in transitions])
        deltas = np.stack([t[2] for t in transitions])

        def report(name, x):
            print(name, "shape", x.shape,
                "min", np.nanmin(x), "max", np.nanmax(x),
                "nan", np.isnan(x).any(), "inf", np.isinf(x).any())

        report("states", states)
        report("deltas", deltas)

        dataset = build_dataset(transitions)
        diff_surrogate.fit(dataset, epochs=config.surrogate_epochs)

        if soh_enabled:
            features = extract_features(segments)
            soh_pred = predictor.predict(features)
            theta_prior = mapper.map(soh_pred)
            simulator = lambda theta: theta  # placeholder for parameterized simulation
            calib = fast_calibrate(theta_prior, simulator, theta_prior, lambda_prior, steps=10, lr=0.05)
            aging = compute_aging_params(cycle, theta_hat=calib.theta_hat.tolist())
            env.set_aging(aging)

        for epoch in range(config.policy_epochs):
            def policy(state: np.ndarray) -> np.ndarray:
                return env.action_space.sample()

            state, _ = env.reset()
            total_reward, infos = rollout_surrogate(
                state=state,
                surrogate=combined.predict,
                policy=policy,
                horizon=env.max_steps,
                reward_cfg=reward_cfg,
                dt=env.dt,
                v_max=env.v_max,
                t_max=env.t_max,
            )
            
            metrics = summarize_episode(infos)
            metrics.update({"epoch": epoch, "cycle": cycle, "phase": "adaptive", "reward": total_reward})
            log_metrics(f"{run_dir}/metrics.jsonl", metrics)
            curve = curve_from_infos(infos)
            plot_episode(curve, f"{run_dir}/cycle_{cycle}_epoch_{epoch}.png")
            all_metrics.append(metrics)

    return all_metrics
