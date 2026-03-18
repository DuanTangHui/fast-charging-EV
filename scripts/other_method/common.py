from __future__ import annotations

import csv
from pathlib import Path
from typing import Callable, Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np


def clip_action(value: float, low: float, high: float) -> np.ndarray:
    return np.array([float(np.clip(value, low, high))], dtype=np.float32)


def stage_policy_from_schedule(
    schedule: List[Dict[str, float]],
    action_low: float,
    action_high: float,
) -> Callable[[np.ndarray, Dict], np.ndarray]:
    """Build a piecewise-constant current policy from SOC breakpoints.

    schedule example:
    [
      {"soc_upper": 0.4, "current": -20.0},
      {"soc_upper": 0.7, "current": -12.0},
      {"soc_upper": 0.9, "current": -7.0},
      {"soc_upper": 1.1, "current": -3.0},
    ]
    """

    sorted_schedule = sorted(schedule, key=lambda x: float(x["soc_upper"]))

    def _policy(_obs: np.ndarray, info: Dict) -> np.ndarray:
        soc = float(info.get("SOC_pack", 0.0))
        for item in sorted_schedule:
            if soc <= float(item["soc_upper"]):
                return clip_action(float(item["current"]), action_low, action_high)
        return clip_action(float(sorted_schedule[-1]["current"]), action_low, action_high)

    return _policy


def rollout_one_episode(
    env,
    policy: Callable[[np.ndarray, Dict], np.ndarray],
    seed: int,
) -> Tuple[List[Dict], Dict[str, float]]:
    obs, info = env.reset(seed=seed)
    traj: List[Dict] = []

    while True:
        action = policy(obs, info)
        action_value = float(action[0])
        obs, _reward, terminated, truncated, info = env.step(action)

        traj.append(
            {
                "t": float(info.get("t", 0.0)),
                "current": action_value,
                "soc": float(info.get("SOC_pack", np.nan)),
                "vmax": float(info.get("V_cell_max", np.nan)),
                "tmax": float(info.get("T_cell_max", np.nan)),
                "terminated_reason": str(info.get("terminated_reason", "running")),
                "violation": bool(info.get("violation", False)),
            }
        )
        if terminated or truncated:
            break

    summary = {
        "steps": float(len(traj)),
        "soc_end": float(traj[-1]["soc"]),
        "current_mean": float(np.mean([x["current"] for x in traj])),
        "vmax_peak": float(np.max([x["vmax"] for x in traj])),
        "tmax_peak": float(np.max([x["tmax"] for x in traj])),
        "violations": float(np.sum([1.0 if x["violation"] else 0.0 for x in traj])),
    }
    return traj, summary


def save_traj_csv(path: Path, traj: List[Dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["t", "current", "soc", "vmax", "tmax", "terminated_reason", "violation"],
        )
        writer.writeheader()
        writer.writerows(traj)


def plot_comparison(method_to_traj: Dict[str, List[Dict]], out_png: Path) -> None:
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(4, 1, figsize=(11, 13), sharex=True)
    metric_keys = ["current", "soc", "vmax", "tmax"]
    ylabels = ["Current (A)", "SOC", "Max Cell Voltage (V)", "Max Cell Temperature (K)"]

    for name, traj in method_to_traj.items():
        t = [x["t"] for x in traj]
        for i, key in enumerate(metric_keys):
            axes[i].plot(t, [x[key] for x in traj], label=name)
            axes[i].set_ylabel(ylabels[i])
            axes[i].grid(alpha=0.3)

    axes[-1].set_xlabel("Time (s)")
    axes[0].legend(loc="best")
    plt.tight_layout()
    plt.savefig(out_png, dpi=180)
    plt.close(fig)