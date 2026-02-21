"""Plotting helpers for episode trajectories."""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt


def plot_episode(curve: Dict[str, List[float]], path: str | Path) -> None:
    """Plot SOC/V/T/I and delta curves to a file.

    Args:
        curve: Dict containing time-series arrays.
        path: Output image path.
    """

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(3, 2, figsize=(10, 8))
    axes = axes.flatten()

    axes[0].plot(curve["t"], curve["SOC_pack"], label="SOC")
    axes[0].set_title("SOC")
    axes[0].legend()

    axes[1].plot(curve["t"], curve["V_cell_max"], label="Vmax")
    axes[1].plot(curve["t"], curve["V_cell_min"], label="Vmin")
    axes[1].set_title("Voltage")
    axes[1].legend()

    axes[2].plot(curve["t"], curve["T_cell_max"], label="Tmax")
    axes[2].plot(curve["t"], curve["T_cell_min"], label="Tmin")
    axes[2].set_title("Temperature")
    axes[2].legend()

    axes[3].plot(curve["t"], curve["I"], label="Current")
    axes[3].set_title("Current")
    axes[3].legend()

    axes[4].plot(curve["t"], curve["dV"], label="dV")
    axes[4].plot(curve["t"], curve["dT"], label="dT")
    axes[4].set_title("Cell Spread")
    axes[4].legend()

    axes[5].plot(curve["t"], curve["reward"], label="Reward")
    axes[5].set_title("Reward")
    axes[5].legend()

    for ax in axes:
        ax.grid(True)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)

    # ---- 2) 奖励分项 ----
    # fig, ax = plt.subplots(1, 1, figsize=(10, 4))
    # for k in ["r_soc","r_v","r_action","r_time","r_const","r_track","r_finish"]:
    #     if k in curve:
    #         ax.plot(curve["t"], curve[k], label=k)
    # ax.set_title("Reward terms")
    # ax.grid(True)
    # ax.legend()
    # fig.tight_layout()
    # fig.savefig(path.with_suffix(".terms.png"))
    # plt.close(fig)