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
    axes[1].plot(curve["t"], curve["V_cell_max"], label="Vmax")
    axes[1].plot(curve["t"], curve["V_cell_min"], label="Vmin")
    axes[1].set_title("Voltage")
    axes[2].plot(curve["t"], curve["T_cell_max"], label="Tmax")
    axes[2].plot(curve["t"], curve["T_cell_min"], label="Tmin")
    axes[2].set_title("Temperature")
    axes[3].plot(curve["t"], curve["I"], label="Current")
    axes[3].set_title("Current")
    axes[4].plot(curve["t"], curve["dV"], label="dV")
    axes[4].plot(curve["t"], curve["dT"], label="dT")
    axes[4].set_title("Cell Spread")
    axes[5].plot(curve["t"], curve["reward"], label="Reward")
    axes[5].set_title("Reward")

    for ax in axes:
        ax.grid(True)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
