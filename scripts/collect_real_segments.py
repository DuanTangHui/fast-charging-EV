"""Collect real environment segments for SOH calibration."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from src.envs.liionpack_spme_pack_env import build_pack_env
from src.utils.config import load_config
from src.utils.logging import ensure_dir


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--episodes", type=int, default=2)
    args = parser.parse_args()

    config = load_config(args.config).data
    env_cfg = dict(config["env"])
    env_cfg["backend"] = config.get("backend", env_cfg.get("backend", "toy"))
    env = build_pack_env(env_cfg)

    segments = []
    for _ in range(args.episodes):
        state, info = env.reset()
        done = False
        socs = []
        volts = []
        temps = []
        times = []
        while not done:
            action = env.action_space.sample()
            next_state, _, terminated, truncated, next_info = env.step(action)
            socs.append(next_info["SOC_pack"])
            volts.append(next_info["V_cell_max"])
            temps.append(next_info["T_cell_max"])
            times.append(next_info["t"])
            done = terminated or truncated
        segments.append({"soc": socs, "voltage": volts, "temperature": temps, "time": times})

    out_dir = ensure_dir(Path(config["logging"]["runs_dir"]) / "segments")
    np.savez(out_dir / "segments.npz", segments=segments)


if __name__ == "__main__":
    main()
