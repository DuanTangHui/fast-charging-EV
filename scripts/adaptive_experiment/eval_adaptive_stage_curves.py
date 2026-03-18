"""Evaluate adaptive stage agents and export full-charge curves to CSV.

This script scans `runs/adaptive/adaptive_cycle*/agent_ckpt.pt`, and for each stage-N
agent runs one full charging episode in aging stage N, following the same episode stop
logic as `scripts/test_real_policy.py` (loop until terminated or truncated).
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path

import numpy as np
import torch

CURRENT_DIR = Path(__file__).resolve().parent
ROOT_DIR = CURRENT_DIR.parent.parent
sys.path.append(str(ROOT_DIR))

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
STAGE_PATTERN = re.compile(r"adaptive_cycle(\d+)$")


def guard_action(action: float, info: dict, low: float, high: float) -> float:
    """Safety guard copied from scripts/test_real_policy.py."""
    v = float(info.get("V_cell_max", -1e9))
    t = float(info.get("T_cell_max", -1e9))
    viol = bool(info.get("violation", False))

    v_soft = 4.17
    t_soft = 318.15 - 1.5

    if viol:
        return float(np.clip(action + 5.0, low, high))

    if v >= v_soft or t >= t_soft:
        return float(np.clip(action + 2.0, low, high))

    return float(np.clip(action, low, high))


def _round_seq(values: list[float], digits: int = 6) -> list[float]:
    return [round(float(v), digits) for v in values]


def discover_stage_agents(adaptive_dir: Path, expected_stages: int) -> list[tuple[int, Path]]:
    pairs: list[tuple[int, Path]] = []
    for d in sorted(adaptive_dir.glob("adaptive_cycle*")):
        if not d.is_dir():
            continue
        m = STAGE_PATTERN.search(d.name)
        if not m:
            continue
        stage = int(m.group(1))
        ckpt = d / "agent_ckpt.pt"
        if ckpt.exists():
            pairs.append((stage, ckpt))

    if len(pairs) < expected_stages:
        print(
            f"[WARN] Found {len(pairs)} stage checkpoints under {adaptive_dir}, "
            f"expected up to {expected_stages}. Missing stages will be skipped.",
        )

    return sorted(pairs, key=lambda x: x[0])


def rollout_one_stage(agent: object, env: object, low: float, high: float, use_guard: bool) -> dict:
    state, info = env.reset()
    done = False

    series = {
        "total_current": [],
        "total_voltage": [],
        "total_soc": [],
        "voltage_diff": [],
        "temperature_diff": [],
    }

    while not done:
        action = agent.act(state)
        raw_action = float(action[0])
        final_action = guard_action(raw_action, info, low, high) if use_guard else float(np.clip(raw_action, low, high))

        next_state, _, terminated, truncated, next_info = env.step(np.array([final_action], dtype=np.float32))

        series["total_current"].append(next_info.get("I_pack_true", final_action))
        series["total_voltage"].append(next_info.get("V_pack", 0.0))
        series["total_soc"].append(next_info.get("SOC_pack", 0.0))
        series["voltage_diff"].append(next_info.get("dV", 0.0))
        series["temperature_diff"].append(next_info.get("dT", 0.0))

        state = next_state
        info = next_info
        done = bool(terminated or truncated)

    return {k: _round_seq(v, digits=6) for k, v in series.items()}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/pack_3p6s_spme_with_soh_prior.yaml")
    parser.add_argument("--adaptive-dir", default="runs/adaptive")
    parser.add_argument("--output", default="scripts/adaptive_experiment/adaptive_stage_charge_curves.csv")
    parser.add_argument("--start-stage", type=int, default=1)
    parser.add_argument("--end-stage", type=int, default=4)
    parser.add_argument("--no-guard", action="store_true")
    args = parser.parse_args()

    from src.envs.liionpack_spme_pack_env import build_pack_env
    from src.rl.agent_factory import build_agent_from_config
    from src.utils.config import load_config
    from src.utils.seeds import set_global_seed

    config = load_config(args.config).data
    set_global_seed(config.get("seed", 42))

    env = build_pack_env(config["env"])
    low = float(config["rl"]["action_low"])
    high = float(config["rl"]["action_high"])

    adaptive_dir = Path(args.adaptive_dir)
    stage_agents = discover_stage_agents(adaptive_dir, expected_stages=args.end_stage - args.start_stage + 1)
    stage_agents = [(s, p) for (s, p) in stage_agents if args.start_stage <= s <= args.end_stage]

    if not stage_agents:
        raise FileNotFoundError(f"No checkpoints found in stage range [{args.start_stage}, {args.end_stage}] under {adaptive_dir}")

    rows: list[dict[str, str]] = []
    for stage, ckpt in stage_agents:
        print(f"[INFO] Evaluating stage={stage}, ckpt={ckpt}")
        if hasattr(env, "set_aging_stage"):
            env.set_aging_stage(stage)

        agent = build_agent_from_config(
            state_dim=env.observation_space.shape[0],
            action_dim=1,
            rl_config=config["rl"],
        )
        agent.load(str(ckpt), map_location=str(DEVICE))
        agent.actor.eval()

        try:
            curves = rollout_one_stage(
                agent,
                env,
                low=low,
                high=high,
                use_guard=not args.no_guard,
                max_steps=args.max_steps,
            )
        except Exception as exc:
            if not args.continue_on_error:
                raise
            print(f"[ERROR] stage={stage} failed: {exc}. Skipping due to --continue-on-error.")
            continue
        rows.append(
            {
                "stage": str(stage),
                "total_current": json.dumps(curves["total_current"], ensure_ascii=False),
                "total_voltage": json.dumps(curves["total_voltage"], ensure_ascii=False),
                "total_soc": json.dumps(curves["total_soc"], ensure_ascii=False),
                "voltage_diff": json.dumps(curves["voltage_diff"], ensure_ascii=False),
                "temperature_diff": json.dumps(curves["temperature_diff"], ensure_ascii=False),
            }
        )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["stage", "total_current", "total_voltage", "total_soc", "voltage_diff", "temperature_diff"]
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"[DONE] Saved {len(rows)} stages to {output_path}")


if __name__ == "__main__":
    main()