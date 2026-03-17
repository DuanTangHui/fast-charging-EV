"""实验2：25°C下，静态策略 vs 组合策略在老化1~50阶段的真实环境性能对比。"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np

CURRENT_DIR = Path(__file__).resolve().parent
ROOT_DIR = CURRENT_DIR.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from src.envs.liionpack_spme_pack_env import build_pack_env
from src.evaluation.episode_rollout import rollout_env
from src.rewards.paper_reward import PaperRewardConfig
from src.rl.agent_factory import build_agent_from_config
from src.utils.config import load_config
from src.utils.seeds import set_global_seed


def make_policy(agent, low: float, high: float):
    def _policy(state: np.ndarray) -> np.ndarray:
        action = float(agent.act(state)[0])
        action = float(np.clip(action, low, high))
        return np.array([action], dtype=np.float32)

    return _policy


def episode_stats(infos: List[Dict], v_max: float, t_max: float) -> Dict[str, float]:
    t_end = float(infos[-1].get("t", 0.0)) if infos else 0.0
    vmax = max(float(i.get("V_cell_max", v_max)) for i in infos) if infos else v_max
    tcell = max(float(i.get("T_cell_max", t_max)) for i in infos) if infos else t_max
    return {
        "charge_time_s": t_end,
        "voltage_violation": float(max(0.0, vmax - v_max)),
        "temperature_violation": float(max(0.0, tcell - t_max)),
    }


def plot_curves(rows: List[Dict[str, float]], out_dir: Path) -> None:
    x = np.array([int(r["stage"]) for r in rows])
    fig, axs = plt.subplots(2, 2, figsize=(12, 8), sharex=True)

    def y(key: str) -> np.ndarray:
        return np.array([r[key] for r in rows])

    axs[0, 0].plot(x, y("static_total_reward"), label="Static", marker="o", ms=3)
    axs[0, 0].plot(x, y("combined_total_reward"), label="Combined", marker="s", ms=3)
    axs[0, 0].set_ylabel("Cumulative Reward")
    axs[0, 0].grid(alpha=0.3)
    axs[0, 0].legend()

    axs[0, 1].plot(x, y("static_charge_time_s"), label="Static", marker="o", ms=3)
    axs[0, 1].plot(x, y("combined_charge_time_s"), label="Combined", marker="s", ms=3)
    axs[0, 1].set_ylabel("Charge Time [s]")
    axs[0, 1].grid(alpha=0.3)

    axs[1, 0].plot(x, y("static_voltage_violation"), label="Static", marker="o", ms=3)
    axs[1, 0].plot(x, y("combined_voltage_violation"), label="Combined", marker="s", ms=3)
    axs[1, 0].set_ylabel("Voltage Violation [V]")
    axs[1, 0].set_xlabel("Aging Stage")
    axs[1, 0].grid(alpha=0.3)

    axs[1, 1].plot(x, y("static_temperature_violation"), label="Static", marker="o", ms=3)
    axs[1, 1].plot(x, y("combined_temperature_violation"), label="Combined", marker="s", ms=3)
    axs[1, 1].set_ylabel("Temperature Violation [K]")
    axs[1, 1].set_xlabel("Aging Stage")
    axs[1, 1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_dir / "exp2_stage1_50_comparison.png", dpi=180)
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/pack_3p6s_spme_with_soh_prior.yaml")
    p.add_argument("--runs-dir", default="runs")
    p.add_argument("--start-stage", type=int, default=1)
    p.add_argument("--end-stage", type=int, default=50)
    p.add_argument("--seed", type=int, default=123)
    p.add_argument("--output-dir", default="runs/adaptive_experiment/result2")
    args = p.parse_args()

    cfg = load_config(args.config).data
    set_global_seed(cfg.get("seed", 42))

    env = build_pack_env(cfg["env"])
    reward_cfg = PaperRewardConfig(**cfg["reward"])

    runs_dir = Path(args.runs_dir)
    static_ckpt = runs_dir / "cycle0" / "agent_ckpt.pt"
    if not static_ckpt.exists():
        raise FileNotFoundError(f"Missing static surrogate: {static_ckpt}")
    
    static_agent = build_agent_from_config(env.observation_space.shape[0], 1, cfg["rl"])
    static_agent.load(str(static_ckpt), map_location="cpu")
    static_agent.actor.eval()
    static_policy = make_policy(static_agent, float(cfg["rl"]["action_low"]), float(cfg["rl"]["action_high"]))

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows: List[Dict[str, float]] = []
    for stage in range(args.start_stage, args.end_stage + 1):
        agent_ckpt = runs_dir / "adaptive" / f"adaptive_cycle{stage}" / "agent_ckpt.pt"
        if not agent_ckpt.exists():
            print(f"[WARN] stage={stage} skipped (missing ckpt)")
            continue

        if hasattr(env, "set_aging_stage"):
            env.set_aging_stage(stage)

        agent = build_agent_from_config(env.observation_space.shape[0], 1, cfg["rl"])
        agent.load(str(agent_ckpt), map_location="cpu")
        agent.actor.eval()

        combined_policy = make_policy(agent, float(cfg["rl"]["action_low"]), float(cfg["rl"]["action_high"]))

        # 在真实物理仿真环境中评估两个策略（避免在各自 surrogate 上各测各的偏差）
        stage_seed = args.seed + stage
        set_global_seed(stage_seed)
        r_static, infos_static = rollout_env(env, static_policy, reward_cfg, reset_seed=stage_seed)
        set_global_seed(stage_seed)
        r_comb, infos_comb = rollout_env(env, combined_policy, reward_cfg, reset_seed=stage_seed)

        s1 = episode_stats(infos_static, env.v_max, env.t_max)
        s2 = episode_stats(infos_comb, env.v_max, env.t_max)
        row = {
            "stage": float(stage),
            "static_total_reward": float(r_static),
            "combined_total_reward": float(r_comb),
            "static_charge_time_s": s1["charge_time_s"],
            "combined_charge_time_s": s2["charge_time_s"],
            "static_voltage_violation": s1["voltage_violation"],
            "combined_voltage_violation": s2["voltage_violation"],
            "static_temperature_violation": s1["temperature_violation"],
            "combined_temperature_violation": s2["temperature_violation"],
        }
        rows.append(row)
        print(f"[EXP2] stage={stage}, R_static={r_static:.2f}, R_comb={r_comb:.2f}")

    if not rows:
        raise RuntimeError("No stages evaluated. Check runs directory.")

    csv_path = out_dir / "exp2_stage1_50_metrics.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    plot_curves(rows, out_dir)
    print(f"[DONE] Saved: {csv_path}")


if __name__ == "__main__":
    main()