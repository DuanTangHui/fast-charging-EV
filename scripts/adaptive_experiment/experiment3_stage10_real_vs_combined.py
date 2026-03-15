"""实验3：第10老化阶段，真实物理仿真训练(630 ep) vs 组合代理方法对比。"""
from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import torch

CURRENT_DIR = Path(__file__).resolve().parent
ROOT_DIR = CURRENT_DIR.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from scripts.static_experiment.common import run_real_training_collect_style
from src.envs.liionpack_spme_pack_env import build_pack_env
from src.evaluation.episode_rollout import rollout_env, rollout_surrogate
from src.rewards.paper_reward import PaperRewardConfig
from src.rl.agent_factory import build_agent_from_config
from src.surrogate.gp_combined import CombinedSurrogate
from src.utils.config import load_config
from src.utils.seeds import set_global_seed


def guarded_policy(agent, low: float, high: float):
    def _policy(state: np.ndarray) -> np.ndarray:
        a = float(agent.act(state)[0])
        return np.array([float(np.clip(a, low, high))], dtype=np.float32)

    return _policy


def infos_to_curve(infos: List[Dict]) -> Dict[str, List[float]]:
    return {
        "time": [float(i.get("t", 0.0)) for i in infos],
        "current": [float(i.get("I_pack_true", i.get("I", 0.0))) for i in infos],
        "voltage": [float(i.get("V_cell_max", 0.0)) for i in infos],
        "soc": [float(i.get("SOC_pack", 0.0)) for i in infos],
        "temperature": [float(i.get("T_cell_max", 0.0)) for i in infos],
    }


def plot_curves(real_curve: Dict[str, List[float]], comb_curve: Dict[str, List[float]], out_png: Path) -> None:
    fig, axs = plt.subplots(2, 2, figsize=(12, 8))

    axs[0, 0].plot(real_curve["time"], real_curve["current"], label="Real-Env Trained")
    axs[0, 0].plot(comb_curve["time"], comb_curve["current"], label="Combined-Surrogate Trained")
    axs[0, 0].set_title("Current")
    axs[0, 0].grid(alpha=0.3)

    axs[0, 1].plot(real_curve["time"], real_curve["voltage"], label="Real-Env Trained")
    axs[0, 1].plot(comb_curve["time"], comb_curve["voltage"], label="Combined-Surrogate Trained")
    axs[0, 1].set_title("Voltage")
    axs[0, 1].grid(alpha=0.3)

    axs[1, 0].plot(real_curve["time"], real_curve["soc"], label="Real-Env Trained")
    axs[1, 0].plot(comb_curve["time"], comb_curve["soc"], label="Combined-Surrogate Trained")
    axs[1, 0].set_title("SOC")
    axs[1, 0].grid(alpha=0.3)

    axs[1, 1].plot(real_curve["time"], real_curve["temperature"], label="Real-Env Trained")
    axs[1, 1].plot(comb_curve["time"], comb_curve["temperature"], label="Combined-Surrogate Trained")
    axs[1, 1].set_title("Temperature")
    axs[1, 1].grid(alpha=0.3)

    for ax in axs.flat:
        ax.set_xlabel("Time [s]")
        ax.legend()

    plt.tight_layout()
    plt.savefig(out_png, dpi=180)
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/pack_3p6s_spme_with_soh_prior.yaml")
    p.add_argument("--runs-dir", default="runs")
    p.add_argument("--stage", type=int, default=10)
    p.add_argument("--real-train-episodes", type=int, default=630)
    p.add_argument("--output-dir", default="runs/adaptive_experiment/result3")
    args = p.parse_args()

    cfg = load_config(args.config).data
    set_global_seed(cfg.get("seed", 42))

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    env = build_pack_env(cfg["env"])
    env.set_aging_stage(args.stage)
    reward_cfg = PaperRewardConfig(**cfg["reward"])

    runs = Path(args.runs_dir)
    stage_agent_ckpt = runs / "adaptive" / f"adaptive_cycle{args.stage}" / "agent_ckpt.pt"
    static_ckpt = runs / "cycle0" / "static_surrogate.pt"
    diff_ckpt = runs / "adaptive" / f"adaptive_cycle{args.stage}" / "diff_surrogate.pt"
    if not stage_agent_ckpt.exists():
        raise FileNotFoundError(f"Missing adaptive stage agent: {stage_agent_ckpt}")
    if not static_ckpt.exists() or not diff_ckpt.exists():
        raise FileNotFoundError("Missing static/diff surrogate ckpt for stage comparison")

    # A) 真实物理环境训练 630 episode
    real_agent = build_agent_from_config(env.observation_space.shape[0], 1, cfg["rl"])
    real_agent.load(str(stage_agent_ckpt), map_location="cpu")

    t0 = time.perf_counter()
    train_metrics, _ = run_real_training_collect_style(
        env,
        real_agent,
        reward_cfg,
        episodes=args.real_train_episodes,
    )
    train_sec = time.perf_counter() - t0
    real_agent_ckpt = out_dir / f"real_env_stage{args.stage}_ep{args.real_train_episodes}_agent_ckpt.pt"
    real_agent.save(str(real_agent_ckpt))

    # B) 组合代理方法（使用已有 adaptive_stage agent）
    comb_agent = build_agent_from_config(env.observation_space.shape[0], 1, cfg["rl"])
    comb_agent.load(str(stage_agent_ckpt), map_location="cpu")
    comb_agent.actor.eval()

    static_surrogate = torch.load(static_ckpt, map_location="cpu", weights_only=False)
    diff_surrogate = torch.load(diff_ckpt, map_location="cpu", weights_only=False)
    combined = CombinedSurrogate(static_surrogate, diff_surrogate)

    low = float(cfg["rl"]["action_low"])
    high = float(cfg["rl"]["action_high"])

    # C) 完整充电模拟与耗时统计
    t1 = time.perf_counter()
    r_real, infos_real = rollout_env(env, guarded_policy(real_agent, low, high), reward_cfg)
    sim_real_sec = time.perf_counter() - t1

    state0, _ = env.reset(seed=777)
    t2 = time.perf_counter()
    r_comb, infos_comb = rollout_surrogate(
        state=state0.copy(),
        surrogate=combined.predict,
        policy=guarded_policy(comb_agent, low, high),
        horizon=int(cfg["env"]["max_steps"]),
        reward_cfg=reward_cfg,
        dt=float(cfg["env"]["dt"]),
        v_max=float(cfg["env"]["v_max"]),
        t_max=float(cfg["env"]["t_max"]),
    )
    sim_comb_sec = time.perf_counter() - t2

    real_curve = infos_to_curve(infos_real)
    comb_curve = infos_to_curve(infos_comb)
    plot_curves(real_curve, comb_curve, out_dir / "exp3_stage10_real_vs_combined_curves.png")

    summary = {
        "stage": args.stage,
        "real_train_episodes": args.real_train_episodes,
        "real_training_walltime_s": train_sec,
        "real_rollout_walltime_s": sim_real_sec,
        "combined_rollout_walltime_s": sim_comb_sec,
        "real_total_reward": float(r_real),
        "combined_total_reward": float(r_comb),
        "real_charge_time_s": float(real_curve["time"][-1] if real_curve["time"] else 0.0),
        "combined_charge_time_s": float(comb_curve["time"][-1] if comb_curve["time"] else 0.0),
    }

    with (out_dir / "exp3_stage10_summary.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(summary.keys()))
        w.writeheader()
        w.writerow(summary)

    with (out_dir / "exp3_real_training_metrics.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["episode", "total_reward", "charge_time_s", "voltage_violation", "temperature_violation", "sim_crash"])
        w.writeheader()
        for m in train_metrics:
            w.writerow({
                "episode": m.episode,
                "total_reward": m.total_reward,
                "charge_time_s": m.charge_time_s,
                "voltage_violation": m.voltage_violation,
                "temperature_violation": m.temperature_violation,
                "sim_crash": m.sim_crash,
            })

    print(f"[DONE] Saved outputs to {out_dir}")


if __name__ == "__main__":
    main()