from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from src.utils.config import load_config
from src.utils.seeds import set_global_seed
from scripts.other_method.common import rollout_one_episode, stage_policy_from_schedule


def _build_pack_env(env_cfg: Dict[str, Any]):
    # 延迟导入：允许在未安装 liionpack 的环境下也能执行 --help。
    from src.envs.liionpack_spme_pack_env import build_pack_env

    return build_pack_env(env_cfg)


def evaluate_stage_schedule(env, schedule: List[Dict[str, float]], seed: int) -> float:
    low = float(env.action_space.low[0])
    high = float(env.action_space.high[0])
    policy = stage_policy_from_schedule(schedule, action_low=low, action_high=high)
    traj, summary = rollout_one_episode(env, policy, seed=seed)
    penalty = 0.0
    for row in traj:
        penalty += 80.0 * max(0.0, row["vmax"] - float(env.v_max))
        penalty += 2.0 * max(0.0, row["tmax"] - float(env.t_max))
    return summary["soc_end"] - 0.002 * summary["steps"] - penalty


def run_simple_ga(env, seed: int, population: int, generations: int) -> List[Dict[str, float]]:
    rng = np.random.default_rng(seed)

    soc_breaks = [0.35, 0.65, 0.85, 1.10]
    low = float(env.action_space.low[0])
    high = float(env.action_space.high[0])

    def random_schedule() -> List[Dict[str, float]]:
        currents = sorted(rng.uniform(low, -1.0, size=4))
        return [
            {"soc_upper": soc_breaks[i], "current": float(currents[i])}
            for i in range(4)
        ]

    pop = [random_schedule() for _ in range(population)]
    best = pop[0]
    best_score = -1e9

    for _ in range(generations):
        scored: List[Tuple[float, List[Dict[str, float]]]] = []
        for ind in pop:
            score = evaluate_stage_schedule(env, ind, seed=seed)
            scored.append((score, ind))
        scored.sort(key=lambda x: x[0], reverse=True)

        if scored[0][0] > best_score:
            best_score = scored[0][0]
            best = scored[0][1]

        elites = [ind for _, ind in scored[: max(2, population // 4)]]
        next_pop = elites.copy()
        while len(next_pop) < population:
            p1, p2 = rng.choice(elites, size=2, replace=True)
            child = []
            for i in range(4):
                base = p1[i]["current"] if rng.random() < 0.5 else p2[i]["current"]
                mutated = float(np.clip(base + rng.normal(0.0, 1.5), low, high))
                child.append({"soc_upper": soc_breaks[i], "current": mutated})
            child = sorted(child, key=lambda x: x["soc_upper"])
            next_pop.append(child)
        pop = next_pop

    return best


def train_rl_method(config_data: Dict[str, Any], algo: str, output_dir: Path) -> Path:
    from src.rewards.paper_reward import PaperRewardConfig
    from src.rl.agent_factory import build_agent_from_config
    from src.rl.trainers.trainer_static_gp import Cycle0Config, train_cycle0
    from src.surrogate.gp_static import StaticSurrogate

    cfg = json.loads(json.dumps(config_data))
    cfg["rl"]["algorithm"] = algo

    env = _build_pack_env(cfg["env"])
    reward_cfg = PaperRewardConfig(**cfg["reward"])
    agent = build_agent_from_config(
        state_dim=env.observation_space.shape[0],
        action_dim=1,
        rl_config=cfg["rl"],
    )
    surrogate = StaticSurrogate(
        input_dim=env.observation_space.shape[0] + 1,
        output_dim=env.observation_space.shape[0] - 1,
        hidden_sizes=cfg["surrogate"]["hidden_sizes"],
        ensemble_size=cfg["surrogate"]["ensemble_size"],
        lr=cfg["surrogate"]["learning_rate"],
    )
    cycle0_cfg = Cycle0Config(**cfg["trainer"]["cycle0"])
    run_dir = output_dir / f"{algo}_static_gp"
    run_dir.mkdir(parents=True, exist_ok=True)
    train_cycle0(env, agent, reward_cfg, surrogate, cycle0_cfg, str(run_dir))

    ckpt = run_dir / "agent_ckpt.pt"
    agent.save(str(ckpt))
    torch.save(surrogate, run_dir / "static_surrogate.pt")
    return ckpt


def main() -> None:
    parser = argparse.ArgumentParser(description="Train comparison methods for charging experiment.")
    parser.add_argument("--config", required=True, help="YAML config path.")
    parser.add_argument("--output", default="runs/other_method", help="Output root directory.")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--population", type=int, default=12, help="GA population size.")
    parser.add_argument("--generations", type=int, default=8, help="GA generations.")
    parser.add_argument(
        "--methods",
        nargs="+",
        default=["multistage_cc", "ga_cc", "td3", "ddpg", "ppo"],
        help="Subset of methods to train/build.",
    )
    args = parser.parse_args()

    cfg = load_config(args.config).data
    set_global_seed(args.seed)

    output_dir = Path(args.output)
    models_dir = output_dir / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    manifest: Dict[str, Dict[str, Any]] = {}

    if "multistage_cc" in args.methods:
        multistage_schedule = [
            {"soc_upper": 0.40, "current": -20.0},
            {"soc_upper": 0.70, "current": -12.0},
            {"soc_upper": 0.90, "current": -7.0},
            {"soc_upper": 1.10, "current": -3.0},
        ]
        path = models_dir / "multistage_cc_schedule.json"
        path.write_text(json.dumps(multistage_schedule, indent=2), encoding="utf-8")
        manifest["multistage_cc"] = {"type": "schedule", "path": str(path)}

    if "ga_cc" in args.methods:
        env = _build_pack_env(cfg["env"])
        ga_schedule = run_simple_ga(env, seed=args.seed, population=args.population, generations=args.generations)
        path = models_dir / "ga_schedule.json"
        path.write_text(json.dumps(ga_schedule, indent=2), encoding="utf-8")
        manifest["ga_cc"] = {"type": "schedule", "path": str(path)}

    for algo in ["td3", "ddpg", "ppo"]:
        if algo in args.methods:
            ckpt = train_rl_method(cfg, algo=algo, output_dir=models_dir)
            manifest[algo] = {"type": "rl_agent", "algorithm": algo, "path": str(ckpt)}

    manifest_path = models_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")

    config_snapshot = output_dir / "train_config_snapshot.json"
    config_snapshot.write_text(json.dumps(cfg, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Saved model manifest to: {manifest_path}")


if __name__ == "__main__":
    main()