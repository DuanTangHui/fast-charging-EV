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
    from src.envs.liionpack_spme_pack_env import build_pack_env
    return build_pack_env(env_cfg)



def evaluate_stage_schedule(
    env,
    schedule: List[Dict[str, float]],
    seed: int,
    alpha: float = 0.7,     # 时间权重，参考论文可偏大
    beta: float = 0.3,      # 温升权重
    temp_budget: float = 8.0,
    voltage_soft: float = 4.10,
) -> float:
    """
    论文思路的 pack 适配版：
    - 优化目标：更短充电时间 + 更低温升
    - 同时保留电压/温度硬约束惩罚
    - 若没有达到 0.8 SOC，给予较大惩罚
    """
    low = float(env.action_space.low[0])
    high = float(env.action_space.high[0])

    policy = stage_policy_from_schedule(schedule, action_low=low, action_high=high)
    traj, summary = rollout_one_episode(env, policy, seed=seed)

    soc_end = float(summary["soc_end"])
    steps = float(summary["steps"])

    # 取轨迹起终温度估计温升
    if len(traj) > 0:
        t0 = float(traj[0]["tmax"])
        t_end = max(float(row["tmax"]) for row in traj)
        vmax_peak = max(float(row["vmax"]) for row in traj)
    else:
        t0 = 298.15
        t_end = 298.15
        vmax_peak = 0.0

    delta_t = max(0.0, t_end - t0)

    # ---------- 论文风格：时间 + 温升 双目标 ----------
    # 时间越短越好：最大步长720步，归一化后越大越好
    time_score = 1.0 - steps / 720.0
    # 温升越小越好：temp_budget 可按经验设定
    temp_score = 1.0 - min(delta_t / temp_budget, 1.5)

    score = alpha * time_score + beta * temp_score

    # ---------- 终点约束 ----------
    # 没有充到 0.8，强罚
    if soc_end < 0.80:
        score -= 8.0 * (0.80 - soc_end)

    # ---------- 硬约束惩罚 ----------
    penalty = 0.0
    for row in traj:
        vmax = float(row["vmax"])
        tmax = float(row["tmax"])

        # 电压超限
        penalty += 200.0 * max(0.0, vmax - float(env.v_max))
        # 温度超限
        penalty += 10.0 * max(0.0, tmax - float(env.t_max))
        # 提前抑制接近过压
        penalty += 25.0 * max(0.0, vmax - voltage_soft)

    # 峰值过压额外惩罚
    penalty += 50.0 * max(0.0, vmax_peak - float(env.v_max))

    return score - penalty


def run_simple_ga(
    env,
    seed: int,
    population: int,
    generations: int,
) -> List[Dict[str, float]]:
    """
    更接近论文思路的分段 GA：
    1) 在每个 SOC 段内独立搜索电流，不再先全局排序
    2) 使用“幅值”编码，保证阶段电流整体递减更自然
    3) 搜索空间不是 baseline 本身，而是每段一个可调范围
    """
    rng = np.random.default_rng(seed)

    soc_breaks = [0.30, 0.60, 0.70, 0.80]

    # 环境动作边界（负号表示充电）
    action_low = float(env.action_space.low[0])   # e.g. -40
    action_high = float(env.action_space.high[0]) # e.g. 0

    # 用“电流幅值”做编码，更方便处理单调递减
    global_amp_max = abs(action_low)

    # 每段允许搜索的幅值范围（单位 A）
    # 这是“基于 baseline 扩展出来的范围”，不是把 baseline 锁死
    # 你可按 pack 实际再微调
    amp_bounds = [
        (18.0, min(30.0, global_amp_max)),  # stage 1, 对应大约 -18 ~ -30 A
        (8.0, 20.0),                        # stage 2, 对应大约 -8  ~ -20 A
        (3.0, 12.0),                        # stage 3, 对应大约 -3  ~ -12 A
        (1.0, 6.0),                         # stage 4, 对应大约 -1  ~ -6  A
    ]

    # 防止超过环境边界
    amp_bounds = [(lo, min(hi, global_amp_max)) for lo, hi in amp_bounds]

    def repair_amplitudes(amps: List[float]) -> List[float]:
        """
        强制阶段幅值非增：
        a1 >= a2 >= a3 >= a4
        这样转换成负电流后就是：
        I1 <= I2 <= I3 <= I4 (数值上越来越接近0)
        即充电电流绝对值逐段减小
        """
        amps = [float(a) for a in amps]

        # 先裁剪到各段边界
        for i, (lo, hi) in enumerate(amp_bounds):
            amps[i] = float(np.clip(amps[i], lo, hi))

        # 再做单调修复：前一段幅值 >= 后一段幅值
        for i in range(1, len(amps)):
            if amps[i] > amps[i - 1]:
                amps[i] = amps[i - 1]

        # 修复后再次保证不低于各段下界
        for i in range(len(amps) - 2, -1, -1):
            lo_i, hi_i = amp_bounds[i]
            lo_next, hi_next = amp_bounds[i + 1]

            amps[i + 1] = float(np.clip(amps[i + 1], lo_next, min(hi_next, amps[i])))
            amps[i] = float(np.clip(amps[i], lo_i, hi_i))

        return amps

    def amps_to_schedule(amps: List[float]) -> List[Dict[str, float]]:
        amps = repair_amplitudes(amps)
        currents = [-a for a in amps]  # 转成负电流（充电）
        currents = [float(np.clip(c, action_low, action_high)) for c in currents]
        return [
            {"soc_upper": soc_breaks[i], "current": currents[i]}
            for i in range(4)
        ]

    def random_schedule() -> List[Dict[str, float]]:
        amps = [rng.uniform(lo, hi) for lo, hi in amp_bounds]
        return amps_to_schedule(amps)

    def schedule_to_amps(schedule: List[Dict[str, float]]) -> List[float]:
        return [abs(float(stage["current"])) for stage in schedule]

    pop = [random_schedule() for _ in range(population)]
    best = pop[0]
    best_score = -1e18

    elite_num = max(4, population // 5)

    for _ in range(generations):
        scored: List[Tuple[float, List[Dict[str, float]]]] = []

        for ind in pop:
            score = evaluate_stage_schedule(
                env,
                ind,
                seed=seed,
                alpha=0.7,
                beta=0.3,
                temp_budget=8.0,
                voltage_soft=4.10,
            )
            scored.append((score, ind))

        scored.sort(key=lambda x: x[0], reverse=True)

        if scored[0][0] > best_score:
            best_score = scored[0][0]
            best = scored[0][1]

        elites = [ind for _, ind in scored[:elite_num]]
        next_pop = elites.copy()

        while len(next_pop) < population:
            p1 = elites[rng.integers(0, len(elites))]
            p2 = elites[rng.integers(0, len(elites))]

            a1 = schedule_to_amps(p1)
            a2 = schedule_to_amps(p2)

            child_amps = []
            for i in range(4):
                # 算术交叉
                lam = rng.uniform(0.2, 0.8)
                base = lam * a1[i] + (1.0 - lam) * a2[i]

                # 变异：不同阶段给不同扰动
                sigma = [2.0, 1.5, 1.0, 0.5][i]
                val = base + rng.normal(0.0, sigma)
                child_amps.append(val)

            child_amps = repair_amplitudes(child_amps)
            child = amps_to_schedule(child_amps)
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
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", default="runs/other_method")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--population", type=int, default=40)
    parser.add_argument("--generations", type=int, default=30)
    parser.add_argument("--methods", nargs="+", default=["multistage_cc", "ga_cc"])
    args = parser.parse_args()

    cfg = load_config(args.config).data
    set_global_seed(args.seed)

    output_dir = Path(args.output)
    models_dir = output_dir / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    manifest: Dict[str, Dict[str, Any]] = {}

    # =========================
    # 修改②：baseline补到0.8
    # =========================
    if "multistage_cc" in args.methods:
        multistage_schedule = [
            {"soc_upper": 0.30, "current": -25.0},
            {"soc_upper": 0.60, "current": -15.0},
            {"soc_upper": 0.70, "current": -8.0},
            {"soc_upper": 0.80, "current": -4.0},
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

    print(f"Saved model manifest to: {manifest_path}")


if __name__ == "__main__":
    main()