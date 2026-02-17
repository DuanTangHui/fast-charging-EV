# scripts/baseline_cc_vs_taper.py
from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Tuple
import numpy as np
import sys
# 获取当前脚本的绝对路径
current_dir = Path(__file__).resolve().parent
# 获取项目根目录 (即 scripts 的上一级)
root_dir = current_dir.parent
# 将根目录添加到 Python 搜索路径中
sys.path.append(str(root_dir))

# ==== 按你工程实际路径改一下 import ====
# 你训练脚本里就是这么用的：
# config = load_config(args.config).data
# set_global_seed(config.get("seed"))
# env = build_pack_env(config["env"])
from src.utils.config import load_config
from src.utils.seeds import set_global_seed
from src.envs.liionpack_spme_pack_env import build_pack_env
# python scripts/baseline_cc.py --config configs/pack_3p6s_spme.yaml  --episodes 5 --no_csv 

@dataclass
class EpisodeSummary:
    policy_name: str
    episode_idx: int
    seed: int
    steps: int
    soc_end: float
    v_cell_max: float
    t_cell_max: float
    i_mean: float
    violations: int
    terminated_reason: str


def _get_soc_pack_from_info(info: Dict) -> float:
    # 你 _build_info 里传了 soc ndarray
    soc = info.get("soc", None)
    if soc is None:
        # 兜底：有些 info 可能直接给 SOC_pack
        return float(info.get("SOC_pack", np.nan))
    soc = np.asarray(soc, dtype=float)
    return float(np.mean(soc))


def _get_vmax_from_info(info: Dict) -> float:
    v_cells = info.get("v_cells", None)
    if v_cells is None:
        return float(info.get("V_cell_max", np.nan))
    v_cells = np.asarray(v_cells, dtype=float)
    return float(np.max(v_cells))


def _get_tmax_from_info(info: Dict) -> float:
    t_cells = info.get("t_cells", None)
    if t_cells is None:
        return float(info.get("T_cell_max", np.nan))
    t_cells = np.asarray(t_cells, dtype=float)
    return float(np.max(t_cells))


def _action(a: float) -> np.ndarray:
    # 你的 env.step 里用 action[0]
    return np.array([float(a)], dtype=np.float32)


def make_cc_policy(i_cc: float, low: float, high: float) -> Callable[[Dict], np.ndarray]:
    """恒流策略：一直输出 i_cc（负数表示充电）。"""
    i_cc = float(np.clip(i_cc, low, high))

    def _pi(obs: np.ndarray, info: Dict) -> np.ndarray:
        return _action(i_cc)

    return _pi


def make_taper_policy(
    i_cc: float,
    low: float,
    high: float,
    v_limit: float,
    v_soft_margin: float = 0.05,  # 4.15 if v_limit=4.2
    soc_soft: float = 0.79,
    soc_full: float = 0.80,
) -> Callable[[np.ndarray, Dict], np.ndarray]:
    """
    一个“很干净但有效”的快满降流策略（类似 CC->CV 的简化版）：
    - 平时用恒流 i_cc
    - 当 Vmax 接近上限或 SOC 接近满，线性把电流收敛到 0
    """
    i_cc = float(np.clip(i_cc, low, high))
    v_soft = v_limit - v_soft_margin

    def _pi(obs: np.ndarray, info: Dict) -> np.ndarray:
        soc = _get_soc_pack_from_info(info)
        vmax = _get_vmax_from_info(info)

        # 1) 基于电压的 taper：v_soft ~ v_limit 区间把 |I| 线性拉到 0
        if np.isfinite(vmax) and vmax >= v_soft:
            # alpha=1 表示还没到 v_soft（不 taper）；alpha=0 表示到 v_limit（电流->0）
            alpha_v = (v_limit - vmax) / max(1e-6, (v_limit - v_soft))
            alpha_v = float(np.clip(alpha_v, 0.0, 1.0))
        else:
            alpha_v = 1.0

        # 2) 基于 SOC 的 taper：soc_soft ~ soc_full 区间把 |I| 线性拉到 0
        if np.isfinite(soc) and soc >= soc_soft:
            alpha_soc = (soc_full - soc) / max(1e-6, (soc_full - soc_soft))
            alpha_soc = float(np.clip(alpha_soc, 0.0, 1.0))
        else:
            alpha_soc = 1.0

        alpha = min(alpha_v, alpha_soc)

        i_out = i_cc * alpha  # i_cc 为负，alpha 越小越接近 0
        i_out = float(np.clip(i_out, low, high))
        return _action(i_out)

    return _pi


def run_one_episode(
    env,
    policy_name: str,
    policy: Callable[[np.ndarray, Dict], np.ndarray],
    reset_seed: int,
    save_traj_csv: Path | None = None,
    max_steps_override: int | None = None,
) -> Tuple[EpisodeSummary, List[Dict]]:
    obs, info = env.reset(seed=reset_seed)
    traj: List[Dict] = []

    i_list: List[float] = []
    violations = 0
    terminated_reason = "running"

    # 记录 step=0（reset）
    traj.append(
        {
            "k": 0,
            "t": float(info.get("t", 0.0)),
            "soc_pack": _get_soc_pack_from_info(info),
            "v_cell_max": _get_vmax_from_info(info),
            "t_cell_max": _get_tmax_from_info(info),
            "a_cmd": 0.0,
            "violation": bool(info.get("violation", False)),
            "reason": str(info.get("reason", "reset")),
        }
    )

    k = 0
    while True:
        k += 1
        if max_steps_override is not None and k >= max_steps_override:
            terminated_reason = "baseline_max_steps_override"
            break

        a = policy(obs, info)
        a_val = float(a[0])

        obs, r_env, terminated, truncated, info = env.step(a)

        v_max = _get_vmax_from_info(info)
        t_max = _get_tmax_from_info(info)
        soc_pack = _get_soc_pack_from_info(info)

        violation = bool(info.get("violation", False))
        if violation:
            violations += 1

        reason = str(info.get("reason", "running"))
        terminated_reason = reason

        i_list.append(a_val)

        traj.append(
            {
                "k": k,
                "t": float(info.get("t", k * getattr(env, "dt", np.nan))),
                "soc_pack": soc_pack,
                "v_cell_max": v_max,
                "t_cell_max": t_max,
                "a_cmd": a_val,
                "violation": violation,
                "reason": reason,
            }
        )

        if terminated or truncated:
            break

    soc_end = float(traj[-1]["soc_pack"])
    v_end = float(traj[-1]["v_cell_max"])
    t_end = float(traj[-1]["t_cell_max"])
    i_mean = float(np.mean(i_list)) if len(i_list) > 0 else 0.0

    summary = EpisodeSummary(
        policy_name=policy_name,
        episode_idx=-1,
        seed=reset_seed,
        steps=len(traj) - 1,
        soc_end=soc_end,
        v_cell_max=v_end,
        t_cell_max=t_end,
        i_mean=i_mean,
        violations=violations,
        terminated_reason=terminated_reason,
    )

    if save_traj_csv is not None:
        save_traj_csv.parent.mkdir(parents=True, exist_ok=True)
        with save_traj_csv.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(
                f,
                fieldnames=["k", "t", "soc_pack", "v_cell_max", "t_cell_max", "a_cmd", "violation", "reason"],
            )
            w.writeheader()
            for row in traj:
                w.writerow(row)

    return summary, traj


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--outdir", type=str, default="runs/baselines")
    parser.add_argument("--no_csv", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config).data
    set_global_seed(cfg.get("seed"))

    env = build_pack_env(cfg["env"])

    # action bounds：你的是 [-20, 0]
    low = float(env.action_space.low[0])
    high = float(env.action_space.high[0])

    # v_limit/t_limit 用 env 自身
    v_limit = float(getattr(env, "v_max", cfg["env"]["v_max"]))
    t_limit = float(getattr(env, "t_max", cfg["env"]["t_max"]))

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # ====== 你要的“干净对照组”策略集合 ======
    policies: List[Tuple[str, Callable]] = []

    # 1) 恒流（你可以增删）
    for i_cc in [-2.0, -5.0, -10.0, -15.0, -20.0]:
        policies.append((f"CC_{i_cc:.0f}A", make_cc_policy(i_cc, low, high)))

    # 2) 简单 taper（快满自动降流）
    policies.append(
        (
            "TAPER_CC_-20A",
            make_taper_policy(
                i_cc=-20.0,
                low=low,
                high=high,
                v_limit=v_limit,
                v_soft_margin=0.05,
                soc_soft=0.90,
                soc_full=0.995,
            ),
        )
    )

    # ====== 跑对照：每个 episode 用同一批 reset seeds ======
    base_seed = int(cfg.get("seed", 0))
    reset_seeds = [base_seed + 1000 + i for i in range(args.episodes)]

    all_summaries: List[EpisodeSummary] = []

    for ep_idx, s in enumerate(reset_seeds):
        print(f"\n=== Episode {ep_idx} reset_seed={s} ===")
        for policy_name, policy in policies:
            traj_path = None
            if not args.no_csv:
                traj_path = outdir / f"traj_ep{ep_idx:02d}_{policy_name}.csv"

            summary, _traj = run_one_episode(
                env=env,
                policy_name=policy_name,
                policy=policy,
                reset_seed=s,
                save_traj_csv=traj_path,
            )
            summary.episode_idx = ep_idx
            all_summaries.append(summary)

            print(
                f"[{policy_name:14s}] "
                f"steps={summary.steps:3d} "
                f"SOC_end={summary.soc_end:.4f} "
                f"Vmax={summary.v_cell_max:.4f} "
                f"Tmax={summary.t_cell_max:.2f} "
                f"I_mean={summary.i_mean:.3f} "
                f"viol={summary.violations} "
                f"reason={summary.terminated_reason}"
            )

    # ====== 汇总（按 policy 聚合）======
    print("\n=== Aggregate by policy (mean over episodes) ===")
    by_policy: Dict[str, List[EpisodeSummary]] = {}
    for sm in all_summaries:
        by_policy.setdefault(sm.policy_name, []).append(sm)

    for pname, rows in by_policy.items():
        soc_end = np.mean([r.soc_end for r in rows])
        vmax = np.mean([r.v_cell_max for r in rows])
        tmax = np.mean([r.t_cell_max for r in rows])
        imean = np.mean([r.i_mean for r in rows])
        viol = int(np.sum([r.violations for r in rows]))
        steps = np.mean([r.steps for r in rows])

        print(
            f"[{pname:14s}] "
            f"steps~{steps:6.1f} "
            f"SOC_end~{soc_end:.4f} "
            f"Vmax~{vmax:.4f} "
            f"Tmax~{tmax:.2f} "
            f"I_mean~{imean:.3f} "
            f"viol_total={viol}"
        )


if __name__ == "__main__":
    main()
