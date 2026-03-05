"""静态代理模型有效性分析实验。

1) 基于统计指标的交叉验证：
   - 使用带 episode 列的数据文件（dataset_with_episode.csv）
   - 对 E=20..50，每个 E 采用累计数据 (episode<=E)
   - 在每个 E 上执行 5-fold，并重复 2 次，得到平均 R² / MSE / MAE 曲线

2) 完整环境仿真对比：
   - 同一 RL 智能体分别在真实物理环境 (liionpack+PyBaMM) 与静态代理上交互
   - 单回合，max_steps=720，SOC>=0.8 终止
   - 输出电流/电压/SOC/温度四条曲线与耗时对比
"""
from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch

current_dir = Path(__file__).resolve().parent
root_dir = current_dir.parent
sys.path.append(str(root_dir))

from src.envs.liionpack_spme_pack_env import build_pack_env
from src.rl.agent_factory import build_agent_from_config
from src.surrogate.dataset import build_dataset
from src.surrogate.gp_static import StaticSurrogate
from src.utils.config import load_config
from src.utils.seeds import set_global_seed


# Nature-like colors
NATURE_BLUE = "#3B6FB6"
NATURE_GREEN = "#7AA974"
GRID_COLOR = "#D0D0D0"


def load_dataset_with_episode(path: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    data = np.genfromtxt(path, delimiter=",", names=True, dtype=np.float32)
    if "episode" not in data.dtype.names:
        raise ValueError(f"{path} must contain an 'episode' column")
    states = np.stack([data[f"s_{i}"] for i in range(7)], axis=1)
    actions = data["action"].reshape(-1, 1)
    deltas = np.stack([data[f"d_{i}"] for i in range(6)], axis=1)
    episodes = data["episode"].astype(np.int32)
    return states, actions, deltas, episodes


def make_transitions(states: np.ndarray, actions: np.ndarray, deltas: np.ndarray) -> List[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    return [(states[i], actions[i], deltas[i]) for i in range(states.shape[0])]


def kfold_indices(n: int, k: int, seed: int) -> List[Tuple[np.ndarray, np.ndarray]]:
    rng = np.random.default_rng(seed)
    idx = np.arange(n)
    rng.shuffle(idx)
    folds = np.array_split(idx, k)
    return [
        (np.concatenate([folds[j] for j in range(k) if j != i]), folds[i])
        for i in range(k)
    ]


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    err = y_true - y_pred
    mse = float(np.mean(err**2))
    mae = float(np.mean(np.abs(err)))
    ss_res = float(np.sum(err**2))
    ss_tot = float(np.sum((y_true - np.mean(y_true, axis=0, keepdims=True)) ** 2))
    r2 = 1.0 - ss_res / (ss_tot + 1e-12)
    return {"r2": r2, "mse": mse, "mae": mae}


def evaluate_cv_once(
    states: np.ndarray,
    actions: np.ndarray,
    deltas: np.ndarray,
    k_folds: int,
    epochs: int,
    hidden_sizes: List[int],
    ensemble_size: int,
    lr: float,
    seed: int,
) -> Dict[str, float]:
    folds = kfold_indices(states.shape[0], k_folds, seed)
    metrics = []
    for train_idx, val_idx in folds:
        train_ds = build_dataset(make_transitions(states[train_idx], actions[train_idx], deltas[train_idx]))
        model = StaticSurrogate(8, 6, hidden_sizes=hidden_sizes, ensemble_size=ensemble_size, lr=lr)
        model.fit(train_ds, epochs=epochs)
        preds = []
        for s, a in zip(states[val_idx], actions[val_idx]):
            d, _ = model.predict(s, a)
            preds.append(d)
        metrics.append(regression_metrics(deltas[val_idx], np.stack(preds, axis=0)))

    return {
        "r2": float(np.mean([m["r2"] for m in metrics])),
        "mse": float(np.mean([m["mse"] for m in metrics])),
        "mae": float(np.mean([m["mae"] for m in metrics])),
    }


def run_episode_curve_cv(
    states: np.ndarray,
    actions: np.ndarray,
    deltas: np.ndarray,
    episodes: np.ndarray,
    ep_start: int,
    ep_end: int,
    repeats: int,
    k_folds: int,
    cv_epochs: int,
    hidden_sizes: List[int],
    ensemble_size: int,
    lr: float,
    seed: int,
) -> List[Dict[str, float]]:
    rows: List[Dict[str, float]] = []
    for ep in range(ep_start, ep_end + 1):
        mask = episodes <= ep
        s_sub, a_sub, d_sub = states[mask], actions[mask], deltas[mask]
        if len(s_sub) < k_folds:
            continue

        rep_metrics = []
        for r in range(repeats):
            rep_metrics.append(
                evaluate_cv_once(
                    s_sub,
                    a_sub,
                    d_sub,
                    k_folds=k_folds,
                    epochs=cv_epochs,
                    hidden_sizes=hidden_sizes,
                    ensemble_size=ensemble_size,
                    lr=lr,
                    seed=seed + r,
                )
            )

        row = {
            "episode": float(ep),
            "r2": float(np.mean([m["r2"] for m in rep_metrics])),
            "mse": float(np.mean([m["mse"] for m in rep_metrics])),
            "mae": float(np.mean([m["mae"] for m in rep_metrics])),
            "samples": float(len(s_sub)),
        }
        print(f"[CV-Curve] Ep={ep}, N={len(s_sub)}, R2={row['r2']:.4f}, MSE={row['mse']:.6e}, MAE={row['mae']:.6e}")
        rows.append(row)
    return rows


def save_episode_curve_csv(rows: List[Dict[str, float]], path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["episode", "samples", "r2", "mse", "mae"])
        w.writeheader()
        w.writerows(rows)


def plot_r2_curve(rows: List[Dict[str, float]], path: Path) -> None:
    x = np.array([int(r["episode"]) for r in rows])
    y = np.array([r["r2"] for r in rows])

    fig, ax = plt.subplots(figsize=(6.0, 4.6))
    ax.plot(x, y, color=NATURE_BLUE, linewidth=2.0)
    ax.set_xlim(20, 50)
    ax.set_ylim(0.0, 1.0)
    ax.set_xlabel("Episode Number")
    ax.set_ylabel("5-Fold R$^2$ of GP [-]")
    ax.grid(True, color=GRID_COLOR, alpha=0.7)
    plt.tight_layout()
    plt.savefig(path, dpi=220)
    plt.close(fig)


def plot_mse_mae_curve(rows: List[Dict[str, float]], path: Path) -> None:
    x = np.array([int(r["episode"]) for r in rows])
    mse = np.array([r["mse"] for r in rows])
    mae = np.array([r["mae"] for r in rows])

    fig, ax1 = plt.subplots(figsize=(6.0, 4.6))
    ax2 = ax1.twinx()

    ax1.plot(x, mse, color=NATURE_BLUE, linewidth=2.0)
    ax2.plot(x, mae, color=NATURE_GREEN, linewidth=2.0)

    ax1.set_xlim(20, 50)
    ax1.set_xlabel("Episode Number")
    ax1.set_ylabel("5-Fold MSE of GP [-]", color=NATURE_BLUE)
    ax2.set_ylabel("5-Fold MAE of GP [-]", color=NATURE_GREEN)
    ax1.tick_params(axis="y", colors=NATURE_BLUE)
    ax2.tick_params(axis="y", colors=NATURE_GREEN)
    ax1.grid(True, color=GRID_COLOR, alpha=0.7)

    plt.tight_layout()
    plt.savefig(path, dpi=220)
    plt.close(fig)


def rollout_real_env(env, agent, seed: int, max_steps: int, soc_stop: float) -> Dict[str, np.ndarray]:
    s, info = env.reset(seed=seed)
    soc, volt, temp, curr = [float(info.get("SOC_pack", s[0]))], [float(info.get("V_cell_max", s[2]))], [float(info.get("T_cell_max", s[4]))], [0.0]
    t0 = time.perf_counter()
    for _ in range(max_steps):
        a = agent.act(s)
        s, _, term, trunc, info = env.step(a)
        soc.append(float(info.get("SOC_pack", s[0])))
        volt.append(float(info.get("V_cell_max", s[2])))
        temp.append(float(info.get("T_cell_max", s[4])))
        curr.append(float(info.get("I_pack_true", float(a[0]))))
        if soc[-1] >= soc_stop or term or trunc:
            break
    return {
        "soc": np.asarray(soc),
        "voltage": np.asarray(volt),
        "temperature": np.asarray(temp),
        "current": np.asarray(curr),
        "time_s": np.asarray([time.perf_counter() - t0], dtype=np.float32),
    }


def rollout_surrogate(initial_state: np.ndarray, agent, surrogate: StaticSurrogate, max_steps: int, soc_stop: float) -> Dict[str, np.ndarray]:
    s = initial_state.astype(np.float32).copy()
    soc, volt, temp, curr = [float(s[0])], [float(s[2])], [float(s[4])], [float(s[6])]
    t0 = time.perf_counter()
    for _ in range(max_steps):
        a = agent.act(s)
        d, _ = surrogate.predict(s, a)
        s_next = s.copy()
        s_next[:6] = s_next[:6] + d
        s_next[6] = float(a[0])
        s = s_next
        soc.append(float(s[0])); volt.append(float(s[2])); temp.append(float(s[4])); curr.append(float(s[6]))
        if soc[-1] >= soc_stop:
            break
    return {
        "soc": np.asarray(soc),
        "voltage": np.asarray(volt),
        "temperature": np.asarray(temp),
        "current": np.asarray(curr),
        "time_s": np.asarray([time.perf_counter() - t0], dtype=np.float32),
    }


def plot_rollout_compare(real: Dict[str, np.ndarray], gp: Dict[str, np.ndarray], path: Path) -> None:
    tr = np.arange(len(real["soc"]))
    tg = np.arange(len(gp["soc"]))
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes[0, 0].plot(tr, real["current"], color=NATURE_BLUE, label="Real Env")
    axes[0, 0].plot(tg, gp["current"], "--", color=NATURE_GREEN, label="Static GP")
    axes[0, 0].set_title("Current (A)")
    axes[0, 1].plot(tr, real["voltage"], color=NATURE_BLUE); axes[0, 1].plot(tg, gp["voltage"], "--", color=NATURE_GREEN); axes[0, 1].set_title("Voltage (V)")
    axes[1, 0].plot(tr, real["soc"], color=NATURE_BLUE); axes[1, 0].plot(tg, gp["soc"], "--", color=NATURE_GREEN); axes[1, 0].set_title("SOC")
    axes[1, 1].plot(tr, real["temperature"], color=NATURE_BLUE); axes[1, 1].plot(tg, gp["temperature"], "--", color=NATURE_GREEN); axes[1, 1].set_title("Temperature (K)")
    for ax in axes.ravel():
        ax.set_xlabel("Step")
        ax.grid(True, color=GRID_COLOR, alpha=0.7)
    axes[0, 0].legend()
    plt.tight_layout(); plt.savefig(path, dpi=220); plt.close(fig)


def plot_efficiency(real_t: float, gp_t: float, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(["liionpack+PyBaMM", "Static GP"], [real_t, gp_t], color=[NATURE_BLUE, NATURE_GREEN])
    ax.set_ylabel("Wall-clock time (s)")
    ax.set_title("Single-Episode Efficiency")
    plt.tight_layout(); plt.savefig(path, dpi=220); plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", type=Path, default=Path("dataset_with_episode.csv"))
    p.add_argument("--config", type=Path, default=Path("configs/pack_3p6s_spme.yaml"))
    p.add_argument("--agent-ckpt", type=Path, default=Path("runs/cycle0/agent_ckpt.pt"))
    p.add_argument("--surrogate-ckpt", type=Path, default=Path("runs/cycle0/static_surrogate.pt"))
    p.add_argument("--output-dir", type=Path, default=Path("runs/static_surrogate_validity_real"))
    p.add_argument("--episode-start", type=int, default=20)
    p.add_argument("--episode-end", type=int, default=50)
    p.add_argument("--k-folds", type=int, default=5)
    p.add_argument("--cv-repeats", type=int, default=2)
    p.add_argument("--cv-epochs", type=int, default=20)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--soc-stop", type=float, default=0.8)
    p.add_argument("--max-steps", type=int, default=720)
    args = p.parse_args()

    for req in [args.dataset, args.config, args.agent_ckpt, args.surrogate_ckpt]:
        if not req.exists():
            raise FileNotFoundError(f"Required file not found: {req}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    cfg = load_config(str(args.config)).data
    set_global_seed(args.seed)

    states, actions, deltas, episodes = load_dataset_with_episode(args.dataset)
    cv_rows = run_episode_curve_cv(
        states, actions, deltas, episodes,
        ep_start=args.episode_start,
        ep_end=args.episode_end,
        repeats=args.cv_repeats,
        k_folds=args.k_folds,
        cv_epochs=args.cv_epochs,
        hidden_sizes=cfg["surrogate"]["hidden_sizes"],
        ensemble_size=cfg["surrogate"]["ensemble_size"],
        lr=cfg["surrogate"]["learning_rate"],
        seed=args.seed,
    )
    save_episode_curve_csv(cv_rows, args.output_dir / "episode_cv_metrics.csv")
    plot_r2_curve(cv_rows, args.output_dir / "fig6_r2_vs_episode.png")
    plot_mse_mae_curve(cv_rows, args.output_dir / "fig7_mse_mae_vs_episode.png")

    env = build_pack_env(cfg["env"])
    agent = build_agent_from_config(state_dim=env.observation_space.shape[0], action_dim=1, rl_config=cfg["rl"])
    agent.load(str(args.agent_ckpt), map_location="cpu")

    surrogate = torch.load(args.surrogate_ckpt, map_location="cpu", weights_only=False)
    if not isinstance(surrogate, StaticSurrogate):
        raise TypeError("Loaded surrogate is not StaticSurrogate")

    real = rollout_real_env(env, agent, seed=args.seed, max_steps=args.max_steps, soc_stop=args.soc_stop)
    init_state, _ = env.reset(seed=args.seed)
    gp = rollout_surrogate(init_state, agent, surrogate, max_steps=args.max_steps, soc_stop=args.soc_stop)

    plot_rollout_compare(real, gp, args.output_dir / "full_rollout_compare.png")
    rt, gt = float(real["time_s"][0]), float(gp["time_s"][0])
    plot_efficiency(rt, gt, args.output_dir / "efficiency_compare.png")

    with (args.output_dir / "summary.txt").open("w", encoding="utf-8") as f:
        f.write("Static surrogate validity experiment summary (real env)\n")
        f.write(f"Episode range = [{args.episode_start}, {args.episode_end}]\n")
        f.write(f"CV scheme = {args.k_folds}-fold repeated {args.cv_repeats} times\n")
        f.write(f"CV average R2 (curve mean) = {np.mean([r['r2'] for r in cv_rows]):.6f}\n")
        f.write(f"CV average MSE (curve mean) = {np.mean([r['mse'] for r in cv_rows]):.6e}\n")
        f.write(f"CV average MAE (curve mean) = {np.mean([r['mae'] for r in cv_rows]):.6e}\n")
        f.write(f"Single rollout final SOC (real) = {real['soc'][-1]:.6f}\n")
        f.write(f"Single rollout final SOC (gp) = {gp['soc'][-1]:.6f}\n")
        f.write(f"Runtime real env = {rt:.6f} s\n")
        f.write(f"Runtime static GP = {gt:.6f} s\n")
        f.write(f"Speedup (real/gp) = {rt / (gt + 1e-12):.2f}x\n")

    print(f"[Done] output_dir={args.output_dir}")


if __name__ == "__main__":
    main()