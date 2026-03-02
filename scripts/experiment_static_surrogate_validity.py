"""静态代理模型有效性分析实验。

实验一：基于统计指标的 K 折交叉验证（R² / MSE / MAE）。
实验二：完整环境仿真对比（同一 RL 智能体 vs liionpack+PyBaMM 与静态代理）。
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


def load_dataset_csv(path: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    data = np.genfromtxt(path, delimiter=",", names=True, dtype=np.float32)
    states = np.stack([data[f"s_{i}"] for i in range(7)], axis=1)
    actions = data["action"].reshape(-1, 1)
    deltas = np.stack([data[f"d_{i}"] for i in range(6)], axis=1)
    return states, actions, deltas


def make_transitions(states: np.ndarray, actions: np.ndarray, deltas: np.ndarray) -> List[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    return [(states[i], actions[i], deltas[i]) for i in range(states.shape[0])]


def kfold_indices(n: int, k: int, seed: int) -> List[Tuple[np.ndarray, np.ndarray]]:
    rng = np.random.default_rng(seed)
    idx = np.arange(n)
    rng.shuffle(idx)
    folds = np.array_split(idx, k)
    return [
        (
            np.concatenate([folds[j] for j in range(k) if j != i]),
            folds[i],
        )
        for i in range(k)
    ]


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    err = y_true - y_pred
    mse = float(np.mean(err**2))
    mae = float(np.mean(np.abs(err)))
    ss_res = float(np.sum(err**2))
    y_mean = np.mean(y_true, axis=0, keepdims=True)
    ss_tot = float(np.sum((y_true - y_mean) ** 2))
    r2 = 1.0 - ss_res / (ss_tot + 1e-12)
    return {"r2": r2, "mse": mse, "mae": mae}


def run_cross_validation(
    states: np.ndarray,
    actions: np.ndarray,
    deltas: np.ndarray,
    k_folds: int,
    epochs: int,
    hidden_sizes: List[int],
    ensemble_size: int,
    lr: float,
    seed: int,
) -> List[Dict[str, float]]:
    folds = kfold_indices(states.shape[0], k_folds, seed)
    fold_metrics: List[Dict[str, float]] = []
    for fold_i, (train_idx, val_idx) in enumerate(folds, start=1):
        train_ds = build_dataset(make_transitions(states[train_idx], actions[train_idx], deltas[train_idx]))
        model = StaticSurrogate(8, 6, hidden_sizes=hidden_sizes, ensemble_size=ensemble_size, lr=lr)
        model.fit(train_ds, epochs=epochs)

        preds = []
        for s, a in zip(states[val_idx], actions[val_idx]):
            d, _ = model.predict(s, a)
            preds.append(d)
        metric = regression_metrics(deltas[val_idx], np.stack(preds, axis=0))
        metric["fold"] = float(fold_i)
        fold_metrics.append(metric)
        print(f"[CV] Fold {fold_i}/{k_folds}: R2={metric['r2']:.4f}, MSE={metric['mse']:.6e}, MAE={metric['mae']:.6e}")
    return fold_metrics


def save_metrics_csv(metrics: List[Dict[str, float]], save_path: Path) -> None:
    with save_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["fold", "r2", "mse", "mae"])
        w.writeheader()
        w.writerows(metrics)


def plot_cv_metrics(metrics: List[Dict[str, float]], save_path: Path) -> None:
    folds = [int(m["fold"]) for m in metrics]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    axes[0].bar(folds, [m["r2"] for m in metrics]); axes[0].set_title("Cross-Validation R²")
    axes[1].bar(folds, [m["mse"] for m in metrics]); axes[1].set_title("Cross-Validation MSE")
    axes[2].bar(folds, [m["mae"] for m in metrics]); axes[2].set_title("Cross-Validation MAE")
    for ax in axes:
        ax.set_xlabel("Fold")
    plt.tight_layout(); plt.savefig(save_path, dpi=180); plt.close(fig)


def rollout_real_env(env, agent, seed: int, max_steps: int, soc_stop: float) -> Dict[str, np.ndarray]:
    s, info = env.reset(seed=seed)
    soc, volt, temp, curr = [float(info.get("SOC_pack", s[0]))], [float(info.get("V_cell_max", s[2]))], [float(info.get("T_cell_max", s[4]))], [0.0]

    t0 = time.perf_counter()
    for _ in range(max_steps):
        a = agent.act(s)
        ns, _, term, trunc, ninfo = env.step(a)
        s = ns
        soc.append(float(ninfo.get("SOC_pack", s[0])))
        volt.append(float(ninfo.get("V_cell_max", s[2])))
        temp.append(float(ninfo.get("T_cell_max", s[4])))
        curr.append(float(ninfo.get("I_pack_true", float(a[0]))))
        if soc[-1] >= soc_stop or term or trunc:
            break
    elapsed = time.perf_counter() - t0
    return {
        "state0": s,
        "soc": np.asarray(soc),
        "voltage": np.asarray(volt),
        "temperature": np.asarray(temp),
        "current": np.asarray(curr),
        "time_s": np.asarray([elapsed], dtype=np.float32),
    }


def rollout_surrogate(initial_state: np.ndarray, agent, surrogate: StaticSurrogate, max_steps: int, soc_stop: float) -> Dict[str, np.ndarray]:
    s = initial_state.astype(np.float32).copy()
    soc, volt, temp, curr = [float(s[0])], [float(s[2])], [float(s[4])], [float(s[6])]

    t0 = time.perf_counter()
    for _ in range(max_steps):
        a = agent.act(s)
        d, _ = surrogate.predict(s, a)
        ns = s.copy()
        ns[:6] = ns[:6] + d
        ns[6] = float(a[0])
        s = ns
        soc.append(float(s[0]))
        volt.append(float(s[2]))
        temp.append(float(s[4]))
        curr.append(float(s[6]))
        if soc[-1] >= soc_stop:
            break
    elapsed = time.perf_counter() - t0
    return {
        "soc": np.asarray(soc),
        "voltage": np.asarray(volt),
        "temperature": np.asarray(temp),
        "current": np.asarray(curr),
        "time_s": np.asarray([elapsed], dtype=np.float32),
    }


def plot_rollout_compare(real: Dict[str, np.ndarray], gp: Dict[str, np.ndarray], save_path: Path) -> None:
    n = max(len(real["soc"]), len(gp["soc"]))
    tr = np.arange(len(real["soc"]))
    tg = np.arange(len(gp["soc"]))

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes[0, 0].plot(tr, real["current"], label="Real Env"); axes[0, 0].plot(tg, gp["current"], "--", label="Static GP"); axes[0, 0].set_title("Current (A)")
    axes[0, 1].plot(tr, real["voltage"], label="Real Env"); axes[0, 1].plot(tg, gp["voltage"], "--", label="Static GP"); axes[0, 1].set_title("Voltage (V)")
    axes[1, 0].plot(tr, real["soc"], label="Real Env"); axes[1, 0].plot(tg, gp["soc"], "--", label="Static GP"); axes[1, 0].set_title("SOC")
    axes[1, 1].plot(tr, real["temperature"], label="Real Env"); axes[1, 1].plot(tg, gp["temperature"], "--", label="Static GP"); axes[1, 1].set_title("Temperature (K)")
    for ax in axes.ravel():
        ax.set_xlabel("Step")
        ax.grid(True, alpha=0.25)
        ax.legend()
    plt.tight_layout(); plt.savefig(save_path, dpi=180); plt.close(fig)


def plot_efficiency(real_t: float, gp_t: float, save_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(["liionpack+PyBaMM", "Static GP"], [real_t, gp_t])
    ax.set_ylabel("Wall-clock time (s)")
    ax.set_title("Single-Episode Efficiency")
    plt.tight_layout(); plt.savefig(save_path, dpi=180); plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", type=Path, default=Path("dataset.csv"))
    p.add_argument("--config", type=Path, default=Path("configs/pack_3p6s_spme.yaml"))
    p.add_argument("--agent-ckpt", type=Path, default=Path("runs/cycle0/agent_ckpt.pt"))
    p.add_argument("--surrogate-ckpt", type=Path, default=Path("runs/cycle0/static_surrogate.pt"))
    p.add_argument("--output-dir", type=Path, default=Path("runs/static_surrogate_validity_real"))
    p.add_argument("--k-folds", type=int, default=5)
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

    states, actions, deltas = load_dataset_csv(args.dataset)
    cv = run_cross_validation(
        states, actions, deltas,
        k_folds=args.k_folds,
        epochs=args.cv_epochs,
        hidden_sizes=cfg["surrogate"]["hidden_sizes"],
        ensemble_size=cfg["surrogate"]["ensemble_size"],
        lr=cfg["surrogate"]["learning_rate"],
        seed=args.seed,
    )
    save_metrics_csv(cv, args.output_dir / "cv_metrics.csv")
    plot_cv_metrics(cv, args.output_dir / "cv_metrics.png")

    env = build_pack_env(cfg["env"])
    agent = build_agent_from_config(state_dim=env.observation_space.shape[0], action_dim=1, rl_config=cfg["rl"])
    agent.load(str(args.agent_ckpt), map_location="cpu")

    surrogate = torch.load(args.surrogate_ckpt, map_location="cpu", weights_only=False)
    if not isinstance(surrogate, StaticSurrogate):
        raise TypeError("Loaded surrogate is not StaticSurrogate.")

    real = rollout_real_env(env, agent, seed=args.seed, max_steps=args.max_steps, soc_stop=args.soc_stop)
    # 保证代理 rollout 与真实 rollout 从同一初始观测开始：重新 reset 获取初始状态
    init_state, _ = env.reset(seed=args.seed)
    gp = rollout_surrogate(init_state, agent, surrogate, max_steps=args.max_steps, soc_stop=args.soc_stop)

    plot_rollout_compare(real, gp, args.output_dir / "full_rollout_compare.png")
    rt, gt = float(real["time_s"][0]), float(gp["time_s"][0])
    plot_efficiency(rt, gt, args.output_dir / "efficiency_compare.png")

    with (args.output_dir / "summary.txt").open("w", encoding="utf-8") as f:
        f.write("Static surrogate validity experiment summary (real env)\n")
        f.write(f"CV average R2 = {np.mean([m['r2'] for m in cv]):.6f}\n")
        f.write(f"CV average MSE = {np.mean([m['mse'] for m in cv]):.6e}\n")
        f.write(f"CV average MAE = {np.mean([m['mae'] for m in cv]):.6e}\n")
        f.write(f"Single rollout final SOC (real) = {real['soc'][-1]:.6f}\n")
        f.write(f"Single rollout final SOC (gp) = {gp['soc'][-1]:.6f}\n")
        f.write(f"Runtime real env = {rt:.6f} s\n")
        f.write(f"Runtime static GP = {gt:.6f} s\n")
        f.write(f"Speedup (real/gp) = {rt / (gt + 1e-12):.2f}x\n")

    print(f"[Done] output_dir={args.output_dir}")


if __name__ == "__main__":
    main()
