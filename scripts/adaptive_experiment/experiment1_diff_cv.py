"""实验1：差分/组合代理模型在各老化阶段的交叉验证精度。

对每个老化阶段的数据（前30回合）执行：2次重复 * 5折交叉验证，
输出并绘制 R2/MSE/MAE 随老化阶段变化曲线。
"""
from __future__ import annotations

import argparse
import csv
import re
import sys
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict
from pathlib import Path
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch

CURRENT_DIR = Path(__file__).resolve().parent
ROOT_DIR = CURRENT_DIR.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from src.surrogate.dataset import build_dataset
from src.surrogate.gp_combined import CombinedSurrogate
from src.surrogate.gp_differential import DifferentialSurrogate
from src.utils.config import load_config
from src.utils.seeds import set_global_seed


def load_dataset_with_episode(path: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    data = np.genfromtxt(path, delimiter=",", names=True, dtype=np.float32)
    if "episode" not in data.dtype.names:
        raise ValueError(f"{path} must contain an 'episode' column")
    # NOTE:
    # np.genfromtxt returns a 0-d structured scalar when the csv only has one
    # data row, making stack(..., axis=1) fail with AxisError. Normalizing each
    # column to >=1D keeps both single-row and multi-row datasets consistent.
    states = np.stack([np.atleast_1d(data[f"s_{i}"]) for i in range(7)], axis=1)
    actions = np.atleast_1d(data["action"]).reshape(-1, 1)
    deltas = np.stack([np.atleast_1d(data[f"d_{i}"]) for i in range(6)], axis=1)
    episodes = np.atleast_1d(data["episode"]).astype(np.int32)
    return states, actions, deltas, episodes


def kfold_indices(n: int, k: int, seed: int) -> List[Tuple[np.ndarray, np.ndarray]]:
    rng = np.random.default_rng(seed)
    idx = np.arange(n)
    rng.shuffle(idx)
    folds = np.array_split(idx, k)
    return [(np.concatenate([folds[j] for j in range(k) if j != i]), folds[i]) for i in range(k)]


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    err = y_true - y_pred
    mse = float(np.mean(err**2))
    mae = float(np.mean(np.abs(err)))
    ss_res = float(np.sum(err**2))
    ss_tot = float(np.sum((y_true - np.mean(y_true, axis=0, keepdims=True)) ** 2))
    r2 = 1.0 - ss_res / (ss_tot + 1e-12)
    return {"r2": r2, "mse": mse, "mae": mae}


def _parse_stage_from_path(path: Path) -> int:
    m = re.search(r"adaptive_cycle(\d+)", str(path).replace("\\", "/"))
    if not m:
        raise ValueError(f"Cannot parse stage id from path: {path}")
    return int(m.group(1))


def discover_datasets(adaptive_root: Path, dataset_relpath: str) -> List[Path]:
    pattern = f"adaptive_cycle*/{dataset_relpath}"
    matches = sorted(adaptive_root.glob(pattern), key=_parse_stage_from_path)
    if not matches:
        raise FileNotFoundError(
            f"No dataset matched under {adaptive_root} with relative path '{dataset_relpath}'"
        )
    return matches


def evaluate_stage_cv(
    states: np.ndarray,
    actions: np.ndarray,
    deltas: np.ndarray,
    static_surrogate: object,
    hidden_sizes: List[int],
    ensemble_size: int,
    lr: float,
    surrogate_epochs: int,
    k_folds: int,
    repeats: int,
    seed: int,
) -> Dict[str, float]:
    rep_metrics: List[Dict[str, float]] = []

    for r in range(repeats):
        folds = kfold_indices(states.shape[0], k_folds, seed + r)
        fold_metrics: List[Dict[str, float]] = []
        for tr_idx, va_idx in folds:
            residual_transitions = []
            for s, a, d_real in zip(states[tr_idx], actions[tr_idx], deltas[tr_idx]):
                d_static, _ = static_surrogate.predict(s, a)
                d_hat = d_real - d_static
                residual_transitions.append((s, a, d_hat))

            ds = build_dataset(residual_transitions)
            diff = DifferentialSurrogate(
                input_dim=8,
                output_dim=6,
                hidden_sizes=hidden_sizes,
                ensemble_size=ensemble_size,
                lr=lr,
            )
            diff.fit(ds, epochs=surrogate_epochs)
            combined = CombinedSurrogate(static_surrogate, diff)

            preds: List[np.ndarray] = []
            for s, a in zip(states[va_idx], actions[va_idx]):
                d_pred, _ = combined.predict(s, a)
                preds.append(d_pred)
            fold_metrics.append(regression_metrics(deltas[va_idx], np.stack(preds, axis=0)))

        rep_metrics.append({
            "r2": float(np.mean([m["r2"] for m in fold_metrics])),
            "mse": float(np.mean([m["mse"] for m in fold_metrics])),
            "mae": float(np.mean([m["mae"] for m in fold_metrics])),
        })

    return {
        "r2": float(np.mean([m["r2"] for m in rep_metrics])),
        "mse": float(np.mean([m["mse"] for m in rep_metrics])),
        "mae": float(np.mean([m["mae"] for m in rep_metrics])),
    }

def plot_metrics(rows: List[Dict[str, float]], out_dir: Path) -> None:
    import logging
    import os
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    from matplotlib.ticker import MultipleLocator  # 新增：用于控制刻度间隔

    # 1. 强制屏蔽字体内部元数据警告
    logging.getLogger('fontTools.subset').level = logging.ERROR

    # 2. 加载宋体
    simsun_path = r'C:\Windows\Fonts\simsun.ttc'
    if os.path.exists(simsun_path):
        font_manager.fontManager.addfont(simsun_path)

    # 3. 顶刊全局标准配置
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["SimSun"], 
        "mathtext.fontset": "custom",
        "mathtext.rm": "Times New Roman",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "axes.unicode_minus": False,
        
        # 边框与刻度
        "axes.spines.top": True,
        "axes.spines.right": True,
        "axes.linewidth": 1.0,
        "xtick.major.width": 1.0,
        "ytick.major.width": 1.0,
        "xtick.direction": "in",
        "ytick.direction": "in",
        
        # 严格的 10.5pt 字号体系 (对应 Word 五号字)
        "font.size": 10.5,                     
        "axes.labelsize": 10.5,
        "xtick.labelsize": 10.5,
        "ytick.labelsize": 10.5,
        "legend.fontsize": 9  
    })

    # Nature 经典学术配色
    color_r2 = '#3C5488'    # 深蓝色
    color_mse = '#E64B35'   # 红色
    color_mae = '#00A087'   # 蓝绿色

    stages = np.array([int(r["stage"]) for r in rows])
    r2 = np.array([r["r2"] for r in rows])
    mse = np.array([r["mse"] for r in rows])
    mae = np.array([r["mae"] for r in rows])

    # 确保输出目录存在
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # 物理尺寸规范：单张局部图 8.0cm × 5.5cm
    cm_to_inch = 1 / 2.54
    figsize_single = (8.0 * cm_to_inch, 5.5 * cm_to_inch)

    # ==========================================
    # 1. 绘制 R2 vs Aging Stage
    # ==========================================
    fig1, ax1 = plt.subplots(figsize=figsize_single)
    
    ax1.plot(stages, r2, marker="o", markersize=5, linewidth=1.5, 
             color=color_r2, markeredgecolor="white", markeredgewidth=0.8)
    
    ax1.set_xlabel(r"老化阶段")
    ax1.set_ylabel(r"决定系数 ($R^2$)") 
    
    ax1.grid(True, linestyle=":", alpha=0.6, color="#CCCCCC")
    
    # 【核心修改点】设置 x 轴刻度为主刻度每 5 个单位一标
    ax1.xaxis.set_major_locator(MultipleLocator(5))
    ax1.set_xlim(left=0, right=max(stages) + 1) # 适当留白，防止最右侧点贴边
    
    ax1.tick_params(axis='x', pad=2, length=3)
    ax1.tick_params(axis='y', pad=2, length=3)
    for label in ax1.get_xticklabels() + ax1.get_yticklabels():
        label.set_fontname('Times New Roman')

    fig1.tight_layout()
    fig1.savefig(out_dir / "exp1_r2_vs_stage.pdf", format='pdf', bbox_inches="tight")
    fig1.savefig(out_dir / "exp1_r2_vs_stage.png", dpi=300, bbox_inches="tight")
    plt.close(fig1)

    # ==========================================
    # 2. 绘制 MSE/MAE vs Aging Stage
    # ==========================================
    fig2, ax2 = plt.subplots(figsize=figsize_single)
    
    ax2.plot(stages, mse, marker="o", markersize=5, linewidth=1.5, 
             label=r"均方误差", color=color_mse, markeredgecolor="white", markeredgewidth=0.8)
    ax2.plot(stages, mae, marker="s", markersize=5, linewidth=1.5, 
             label=r"平均绝对误差", color=color_mae, markeredgecolor="white", markeredgewidth=0.8)
    
    ax2.set_xlabel(r"老化阶段")
    ax2.set_ylabel(r"误差值")
    
    ax2.grid(True, linestyle=":", alpha=0.6, color="#CCCCCC")
    
    # 【核心修改点】设置 x 轴刻度为主刻度每 5 个单位一标
    ax2.xaxis.set_major_locator(MultipleLocator(5))
    ax2.set_xlim(left=0, right=max(stages) + 1) # 适当留白
    
    ax2.legend(
        loc='best', 
        ncol=1,                    
        frameon=True,              
        facecolor='white',         
        framealpha=0.9,            
        edgecolor=(0.7, 0.7, 0.7, 0.5),
        borderpad=0.4,       
        handletextpad=0.3,   
        labelspacing=0.4
    )
    
    ax2.tick_params(axis='x', pad=2, length=3)
    ax2.tick_params(axis='y', pad=2, length=3)
    for label in ax2.get_xticklabels() + ax2.get_yticklabels():
        label.set_fontname('Times New Roman')

    fig2.tight_layout()
    fig2.savefig(out_dir / "exp1_mse_mae_vs_stage.pdf", format='pdf', bbox_inches="tight")
    fig2.savefig(out_dir / "exp1_mse_mae_vs_stage.png", dpi=300, bbox_inches="tight")
    plt.close(fig2)
    
    print(f"[Done] 评估指标图表(5刻度间隔版)已保存至: {out_dir}")
# def plot_metrics(rows: List[Dict[str, float]], out_dir: Path) -> None:
#     stages = np.array([int(r["stage"]) for r in rows])
#     r2 = np.array([r["r2"] for r in rows])
#     mse = np.array([r["mse"] for r in rows])
#     mae = np.array([r["mae"] for r in rows])

#     plt.figure(figsize=(8, 5))
#     plt.plot(stages, r2, marker="o")
#     plt.xlabel("Aging Stage")
#     plt.ylabel("R2")
#     plt.title("R2 vs Aging Stage")
#     plt.grid(True, alpha=0.3)
#     plt.tight_layout()
#     plt.savefig(out_dir / "exp1_r2_vs_stage.png", dpi=180)
#     plt.close()

#     plt.figure(figsize=(8, 5))
#     plt.plot(stages, mse, marker="o", label="MSE")
#     plt.plot(stages, mae, marker="s", label="MAE")
#     plt.xlabel("Aging Stage")
#     plt.ylabel("Error")
#     plt.title("MSE/MAE vs Aging Stage")
#     plt.grid(True, alpha=0.3)
#     plt.legend()
#     plt.tight_layout()
#     plt.savefig(out_dir / "exp1_mse_mae_vs_stage.png", dpi=180)
#     plt.close()


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/pack_3p6s_spme_with_soh_prior.yaml")
    p.add_argument("--static-surrogate", default="runs/cycle00/static_surrogate.pt")
    p.add_argument("--datasets", nargs="+", default=None)
    p.add_argument("--adaptive-root", default="runs/adaptive")
    p.add_argument(
        "--dataset-relpath",
        default="cycle_1/episodes/episode_30/dataset_with_episode.csv",
        help="relative path under each adaptive_cycle* directory for auto discovery",
    )
    p.add_argument("--max-episode", type=int, default=30)
    p.add_argument("--k-folds", type=int, default=5)
    p.add_argument("--repeats", type=int, default=2)
    p.add_argument("--surrogate-epochs", type=int, default=20)
    p.add_argument("--output-dir", default="runs/adaptive_experiment/result1")
    args = p.parse_args()

    cfg = load_config(args.config).data
    set_global_seed(cfg.get("seed", 42))

    static_ckpt = Path(args.static_surrogate)
    if not static_ckpt.exists():
        raise FileNotFoundError(f"Missing static surrogate: {static_ckpt}")
    static_surrogate = torch.load(static_ckpt, map_location="cpu", weights_only=False)

    if args.datasets:
        dataset_paths = [Path(ds_str) for ds_str in args.datasets]
        print(f"[EXP1] Using manual datasets, count={len(dataset_paths)}")
    else:
        dataset_paths = discover_datasets(Path(args.adaptive_root), args.dataset_relpath)
        print(f"[EXP1] Auto-discovered datasets, count={len(dataset_paths)}")

    rows: List[Dict[str, float]] = []
    for ds_path in dataset_paths:
        if not ds_path.exists():
            raise FileNotFoundError(f"Missing dataset: {ds_path}")

        stage = _parse_stage_from_path(ds_path)
        s, a, d, ep = load_dataset_with_episode(ds_path)
        mask = ep <= args.max_episode
        s, a, d = s[mask], a[mask], d[mask]
        if s.shape[0] < args.k_folds:
            print(f"[WARNING] ⚠️ 跳过 stage={stage}：样本不足 (n={s.shape[0]} < k_folds={args.k_folds})。")
            continue
        m = evaluate_stage_cv(
            s,
            a,
            d,
            static_surrogate=static_surrogate,
            hidden_sizes=cfg["surrogate"]["hidden_sizes"],
            ensemble_size=cfg["surrogate"]["ensemble_size"],
            lr=cfg["surrogate"]["learning_rate"],
            surrogate_epochs=args.surrogate_epochs,
            k_folds=args.k_folds,
            repeats=args.repeats,
            seed=cfg.get("seed", 42),
        )
        row = {"stage": float(stage), "samples": float(s.shape[0]), **m}
        rows.append(row)
        print(f"[EXP1] stage={stage}, n={s.shape[0]}, R2={m['r2']:.4f}, MSE={m['mse']:.6e}, MAE={m['mae']:.6e}")

    rows = sorted(rows, key=lambda x: x["stage"])
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    csv_path = out_dir / "exp1_stage_cv_metrics.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["stage", "samples", "r2", "mse", "mae"])
        w.writeheader()
        w.writerows(rows)

    plot_metrics(rows, out_dir)
    print(f"[DONE] Saved metrics: {csv_path}")


if __name__ == "__main__":
    main()