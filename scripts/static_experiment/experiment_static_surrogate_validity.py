
"""静态代理模型有效性分析实验一：验证静态代理模型的拟合有效性

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
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple
import matplotlib.ticker as ticker
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib import font_manager
from matplotlib.font_manager import FontProperties

current_dir = Path(__file__).resolve().parent
root_dir = current_dir.parent.parent
if str(root_dir) not in sys.path:
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

CM_TO_INCH = 1.0 / 2.54
SUBPLOT_WIDTH_CM = 8.0
SUBPLOT_HEIGHT_CM = 5.5
FONT_SIZE = 9


def _register_font_if_exists(path: str) -> bool:
    if os.path.exists(path):
        font_manager.fontManager.addfont(path)
        return True
    return False


def _pick_font(candidates: List[str]) -> str:
    available = {f.name for f in font_manager.fontManager.ttflist}
    for name in candidates:
        if name in available:
            return name
    return candidates[-1]


# 尝试注册常见系统宋体路径（含 Windows）
for _path in [
    r"C:\Windows\Fonts\simsun.ttc",
    "/usr/share/fonts/truetype/arphic/uming.ttc",
    "/usr/share/fonts/opentype/noto/NotoSerifCJK-Regular.ttc",
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
]:
    _register_font_if_exists(_path)

ZH_FONT_NAME = _pick_font(["SimSun", "Songti SC", "Noto Serif CJK SC", "Noto Sans CJK SC", "AR PL UMing CN", "DejaVu Sans"])
EN_FONT_NAME = _pick_font(["Times New Roman", "Times", "Nimbus Roman", "DejaVu Serif"])

ZH_FONT = FontProperties(family=ZH_FONT_NAME, size=FONT_SIZE)
EN_FONT = FontProperties(family=EN_FONT_NAME, size=FONT_SIZE)


def setup_plot_style() -> None:
    
    plt.rcParams.update({
        'font.family': ['Times New Roman', 'SimSun'],  # 关键：先英文字体，再中文
        'axes.unicode_minus': False,
        'figure.dpi': 300,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "axes.spines.top": True,
        "axes.spines.right": True,
        "axes.linewidth": 0.5,
        "xtick.major.width": 0.5,
        "ytick.major.width": 0.5,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "font.size": 9,                     
        "axes.labelsize": 9,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9  
    })



def apply_axis_fonts(ax) -> None:
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontproperties(EN_FONT)
    for spine in ax.spines.values():
        spine.set_linewidth(0.8)

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



def load_episode_curve_csv(path: Path) -> List[Dict[str, float]]:
    rows: List[Dict[str, float]] = []
    with path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append({
                "episode": float(row["episode"]),
                "samples": float(row.get("samples", 0.0)),
                "r2": float(row.get("r2", 0.0)),
                "mse": float(row["mse"]),
                "mae": float(row["mae"]),
            })
    if not rows:
        raise ValueError(f"CSV is empty: {path}")
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
    # ax.grid(True, color=GRID_COLOR, alpha=0.7)
    plt.tight_layout()
    plt.savefig(path, dpi=220)
    plt.close(fig)

def plot_mse_mae_curve(rows: List[Dict[str, float]], out_dir: Path) -> None:
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MultipleLocator, ScalarFormatter

    setup_plot_style()


    # Nature 经典学术配色
    color_mse = '#E64B35'   # 红色 (用于MSE)
    color_mae = '#00A087'   # 蓝绿色 (用于MAE)

    # 解析数据
    episodes = np.array([int(r["episode"]) for r in rows])
    mse = np.array([r["mse"] for r in rows])
    mae = np.array([r["mae"] for r in rows])

    # 确保输出目录存在
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # 物理尺寸规范：单张局部图 8.0cm × 5.5cm
    cm_to_inch = 1 / 2.54
    figsize_single = (11.0 * cm_to_inch, 6.5 * cm_to_inch)

    # 辅助函数：统一设置科学计数法
    def setup_sci_limits(ax_axis):
        fmt = ScalarFormatter(useMathText=True)
        fmt.set_scientific(True)
        fmt.set_powerlimits((-1, 1))
        ax_axis.set_major_formatter(fmt)
        # 确保科学计数法的 10^n 字体也是 Times New Roman
        ax_axis.get_offset_text().set_fontname('Times New Roman')
        ax_axis.get_offset_text().set_fontsize(10.5)

    # 辅助函数：统一坐标轴刻度字体
    def apply_times_new_roman_ticks(*axes):
        for ax in axes:
            ax.tick_params(axis='x', pad=2, length=3)
            ax.tick_params(axis='y', pad=2, length=3)
            for label in ax.get_xticklabels() + ax.get_yticklabels():
                label.set_fontname('Times New Roman')

    # X 轴边界留白计算
    x_min, x_max = min(episodes), max(episodes)
    x_padding = (x_max - x_min) * 0.05 if x_max > x_min else 2

    # ==========================================
    # 1. 绘制 合并图 (双Y轴: MSE & MAE vs Episode)
    # ==========================================
    fig_comb, ax_mse_c = plt.subplots(figsize=figsize_single)
    ax_mae_c = ax_mse_c.twinx()
    
    line1, = ax_mse_c.plot(episodes, mse, marker="o", markersize=5, linewidth=1.5, 
                           color=color_mse, markeredgecolor="white", markeredgewidth=0.8, label="均方误差")
    line2, = ax_mae_c.plot(episodes, mae, marker="s", markersize=5, linewidth=1.5, 
                           color=color_mae, markeredgecolor="white", markeredgewidth=0.8, label="平均绝对误差")
    
    ax_mse_c.set_xlabel(r"回合数")
    ax_mse_c.set_ylabel(r"均方误差 (MSE)", color=color_mse)
    ax_mae_c.set_ylabel(r"平均绝对误差 (MAE)", color=color_mae)
    
    # 轴刻度颜色与曲线统一
    ax_mse_c.tick_params(axis='y', colors=color_mse)
    ax_mae_c.tick_params(axis='y', colors=color_mae)
    # 对于双Y轴，右侧轴(ax_mae_c)的边框颜色通常保持默认或跟随全局，这里保持默认即可
    
    # ax_mse_c.grid(True, linestyle=":", alpha=0.6, color="#CCCCCC")
    ax_mse_c.xaxis.set_major_locator(MultipleLocator(5))
    ax_mse_c.set_xlim(left=x_min - x_padding, right=x_max + x_padding)
    
    setup_sci_limits(ax_mse_c.yaxis)
    setup_sci_limits(ax_mae_c.yaxis)
    apply_times_new_roman_ticks(ax_mse_c, ax_mae_c)

    # 统一双轴图例
    lines = [line1, line2]
    labels = [l.get_label() for l in lines]
    leg = ax_mse_c.legend(lines, labels, loc='upper right')
    leg.get_frame().set_linewidth(0.5)

    fig_comb.tight_layout()
    fig_comb.savefig(out_dir / "fig7_combined_mse_mae.pdf", format='pdf', bbox_inches="tight")
    fig_comb.savefig(out_dir / "fig7_combined_mse_mae.png", dpi=300, bbox_inches="tight")
    plt.close(fig_comb)

    # ==========================================
    # 2. 独立绘制 MSE vs Episode
    # ==========================================
    fig_mse, ax_mse_s = plt.subplots(figsize=figsize_single)
    
    ax_mse_s.plot(episodes, mse, marker="o", markersize=5, linewidth=1.5, 
                  color=color_mse, markeredgecolor="white", markeredgewidth=0.8)
    
    ax_mse_s.set_xlabel(r"回合数")
    ax_mse_s.set_ylabel(r"均方误差 (MSE)") 
    
    ax_mse_s.grid(True, linestyle=":", alpha=0.6, color="#CCCCCC")
    ax_mse_s.xaxis.set_major_locator(MultipleLocator(5))
    ax_mse_s.set_xlim(left=x_min - x_padding, right=x_max + x_padding)
    
    setup_sci_limits(ax_mse_s.yaxis)
    apply_times_new_roman_ticks(ax_mse_s)

    fig_mse.tight_layout()
    fig_mse.savefig(out_dir / "fig7_mse_vs_episode_subfig.pdf", format='pdf', bbox_inches="tight")
    fig_mse.savefig(out_dir / "fig7_mse_vs_episode_subfig.png", dpi=300, bbox_inches="tight")
    plt.close(fig_mse)

    # ==========================================
    # 3. 独立绘制 MAE vs Episode
    # ==========================================
    fig_mae, ax_mae_s = plt.subplots(figsize=figsize_single)
    
    ax_mae_s.plot(episodes, mae, marker="s", markersize=5, linewidth=1.5, 
                  color=color_mae, markeredgecolor="white", markeredgewidth=0.8)
    
    ax_mae_s.set_xlabel(r"回合数")
    ax_mae_s.set_ylabel(r"平均绝对误差(MAE)")
    
    ax_mae_s.grid(True, linestyle=":", alpha=0.6, color="#CCCCCC")
    ax_mae_s.xaxis.set_major_locator(MultipleLocator(5))
    ax_mae_s.set_xlim(left=x_min - x_padding, right=x_max + x_padding)
    
    setup_sci_limits(ax_mae_s.yaxis)
    apply_times_new_roman_ticks(ax_mae_s)

    fig_mae.tight_layout()
    fig_mae.savefig(out_dir / "fig7_mae_vs_episode_subfig.pdf", format='pdf', bbox_inches="tight")
    fig_mae.savefig(out_dir / "fig7_mae_vs_episode_subfig.png", dpi=300, bbox_inches="tight")
    plt.close(fig_mae)

    print(f"[Done] MSE & MAE 图表已按照顶刊标准保存至: {out_dir}")

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
    NATURE_BLUE = '#4DBBD5'
    NATURE_GREEN = '#E64B35'
    from matplotlib.ticker import ScalarFormatter
    setup_plot_style()
    
    # 横坐标步数乘以 10 转换为时间 (s)
    tr_time = np.arange(len(real["soc"])) * 10.0
    tg_time = np.arange(len(gp["soc"])) * 10.0
    
    # 电流提前取绝对值
    real_current_abs = np.abs(real["current"])
    gp_current_abs = np.abs(gp["current"])
    
    # ---------------------------------------------------------
    # 1. 绘制 2x2 汇总大图 (保持原有排版)
    # ---------------------------------------------------------
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes[0, 0].plot(tr_time, real_current_abs, color=NATURE_BLUE, label="Real Env", linewidth=0.5)
    axes[0, 0].plot(tg_time, gp_current_abs, "--", color=NATURE_GREEN, label="Static GP", linewidth=0.5)
    axes[0, 0].set_title("Absolute Current (A)")
    
    axes[0, 1].plot(tr_time, real["voltage"], color=NATURE_BLUE, linewidth=0.5)
    axes[0, 1].plot(tg_time, gp["voltage"], "--", color=NATURE_GREEN, linewidth=0.5)
    axes[0, 1].set_title("Voltage (V)")
    
    axes[1, 0].plot(tr_time, real["soc"], color=NATURE_BLUE, linewidth=0.5)
    axes[1, 0].plot(tg_time, gp["soc"], "--", color=NATURE_GREEN, linewidth=0.5)
    axes[1, 0].set_title("SOC")
    
    # 汇总图也顺便做一下温度转换 (K -> ℃)
    axes[1, 1].plot(tr_time, real["temperature"] - 273.15, color=NATURE_BLUE, linewidth=0.5)
    axes[1, 1].plot(tg_time, gp["temperature"] - 273.15, "--", color=NATURE_GREEN, linewidth=0.5)
    axes[1, 1].set_title("Temperature (℃)")
    
    for ax in axes.ravel():
        ax.set_xlabel("Time (s)")
    
    axes[0, 0].legend(frameon=True)
    plt.tight_layout()
    plt.savefig(path, dpi=220)
    plt.close(fig)

    # ---------------------------------------------------------
    # 2. 绘制符合顶刊规范的独立子图
    # ---------------------------------------------------------
    sub_size = (SUBPLOT_WIDTH_CM * CM_TO_INCH, SUBPLOT_HEIGHT_CM * CM_TO_INCH)
    
    # 提前计算温度的摄氏度转换
    real_temp_c = real["temperature"] - 273.15
    gp_temp_c = gp["temperature"] - 273.15
    
    series = [
        ("current", "充电电流 (A)", "绝对误差 (A)", "full_rollout_current_subfig.pdf", real_current_abs, gp_current_abs, "upper right"),
        ("voltage", "单体最大电压 (V)", "绝对误差 (V)", "full_rollout_voltage_subfig.pdf", real["voltage"], gp["voltage"], "center right"),
        ("soc", "电池荷电状态 (-)", "绝对误差", "full_rollout_soc_subfig.pdf", real["soc"], gp["soc"], "center right"),
        ("temperature", "最高温度 (℃)", "绝对误差 (℃)", "full_rollout_temperature_subfig.pdf", real_temp_c, gp_temp_c, "upper right"),
    ]

    for key, ylabel, err_label, fname, r_data, g_data, leg_loc in series:
        fig_sub, ax_sub = plt.subplots(figsize=sub_size)
        
        # --- 计算动态误差 ---
        g_data_interp = np.interp(tr_time, tg_time, g_data)
        abs_error = np.abs(g_data_interp - r_data)

        # --- 绘制主 Y 轴 (左侧：状态量) ---
        line1, = ax_sub.plot(tr_time, r_data, color=NATURE_BLUE, label="真实仿真环境", linewidth=1.0)
        line2, = ax_sub.plot(tg_time, g_data, "--", color=NATURE_GREEN, label="静态模型", linewidth=1.0)
        
        ax_sub.set_xlabel("时间 (s)")
        ax_sub.set_ylabel(ylabel)
        
        # --- 绘制副 Y 轴 (右侧：误差) ---
        ax2 = ax_sub.twinx()
        
        # 恢复你原来的灰色配色方案
        fill = ax2.fill_between(tr_time, 0, abs_error, color='#E0F0D4', alpha=0.4, label="绝对误差")
        ax2.plot(tr_time, abs_error, color='#2ca02c', linewidth=0.4, linestyle='-.', alpha=0.6)
        
        ax2.set_ylabel(err_label, color='#2ca02c')
        
        max_err = np.max(abs_error) if np.max(abs_error) > 0 else 1.0
        ax2.set_ylim(0, max_err * 2.5) 
        
        ax2.tick_params(axis='y', colors='#2ca02c')
        ax2.spines['right'].set_color('#2ca02c')
        
        # 强制右侧误差轴使用科学计数法，防止数字过长挤压图片宽度
        fmt = ScalarFormatter(useMathText=True)
        fmt.set_scientific(True)
        fmt.set_powerlimits((-2, 2))
        ax2.yaxis.set_major_formatter(fmt)

        # --- 合并图例并精细化设置 ---
        leg = ax_sub.legend(
            [line1, line2, fill],
            ["真实仿真环境", "静态模型", "绝对误差"],
            loc=leg_loc,
            frameon=True,
            facecolor='white',
            framealpha=0.9,
            edgecolor=(0.7, 0.7, 0.7, 0.5),
            borderpad=0.3,
            handletextpad=0.3,
            labelspacing=0.2
        )
        leg.get_frame().set_linewidth(0.5)
        
        # --- 核心修改：固定子图边距，解决图片大小不一的问题 ---
        # fig_sub.subplots_adjust(left=0.15, right=0.85, top=0.98, bottom=0.18)
        fig_sub.subplots_adjust(left=0.15, right=0.85, top=0.90, bottom=0.10)
        
        fig_sub.savefig(path.with_name(fname), format="pdf")
        plt.close(fig_sub)

# def plot_rollout_compare(real: Dict[str, np.ndarray], gp: Dict[str, np.ndarray], path: Path) -> None:
#     setup_plot_style()
    
#     # 【修改点 1】：横坐标步数乘以 10 转换为时间 (s)
#     tr_time = np.arange(len(real["soc"])) * 10.0
#     tg_time = np.arange(len(gp["soc"])) * 10.0
    
#     # 电流提前取绝对值
#     real_current_abs = np.abs(real["current"])
#     gp_current_abs = np.abs(gp["current"])
    
#     # ---------------------------------------------------------
#     # 1. 绘制 2x2 汇总大图 (保持原有粗排版，仅修改横轴为 Time)
#     # ---------------------------------------------------------
#     fig, axes = plt.subplots(2, 2, figsize=(12, 8))
#     axes[0, 0].plot(tr_time, real_current_abs, color=NATURE_BLUE, label="Real Env")
#     axes[0, 0].plot(tg_time, gp_current_abs, "--", color=NATURE_GREEN, label="Static GP")
#     axes[0, 0].set_title("Absolute Current (A)")
    
#     axes[0, 1].plot(tr_time, real["voltage"], color=NATURE_BLUE)
#     axes[0, 1].plot(tg_time, gp["voltage"], "--", color=NATURE_GREEN)
#     axes[0, 1].set_title("Voltage (V)")
    
#     axes[1, 0].plot(tr_time, real["soc"], color=NATURE_BLUE)
#     axes[1, 0].plot(tg_time, gp["soc"], "--", color=NATURE_GREEN)
#     axes[1, 0].set_title("SOC")
    
#     axes[1, 1].plot(tr_time, real["temperature"], color=NATURE_BLUE)
#     axes[1, 1].plot(tg_time, gp["temperature"], "--", color=NATURE_GREEN)
#     axes[1, 1].set_title("Temperature (K)")
    
#     for ax in axes.ravel():
#         ax.set_xlabel("Time (s)")
#         # ax.grid(True, color=GRID_COLOR, alpha=0.7)
    
#     # 汇总图的图例也加上边框
#     axes[0, 0].legend(frameon=True)
#     plt.tight_layout()
#     plt.savefig(path, dpi=220)
#     plt.close(fig)

#     # ---------------------------------------------------------
#     # 2. 绘制符合顶刊规范的独立子图 (应用 $\mathrm{}$ 与 Formatter)
#     # ---------------------------------------------------------
#     sub_size = (SUBPLOT_WIDTH_CM * CM_TO_INCH, SUBPLOT_HEIGHT_CM * CM_TO_INCH)
    
#     # 【修改点 2】：引入 \mathrm{} 格式化坐标轴标签
#     series = [
#         ("current", r"充电电流绝对值 $\mathrm{(A)}$", "full_rollout_current_subfig.pdf", real_current_abs, gp_current_abs),
#         ("voltage", r"端电压 $\mathrm{(V)}$", "full_rollout_voltage_subfig.pdf", real["voltage"], gp["voltage"]),
#         ("soc", r"电池荷电状态 $\mathrm{(-)}$", "full_rollout_soc_subfig.pdf", real["soc"], gp["soc"]),
#         ("temperature", r"最高温度 $\mathrm{(K)}$", "full_rollout_temperature_subfig.pdf", real["temperature"], gp["temperature"]),
#     ]

#     # 【修改点 3】：定义刻度数字格式化函数，强制使用 Times New Roman
#     def times_formatter(x, pos):
#         return f"$\\mathrm{{{x:g}}}$"

#     for key, ylabel, fname, r_data, g_data in series:
#         fig_sub, ax_sub = plt.subplots(figsize=sub_size)
        
#         # 绘制曲线
#         ax_sub.plot(tr_time, r_data, color=NATURE_BLUE, label="真实仿真环境", linewidth=1.6)
#         ax_sub.plot(tg_time, g_data, "--", color=NATURE_GREEN, label="静态代理模型", linewidth=1.6)
        
#         # 设置横纵标签
#         ax_sub.set_xlabel(r"时间 $\mathrm{(s)}$")
#         ax_sub.set_ylabel(ylabel)
        
#         # 网格设置
#         # ax_sub.grid(True, color=GRID_COLOR, alpha=0.7)
        
#         # 强制坐标轴刻度数字使用我们定义的 times_formatter
#         ax_sub.xaxis.set_major_formatter(ticker.FuncFormatter(times_formatter))
#         ax_sub.yaxis.set_major_formatter(ticker.FuncFormatter(times_formatter))
        
#         # 【修改点 4】：图例加上边框 (frameon=True)，并设置精美的半透明白底灰框
#         ax_sub.legend(
#             loc="best",
#             frameon=True,
#             facecolor='white',
#             framealpha=0.9,
#             edgecolor=(0.7, 0.7, 0.7, 0.5),
#             borderpad=0.3,
#             handletextpad=0.3,
#             labelspacing=0.2
#         )
        
#         fig_sub.tight_layout()
#         # 加入 bbox_inches='tight' 防止标签被裁切
#         fig_sub.savefig(path.with_name(fname), format="pdf", bbox_inches="tight")
#         plt.close(fig_sub)

def plot_efficiency(real_t: float, gp_t: float, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(["liionpack+PyBaMM", "Static GP"], [real_t, gp_t], color=[NATURE_BLUE, NATURE_GREEN])
    ax.set_ylabel("Wall-clock time (s)")
    ax.set_title("Single-Episode Efficiency")
    plt.tight_layout(); plt.savefig(path, dpi=220); plt.close(fig)

# 基于统计指标的交叉验证
def run_cross_validation_part(args: argparse.Namespace, cfg: dict) -> List[Dict[str, float]]:
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
    return cv_rows

# 完整环境仿真对比
def run_full_rollout_compare_part(args: argparse.Namespace, cfg: dict) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray], float, float]:
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
    return real, gp, rt, gt


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", type=Path, default=Path("dataset_with_episode.csv"))
    p.add_argument("--config", type=Path, default=Path("configs/pack_3p6s_spme.yaml"))
    p.add_argument("--agent-ckpt", type=Path, default=Path("runs/cycle0-td3/agent_ckpt.pt"))
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
    p.add_argument("--skip-cv", action="store_true", help="Skip cross-validation metrics and plots.")
    p.add_argument("--skip-rollout", action="store_true", help="Skip real-vs-surrogate full rollout comparison.")
    p.add_argument("--cv-metrics-csv", type=Path, default=None, help="Read episode_cv_metrics.csv and redraw fig7 directly.")
    p.add_argument("--plot-fig7-from-csv", action="store_true", help="Only redraw fig7 from --cv-metrics-csv and exit.")
    args = p.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.plot_fig7_from_csv:
        csv_path = args.cv_metrics_csv or (args.output_dir / "episode_cv_metrics.csv")
        if not csv_path.exists():
            raise FileNotFoundError(f"Required file not found: {csv_path}")
        cv_rows = load_episode_curve_csv(csv_path)
        plot_mse_mae_curve(cv_rows, args.output_dir)
        print(f"[Done] fig7 redrawn from csv: {csv_path}")
        return
    
    if not args.skip_cv and not args.dataset.exists():
        raise FileNotFoundError(f"Required file not found: {args.dataset}")
    if not args.skip_rollout:
        for req in [args.config, args.agent_ckpt, args.surrogate_ckpt]:
            if not req.exists():
                raise FileNotFoundError(f"Required file not found: {req}")

    cfg = load_config(str(args.config)).data
    set_global_seed(args.seed)

    cv_rows: List[Dict[str, float]] = []
    if not args.skip_cv:
        cv_rows = run_cross_validation_part(args, cfg)

    real = gp = None
    rt = gt = float("nan")
    if not args.skip_rollout:
        real, gp, rt, gt = run_full_rollout_compare_part(args, cfg)
    with (args.output_dir / "summary.txt").open("w", encoding="utf-8") as f:
        f.write("Static surrogate validity experiment summary (real env)\n")
        if cv_rows:
            f.write(f"Episode range = [{args.episode_start}, {args.episode_end}]\n")
            f.write(f"CV scheme = {args.k_folds}-fold repeated {args.cv_repeats} times\n")
            f.write(f"CV average R2 (curve mean) = {np.mean([r['r2'] for r in cv_rows]):.6f}\n")
            f.write(f"CV average MSE (curve mean) = {np.mean([r['mse'] for r in cv_rows]):.6e}\n")
            f.write(f"CV average MAE (curve mean) = {np.mean([r['mae'] for r in cv_rows]):.6e}\n")
        else:
            f.write("CV part skipped.\n")

        if real is not None and gp is not None:
            f.write(f"Single rollout final SOC (real) = {real['soc'][-1]:.6f}\n")
            f.write(f"Single rollout final SOC (gp) = {gp['soc'][-1]:.6f}\n")
            f.write(f"Runtime real env = {rt:.6f} s\n")
            f.write(f"Runtime static GP = {gt:.6f} s\n")
            f.write(f"Speedup (real/gp) = {rt / (gt + 1e-12):.2f}x\n")
        else:
            f.write("Full rollout comparison part skipped.\n")

    print(f"[Done] output_dir={args.output_dir}")


if __name__ == "__main__":
    main()