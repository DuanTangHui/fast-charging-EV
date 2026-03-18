"""实验2：25°C下，静态策略 vs 组合策略在老化1~50阶段的真实环境性能对比。"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
from matplotlib import font_manager
import os
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
        "charge_time_min": t_end / 60.0,  # 转换为分钟，与论文对齐
        "voltage_violation": float(vmax - v_max), # 允许负值，体现距边界的距离
        "temperature_violation": float(tcell - t_max),
    }

def plot_curves(rows: List[Dict[str, float]], out_dir: Path) -> None:
    import logging
    import os
    from matplotlib import font_manager
    import matplotlib.pyplot as plt

    # 1. 强制屏蔽字体内部元数据警告 (强迫症福音)
    logging.getLogger('fontTools.subset').level = logging.ERROR

    # 2. 加载宋体 (如果运行在 Linux/Mac 上，需确保路径下有该字体，或视情况略过)
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

    # 4. Nature 经典学术配色 (蓝色代表 Static, 红色代表 Adaptive)
    color_static = '#3C5488'
    color_adapt = '#E64B35'

    x = np.array([r["stage"] for r in rows])

    # 物理尺寸规范：2x2 矩阵子图，单图 7.5cm x 6.0cm，总尺寸约为 15cm x 12cm
    cm_to_inch = 1 / 2.54
    fig, axs = plt.subplots(2, 2, figsize=(15.0 * cm_to_inch, 12.0 * cm_to_inch))
    axs = axs.flatten()

    metrics = [
        ("total_reward", "Cumulative Return [-]"),
        ("charge_time_min", "Charge Time [min]"),
        ("voltage_violation", "Voltage Violation [V]"),
        ("temperature_violation", "Temperature Violation [°C]")
    ]

    for i, (base_key, ylabel) in enumerate(metrics):
        ax = axs[i]
        
        # 获取统计数据
        static_mean = np.array([r[f"static_{base_key}_mean"] for r in rows])
        static_min = np.array([r[f"static_{base_key}_min"] for r in rows])
        static_max = np.array([r[f"static_{base_key}_max"] for r in rows])
        
        comb_mean = np.array([r[f"combined_{base_key}_mean"] for r in rows])
        comb_min = np.array([r[f"combined_{base_key}_min"] for r in rows])
        comb_max = np.array([r[f"combined_{base_key}_max"] for r in rows])

        # 绘制 Static (实线 + 阴影带)
        ax.fill_between(x, static_min, static_max, color=color_static, alpha=0.2, zorder=2, edgecolor='none')
        ax.plot(x, static_mean, color=color_static, label="Static", zorder=3, linewidth=1.5)
        
        # 绘制 Adaptive (实线 + 阴影带)
        ax.fill_between(x, comb_min, comb_max, color=color_adapt, alpha=0.2, zorder=4, edgecolor='none')
        ax.plot(x, comb_mean, color=color_adapt, label="Adaptive", zorder=5, linewidth=1.5)

        # 设置轴标签（强制英文字体）
        ax.set_ylabel(ylabel, fontname='Times New Roman')
        if i >= 2:
            ax.set_xlabel("Cycle Number", fontname='Times New Roman')

        # 添加极淡的浅灰点划网格线
        ax.grid(True, linestyle=':', alpha=0.6, color='#CCCCCC')

        # 强制刻度数字为 Times New Roman，并缩小间距
        ax.tick_params(axis='x', pad=2, length=3)
        ax.tick_params(axis='y', pad=2, length=3)
        for label in ax.get_xticklabels() + ax.get_yticklabels():
            label.set_fontname('Times New Roman')

        # 仅在第一个子图中绘制内置图例 (根据数据走势，累计奖励放左下/右上均可)

def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/pack_3p6s_spme_with_soh_prior.yaml")
    p.add_argument("--runs-dir", default="runs")
    p.add_argument("--start-stage", type=int, default=1)
    p.add_argument("--end-stage", type=int, default=50)
    p.add_argument("--seed", type=int, default=123)
    p.add_argument("--eval-episodes", type=int, default=5) # 新增参数：每个stage测试次数
    p.add_argument("--output-dir", default="runs/adaptive_experiment/result2")
    args = p.parse_args()

    cfg = load_config(args.config).data
    env = build_pack_env(cfg["env"])
    reward_cfg = PaperRewardConfig(**cfg["reward"])

    runs_dir = Path(args.runs_dir)
    static_ckpt = runs_dir / "cycle00" / "agent_ckpt.pt"
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

        # 用于存储单次循环内不同初始条件的多次评估结果
        stage_metrics = {
            "static": {"total_reward": [], "charge_time_min": [], "voltage_violation": [], "temperature_violation": []},
            "combined": {"total_reward": [], "charge_time_min": [], "voltage_violation": [], "temperature_violation": []}
        }

        # 循环多次评估以产生最大值/最小值区间
        for i in range(args.eval_episodes):
            eval_seed = args.seed + stage * 100 + i
            
            # Static Policy 评估
            set_global_seed(eval_seed)
            r_static, infos_static = rollout_env(env, static_policy, reward_cfg, reset_seed=eval_seed)
            s_static = episode_stats(infos_static, env.v_max, env.t_max)
            
            stage_metrics["static"]["total_reward"].append(r_static)
            for k, v in s_static.items():
                stage_metrics["static"][k].append(v)
            
            # Combined Policy 评估
            set_global_seed(eval_seed)
            r_comb, infos_comb = rollout_env(env, combined_policy, reward_cfg, reset_seed=eval_seed)
            s_comb = episode_stats(infos_comb, env.v_max, env.t_max)
            
            stage_metrics["combined"]["total_reward"].append(r_comb)
            for k, v in s_comb.items():
                stage_metrics["combined"][k].append(v)

        # 计算并保存统计量 (mean, min, max)
        row = {"stage": float(stage)}
        for policy_name in ["static", "combined"]:
            for metric_name in stage_metrics[policy_name]:
                data = stage_metrics[policy_name][metric_name]
                row[f"{policy_name}_{metric_name}_mean"] = float(np.mean(data))
                row[f"{policy_name}_{metric_name}_min"] = float(np.min(data))
                row[f"{policy_name}_{metric_name}_max"] = float(np.max(data))
        
        rows.append(row)
        print(f"[EXP2] stage={stage}, R_static(mean)={row['static_total_reward_mean']:.2f}, R_comb(mean)={row['combined_total_reward_mean']:.2f}")

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