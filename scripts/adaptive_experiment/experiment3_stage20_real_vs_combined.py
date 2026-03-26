"""实验3：第10老化阶段，真实物理仿真训练(630 ep) vs 组合代理方法对比。"""
from __future__ import annotations

import argparse
import csv
import sys
import time
import logging       # 新增
import traceback     # 新增
import datetime      # 新增
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import torch
import os
import logging
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager
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


# 1. 强制屏蔽字体内部元数据警告
logging.getLogger('fontTools.subset').level = logging.ERROR

# 2. 加载宋体 (针对 Windows 系统，Linux/Mac 需替换为对应 SimSun 路径)
simsun_path = r'C:\Windows\Fonts\simsun.ttc'
if os.path.exists(simsun_path):
    font_manager.fontManager.addfont(simsun_path)

# 3. 全局标准配置
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


# ==========================================
# 规范化后的画图函数
# ==========================================

def plot_curves(real_curve: Dict[str, List[float]], comb_curve: Dict[str, List[float]], out_png: Path) -> None:
    # Nature 经典学术配色
    colors_nature = ['#E64B35', '#4DBBD5']

    # 物理尺寸换算 (厘米 -> 英寸)
    # 需求：2×2 矩阵，单图 7.5cm × 6.0cm
    # 加上子图间的留白间距，整图画布尺寸设定为 16.0cm × 13.5cm 左右最合适
    cm_to_inch = 1 / 2.54
    fig, axs = plt.subplots(2, 2, figsize=(16.0 * cm_to_inch, 13.5 * cm_to_inch))

    # 定义子图配置列表，方便遍历 (轴对象, 数据键名, 标题名)
    plots_config = [
        (axs[0, 0], "current", "Current [A]"),
        (axs[0, 1], "voltage", "Voltage [V]"),
        (axs[1, 0], "soc", "SOC"),
        (axs[1, 1], "temperature", "Temperature [°C]")
    ]

    for ax, key, title in plots_config:
        # 绘制曲线
        ax.plot(real_curve["time"], real_curve[key], label="Real-Env Trained", color=colors_nature[0], linewidth=1.2)
        ax.plot(comb_curve["time"], comb_curve[key], label="Combined-Surrogate Trained", color=colors_nature[1], linewidth=1.2)
        
        # 强制英文字符使用 Times New Roman (若含中文可去掉 fontname 限制让其自动 fallback)
        ax.set_title(title, fontname='Times New Roman', fontsize=10.5)
        ax.set_xlabel("Time [s]", fontname='Times New Roman', fontsize=10.5)
        
        # 网格线：极淡的浅灰点划线
        ax.grid(True, linestyle=':', alpha=0.6, color='#CCCCCC')
        
        # 图例规范：白底半透明，极浅边框，缩紧间距
        ax.legend(
            loc='best', 
            ncol=1,                    
            frameon=True,              
            facecolor='white',         
            framealpha=0.9,            
            edgecolor=(0.7, 0.7, 0.7, 0.5),
            borderpad=0.4,       
            handletextpad=0.3,   
            labelspacing=0.4,
            prop={'family': 'Times New Roman', 'size': 9} # 强制图例文字字体
        )

    # --- 收尾导出标准 ---
    
    for ax in fig.axes:
        # 缩小刻度线与文字的间距
        ax.tick_params(axis='x', pad=2, length=3)
        ax.tick_params(axis='y', pad=2, length=3)
        
        # 强制坐标轴的刻度数字为 Times New Roman
        for label in ax.get_xticklabels() + ax.get_yticklabels():
            label.set_fontname('Times New Roman')

    # 对齐所有子图的 Y 轴标签
    fig.align_ylabels()
    plt.tight_layout()

    # 1. 导出高分辨率 PNG 用于快速预览
    plt.savefig(out_png, dpi=300, bbox_inches='tight')
    
    # 2. 导出同名矢量 PDF 用于论文插入 (满足 PDF 字体全嵌入需求)
    out_pdf = out_png.with_suffix('.pdf')
    plt.savefig(out_pdf, format='pdf', bbox_inches='tight')
    
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/pack_3p6s_spme_with_soh_prior.yaml")
    p.add_argument("--runs-dir", default="runs")
    p.add_argument("--stage", type=int, default=10)
    p.add_argument("--real-train-episodes", type=int, default=630)
    p.add_argument("--output-dir", default="runs/adaptive_experiment/result3")
    args = p.parse_args()

    # --- 1. 创建输出目录与日志配置 ---
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = out_dir / f"experiment3_stage{args.stage}_{timestamp}.log"
    
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
        
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            logging.StreamHandler(sys.stdout)
        ]
    )

    logging.info("==== 实验3开始 ====")
    logging.info(f"配置参数: {vars(args)}")

    cfg = load_config(args.config).data
    set_global_seed(cfg.get("seed", 42))

    env = build_pack_env(cfg["env"])
    env.set_aging_stage(args.stage)
    reward_cfg = PaperRewardConfig(**cfg["reward"])

    runs = Path(args.runs_dir)
    stage_agent_ckpt = runs / "adaptive" / f"adaptive_cycle{args.stage}" / "agent_ckpt.pt"
    static_ckpt = runs / "cycle0" / "static_surrogate.pt"
    diff_ckpt = runs / "adaptive" / f"adaptive_cycle{args.stage}" / "diff_surrogate.pt"
    
    if not stage_agent_ckpt.exists():
        logging.error(f"Missing adaptive stage agent: {stage_agent_ckpt}")
        sys.exit(1)
    if not static_ckpt.exists() or not diff_ckpt.exists():
        logging.error("Missing static/diff surrogate ckpt for stage comparison")
        sys.exit(1)

    # A) 真实物理环境训练 630 episode
    real_agent = build_agent_from_config(env.observation_space.shape[0], 1, cfg["rl"])
    real_agent.load(str(stage_agent_ckpt), map_location="cpu")

    logging.info(f"开始在真实物理环境中进行训练，总计 {args.real_train_episodes} 个 episodes...")
    t0 = time.perf_counter()
    
    # --- 2. 捕获训练阶段崩溃 ---
    try:
        train_metrics, _ = run_real_training_collect_style(
            env,
            real_agent,
            reward_cfg,
            episodes=args.real_train_episodes,
        )
    except Exception as e:
        logging.error("!!! CASADI 求解器在 [训练阶段] 发生崩溃 !!!")
        logging.error(f"详细报错信息:\n{traceback.format_exc()}")
        logging.info("请检查是否是由于探索初期的极端动作导致的物理越界。程序已安全中断。")
        sys.exit(1)
        
    train_sec = time.perf_counter() - t0
    real_agent_ckpt = out_dir / f"real_env_stage{args.stage}_ep{args.real_train_episodes}_agent_ckpt.pt"
    real_agent.save(str(real_agent_ckpt))
    logging.info(f"训练完成，耗时 {train_sec:.2f} 秒。模型已保存至 {real_agent_ckpt}")

    # B) 组合代理方法
    comb_agent = build_agent_from_config(env.observation_space.shape[0], 1, cfg["rl"])
    comb_agent.load(str(stage_agent_ckpt), map_location="cpu")
    comb_agent.actor.eval()

    static_surrogate = torch.load(static_ckpt, map_location="cpu", weights_only=False)
    diff_surrogate = torch.load(diff_ckpt, map_location="cpu", weights_only=False)
    combined = CombinedSurrogate(static_surrogate, diff_surrogate)

    low = float(cfg["rl"]["action_low"])
    high = float(cfg["rl"]["action_high"])

    # C) 完整充电模拟与耗时统计
    logging.info("开始进行策略评估 (Rollout)...")
    
    # --- 3. 捕获评估阶段崩溃 ---
    try:
        t1 = time.perf_counter()
        r_real, infos_real = rollout_env(env, guarded_policy(real_agent, low, high), reward_cfg)
        sim_real_sec = time.perf_counter() - t1

        state0, _ = env.reset(seed=777)
        t2 = time.perf_counter()
        r_comb, infos_comb = rollout_env(env, guarded_policy(comb_agent, low, high), reward_cfg)
        sim_comb_sec = time.perf_counter() - t2
        
    except Exception as e:
        logging.error("!!! CASADI 求解器在 [测试评估阶段] 发生崩溃 !!!")
        logging.error(f"详细报错信息:\n{traceback.format_exc()}")
        logging.info("环境可能已处于奇点状态，程序已安全中断。")
        sys.exit(1)

    # D) 记录与绘图
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

    logging.info(f"==== 实验成功结束 ==== 数据已保存至 {out_dir}")


if __name__ == "__main__":
    main()