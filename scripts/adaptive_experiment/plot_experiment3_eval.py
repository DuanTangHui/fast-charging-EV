"""
仅用于实验3的评估与绘图 (加载已训练好的真实物理仿真模型 vs 组合代理模型)
支持输出符合顶刊规范的 PDF/PNG 矢量图表。
"""
from __future__ import annotations

import argparse
import csv
import sys
import time
import logging
import traceback
import datetime
import os
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np

from matplotlib import font_manager

CURRENT_DIR = Path(__file__).resolve().parent
ROOT_DIR = CURRENT_DIR.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from src.envs.liionpack_spme_pack_env import build_pack_env
from src.evaluation.episode_rollout import rollout_env
from src.rewards.paper_reward import PaperRewardConfig
from src.rl.agent_factory import build_agent_from_config
from src.surrogate.gp_combined import CombinedSurrogate
from src.utils.config import load_config
from src.utils.seeds import set_global_seed

# ==========================================
# 顶刊级全局图表环境配置
# ==========================================

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

# ==========================================
# 辅助函数
# ==========================================
def guarded_policy(agent, low: float, high: float):
    def _policy(state: np.ndarray) -> np.ndarray:
        a = float(agent.act(state)[0])
        return np.array([float(np.clip(a, low, high))], dtype=np.float32)
    return _policy

def infos_to_curve(infos: List[Dict]) -> Dict[str, List[float]]:
    return {
        "time": [float(i.get("t", 0.0)) for i in infos],
        "current": [abs(float(i.get("I_pack_true", i.get("I", 0.0)))) for i in infos],
        "voltage": [float(i.get("V_cell_max", 0.0)) for i in infos],
        "soc": [float(i.get("SOC_pack", 0.0)) for i in infos],
        "temperature": [float(i.get("T_cell_max", 0.0)) for i in infos],
    }

def plot_curves_separate(real_curve: Dict[str, List[float]], comb_curve: Dict[str, List[float]], out_dir: Path, prefix: str) -> None:
    """
    采用双 Y 轴方案：左轴展示状态量对比，右轴用浅色阴影展示绝对误差。
    尺寸维持单图规范: 8.0cm x 5.5cm，并严格固定子图边距。
    """
    colors_nature = [ '#4DBBD5','#E64B35', '#E0F0D4'] # 红，蓝，浅灰（用于误差阴影）
    
    cm_to_inch = 1 / 2.54
    figsize_single = (8.0 * cm_to_inch, 5.5 * cm_to_inch)

    t_real = np.array(real_curve["time"])
    t_comb = np.array(comb_curve["time"])

    # 配置：键名，左Y轴标题，右Y轴误差标题，文件后缀，图例位置
    plots_config = [
        ("current", "充电电流 (A)", "绝对误差 (A)", "current", "upper right"),
        ("voltage", "单体最大电压 (V)", "绝对误差 (V)", "voltage", "center right"), 
        ("soc", "电池荷电状态 (-)", "绝对误差", "soc", "center right"),                                        
        ("temperature", "最高温度 (℃)", "绝对误差 (℃)", "temp", "upper right")  
    ] 

    for key, title, err_title, suffix, leg_loc in plots_config:
        fig, ax1 = plt.subplots(figsize=figsize_single)
        
        y_real = np.array(real_curve[key])
        y_comb = np.array(comb_curve[key])

        # 如果是温度，将开尔文 (K) 转换为摄氏度 (°C)
        if key == "temperature":
            y_real = y_real - 273.15
            y_comb = y_comb - 273.15

        # 插值计算绝对误差 (Absolute Error)
        y_comb_interp = np.interp(t_real, t_comb, y_comb)
        abs_error = np.abs(y_comb_interp - y_real)

        # --- 绘制主 Y 轴 (左侧：状态量) ---
        line1, = ax1.plot(t_real, y_real, label="真实仿真环境", color=colors_nature[0], linewidth=1.0)
        line2, = ax1.plot(t_comb, y_comb, label="老化衰减模型", color=colors_nature[1], linewidth=1.0)
        
        ax1.set_ylabel(title, fontsize=10.5)
        ax1.set_xlabel("时间（s）", fontsize=10.5)

        # --- 绘制副 Y 轴 (右侧：误差) ---
        ax2 = ax1.twinx()
        # 用浅灰色填充底部作为误差展示，alpha控制透明度
        fill = ax2.fill_between(t_real, 0, abs_error, color=colors_nature[2], alpha=0.4, label="绝对误差")
        # 极细的虚线勾勒误差边缘
        ax2.plot(t_real, abs_error, color='#2ca02c', linewidth=0.3, linestyle='-.', alpha=0.6)
        if key == "current":
            # 增加 labelpad 把标题往外推
            ax2.set_ylabel(err_title, fontsize=9, color='#2ca02c', labelpad=10) 
            # 增加 pad 把刻度数字往外推
            ax2.tick_params(axis='y', pad=4, length=3, colors='#2ca02c')
        else:
            # 其他图保持原样（默认 labelpad 约等于 4）
            ax2.set_ylabel(err_title, fontsize=9, color='#2ca02c')
            ax2.tick_params(axis='y', colors='#2ca02c')
        # ax2.set_ylabel(err_title, fontsize=9, color='gray')
        
        # 动态调整右侧 Y 轴的上限
        max_err = np.max(abs_error) if np.max(abs_error) > 0 else 1.0
        ax2.set_ylim(0, max_err * 2.5) 
        
        # 修改右侧刻度颜色
        # ax2.tick_params(axis='y', colors='gray')
        ax2.spines['right'].set_color('#2ca02c')

        # --- 合并图例 ---
        leg = ax1.legend(
            [line1, line2, fill], 
            ["真实仿真环境", "老化衰减模型", "绝对误差"],
            loc=leg_loc, 
            ncol=1,                    
            frameon=True,              
            facecolor='white',         
            framealpha=0.9,            
            edgecolor=(0.7, 0.7, 0.7, 0.5),
            borderpad=0.3,       
            handletextpad=0.3,   
            labelspacing=0.3,
            fontsize=9
        )
        leg.get_frame().set_linewidth(0.5)

        # --- 刻度与字体规范化 ---
        for ax in [ax1, ax2]:
            ax.tick_params(axis='x', pad=2, length=3)
            ax.tick_params(axis='y', pad=2, length=3)
            for label in ax.get_xticklabels() + ax.get_yticklabels():
                label.set_fontname('Times New Roman')

        # === 核心修改区 ===
        # 1. 移除 plt.tight_layout()，替换为手动固定边距
        # fig.subplots_adjust(left=0.18, right=0.82, top=0.92, bottom=0.18)
        fig.subplots_adjust(left=0.15, right=0.85, top=0.92, bottom=0.18)
        # 导出文件
        out_png = out_dir / f"{prefix}_{suffix}.png"
        out_pdf = out_dir / f"{prefix}_{suffix}.pdf"
        
        # 2. 移除 bbox_inches='tight'，以保证画布严格遵守 figsize 和 subplots_adjust 的设定
        
       
        plt.savefig(out_png, dpi=300)
        plt.savefig(out_pdf, format='pdf')
        plt.close(fig)
        
        logging.info(f"成功保存双Y轴叠图 (固定边距): {out_png.name} (含同名 PDF)")
# def plot_curves_separate(real_curve: Dict[str, List[float]], comb_curve: Dict[str, List[float]], out_dir: Path, prefix: str) -> None:
#     """
#     将四个状态量分别绘制并保存为独立的高质量图片。
#     尺寸严格遵循单图规范: 8.0cm x 5.5cm，横纵坐标及图例已中文化。
#     """
#     # Nature 经典学术配色
#     colors_nature = ['#E64B35', '#4DBBD5']
    
#     # 物理尺寸换算 (厘米 -> 英寸)
#     cm_to_inch = 1 / 2.54
#     figsize_single = (8.0 * cm_to_inch, 5.5 * cm_to_inch)

#     # 定义要绘制的键名、中文化Y轴标题 (英文/单位使用 \mathrm 包裹以调用 Times New Roman) 和保存的文件名后缀
#     plots_config = [
#         ("current", "电流（$\mathrm{A}$）", "current"),
#         ("voltage", "单体最大电压（$\mathrm{V}$）", "voltage"),
#         ("soc", "荷电状态", "soc"),
#         ("temperature", "最高温度（$\mathrm{K}$）", "temp")
#     ]

#     for key, title, suffix in plots_config:
#         # 每次循环新建一个独立画布
#         fig, ax = plt.subplots(figsize=figsize_single)
        
#         # 绘制曲线，修改图例为中文
#         ax.plot(real_curve["time"], real_curve[key], label="真实仿真环境", color=colors_nature[0], linewidth=1.2)
#         ax.plot(comb_curve["time"], comb_curve[key], label="组合代理模型", color=colors_nature[1], linewidth=1.2)
        
#         # 标题与标签 (去掉 fontname 限制，让系统自动使用宋体，其中 \mathrm 部分会自动调用 Times New Roman)
#         ax.set_ylabel(title, fontsize=10.5)
#         ax.set_xlabel("时间（$\mathrm{s}$）", fontsize=10.5)
        
#         # 网格线：极淡的浅灰点划线
#         # ax.grid(True, linestyle=':', alpha=0.6, color='#CCCCCC')
        
#         # 图例规范：白底半透明，极浅边框 (去掉 prop={'family':...} 限制，直接用 fontsize)
#         ax.legend(
#             loc='best', 
#             ncol=1,                    
#             frameon=True,              
#             facecolor='white',         
#             framealpha=0.9,            
#             edgecolor=(0.7, 0.7, 0.7, 0.5),
#             borderpad=0.3,       
#             handletextpad=0.3,   
#             labelspacing=0.3,
#             fontsize=9
#         )

#         # --- 收尾导出标准 ---
#         # 缩小刻度线与文字的间距
#         ax.tick_params(axis='x', pad=2, length=3)
#         ax.tick_params(axis='y', pad=2, length=3)
        
#         # 坐标轴的刻度纯数字，直接强制遍历修改为 Times New Roman
#         for label in ax.get_xticklabels() + ax.get_yticklabels():
#             label.set_fontname('Times New Roman')

#         plt.tight_layout()

#         # 生成独立的文件路径
#         out_png = out_dir / f"{prefix}_{suffix}.png"
#         out_pdf = out_dir / f"{prefix}_{suffix}.pdf"
        
#         # 导出双格式
#         plt.savefig(out_png, dpi=300, bbox_inches='tight')
#         plt.savefig(out_pdf, format='pdf', bbox_inches='tight')
        
#         plt.close(fig)
#         logging.info(f"成功保存单图: {out_png.name} (含同名 PDF)")
# ==========================================
# 主执行流程 (纯评估模式)
# ==========================================
def main() -> None:
    p = argparse.ArgumentParser(description="Evaluate and plot pre-trained Real Env vs Combined Surrogate models.")
    p.add_argument("--config", default="configs/pack_3p6s_spme_with_soh_prior.yaml")
    p.add_argument("--runs-dir", default="runs")
    p.add_argument("--stage", type=int, default=20, help="Aging stage to evaluate (e.g., 20)")
    
    # 新增参数：直接指定你训练好的 Real Env 模型路径
    p.add_argument(
        "--real-ckpt", 
        default="runs/adaptive_experiment/result3/real_env_stage20_ep630_agent_ckpt.pt",
        help="Path to the trained real-env agent checkpoint."
    )
    p.add_argument("--output-dir", default="runs/adaptive_experiment/result3/eval_results")
    args = p.parse_args()

    # --- 1. 创建输出目录与日志 ---
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
        
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        handlers=[logging.StreamHandler(sys.stdout)]
    )

    logging.info("==== 实验3: 离线模型评估与绘图模式开始 ====")
    
    cfg = load_config(args.config).data
    set_global_seed(cfg.get("seed", 42))

    env = build_pack_env(cfg["env"])
    env.set_aging_stage(args.stage)
    reward_cfg = PaperRewardConfig(**cfg["reward"])

    # 路径配置
    runs = Path(args.runs_dir)
    real_ckpt_path = Path(args.real_ckpt)
    stage_agent_ckpt = runs / "adaptive" / f"adaptive_cycle{args.stage}" / "agent_ckpt.pt"
    static_ckpt = runs / "cycle0" / "static_surrogate.pt"
    diff_ckpt = runs / "adaptive" / f"adaptive_cycle{args.stage}" / "diff_surrogate.pt"
    
    # 检查所有需要的文件是否存在
    missing_files = []
    if not real_ckpt_path.exists(): missing_files.append(real_ckpt_path)
    if not stage_agent_ckpt.exists(): missing_files.append(stage_agent_ckpt)
    if not static_ckpt.exists(): missing_files.append(static_ckpt)
    if not diff_ckpt.exists(): missing_files.append(diff_ckpt)
    
    if missing_files:
        logging.error("以下模型文件缺失，请检查路径:")
        for mf in missing_files:
            logging.error(f"  - {mf}")
        sys.exit(1)

    low = float(cfg["rl"]["action_low"])
    high = float(cfg["rl"]["action_high"])

    # --- 2. 加载真实环境模型 (直接评估) ---
    logging.info(f"加载真实物理环境训练出的模型: {real_ckpt_path}")
    real_agent = build_agent_from_config(env.observation_space.shape[0], 1, cfg["rl"])
    real_agent.load(str(real_ckpt_path), map_location="cpu")
    real_agent.actor.eval() # 关键：关闭探索噪声

    # --- 3. 加载组合代理模型 ---
    logging.info("加载组合代理模型 (Combined Surrogate)...")
    comb_agent = build_agent_from_config(env.observation_space.shape[0], 1, cfg["rl"])
    comb_agent.load(str(stage_agent_ckpt), map_location="cpu")
    comb_agent.actor.eval()
    # 注: 原代码构建了 CombinedSurrogate 但并未在 rollout_env 中显式传递它，
    # 这是因为 rollout_env 使用的是真实环境，代理模型的作用体现在 agent_ckpt 里的策略上。
    
    # --- 4. 执行测试 Rollout ---
    logging.info("开始进行策略评估 (Rollout)...")
    try:
        logging.info("--> 运行 Real-Env 策略...")
        t1 = time.perf_counter()
        r_real, infos_real = rollout_env(env, guarded_policy(real_agent, low, high), reward_cfg)
        sim_real_sec = time.perf_counter() - t1

        logging.info("--> 运行 Combined-Surrogate 策略...")
        state0, _ = env.reset(seed=777)
        t2 = time.perf_counter()
        r_comb, infos_comb = rollout_env(env, guarded_policy(comb_agent, low, high), reward_cfg)
        sim_comb_sec = time.perf_counter() - t2
        
    except Exception as e:
        logging.error("!!! CASADI 求解器在测试评估阶段发生崩溃 !!!")
        logging.error(traceback.format_exc())
        sys.exit(1)

    # --- 5. 提取数据与出图 ---
    real_curve = infos_to_curve(infos_real)
    comb_curve = infos_to_curve(infos_comb)
    
    # 修改为文件前缀模式
    file_prefix = f"eval_stage{args.stage}_real_vs_combined"
    logging.info(f"评估完成！开始分别绘制独立对比图...")
    plot_curves_separate(real_curve, comb_curve, out_dir, file_prefix)

    # 保存评估摘要
    summary = {
        "stage": args.stage,
        "eval_model": real_ckpt_path.name,
        "real_rollout_walltime_s": sim_real_sec,
        "combined_rollout_walltime_s": sim_comb_sec,
        "real_total_reward": float(r_real),
        "combined_total_reward": float(r_comb),
        "real_charge_time_s": float(real_curve["time"][-1] if real_curve["time"] else 0.0),
        "combined_charge_time_s": float(comb_curve["time"][-1] if comb_curve["time"] else 0.0),
    }

    csv_path = out_dir / f"eval_stage{args.stage}_summary.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(summary.keys()))
        w.writeheader()
        w.writerow(summary)

    # ================= 修改日志打印 =================
    logging.info(f"==== 绘图完成！====")
    logging.info(f"4组独立图表 (PNG+PDF) 已保存至目录: {out_dir}")
    logging.info(f"文件命名前缀为: {file_prefix}_*.png/pdf")
    logging.info(f"评估数据已保存至: {csv_path}")

if __name__ == "__main__":
    main()