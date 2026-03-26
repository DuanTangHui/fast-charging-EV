"""
仅用于实验2的离线绘图 (加载现有的 CSV 数据，绘制带误差阴影的单独曲线图)
尺寸严格遵循单图规范: 8.0cm x 5.5cm，横纵坐标及图例均已中文化，且单位按要求更新。
"""
import argparse
import csv
import os
import sys
import logging
from pathlib import Path
from typing import Dict, List

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import font_manager

# ==========================================
# 顶刊级全局图表环境配置
# ==========================================
logging.getLogger('fontTools.subset').level = logging.ERROR

# 加载宋体 (Windows 默认路径)
simsun_path = r'C:\Windows\Fonts\simsun.ttc'
if os.path.exists(simsun_path):
    font_manager.fontManager.addfont(simsun_path)

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["SimSun"], 
    "mathtext.fontset": "custom",
    "mathtext.rm": "Times New Roman",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "axes.unicode_minus": False,
    "axes.spines.top": True,
    "axes.spines.right": True,
    "axes.linewidth": 1.0,
    "xtick.major.width": 1.0,
    "ytick.major.width": 1.0,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "font.size": 10.5,                     
    "axes.labelsize": 10.5,
    "xtick.labelsize": 10.5,
    "ytick.labelsize": 10.5,
    "legend.fontsize": 9  
})

def plot_exp2_separate(data_rows: List[Dict], out_dir: Path, prefix: str) -> None:
    color_static = '#3C5488'
    color_adapt = '#E64B35'
    
    cm_to_inch = 1 / 2.54
    figsize_single = (8.0 * cm_to_inch, 5.5 * cm_to_inch)

    x = np.array([r["stage"] for r in data_rows])

    # 定义配置：(CSV键名前缀, Y轴标签, 文件后缀, 乘数换算)
    # 注意：充电时间从 min 换算回 s，需要乘以 60.0
    plots_config = [
        ("total_reward", "累计回报（$\mathrm{-}$）", "reward", 1.0),
        ("charge_time_min", "充电时间 （$\mathrm{s}$）", "charge_time", 60.0),
        ("voltage_violation", "电压越限 （$\mathrm{V}$）", "voltage_viol", 1.0),
        ("temperature_violation", "温度越限（$\mathrm{K}$）", "temp_viol", 1.0)
    ]

    for base_key, ylabel, suffix, multiplier in plots_config:
        fig, ax = plt.subplots(figsize=figsize_single)
        
        # 提取数据并应用换算乘数 (主要是把 min 变回 s)
        static_mean = np.array([r[f"static_{base_key}_mean"] for r in data_rows]) * multiplier
        static_min = np.array([r[f"static_{base_key}_min"] for r in data_rows]) * multiplier
        static_max = np.array([r[f"static_{base_key}_max"] for r in data_rows]) * multiplier
        
        comb_mean = np.array([r[f"combined_{base_key}_mean"] for r in data_rows]) * multiplier
        comb_min = np.array([r[f"combined_{base_key}_min"] for r in data_rows]) * multiplier
        comb_max = np.array([r[f"combined_{base_key}_max"] for r in data_rows]) * multiplier

        # 绘制静态策略
        ax.fill_between(x, static_min, static_max, color=color_static, alpha=0.2, zorder=2, edgecolor='none')
        ax.plot(x, static_mean, color=color_static, label="静态策略", zorder=3, linewidth=1.5)
        
        # 绘制组合策略
        ax.fill_between(x, comb_min, comb_max, color=color_adapt, alpha=0.2, zorder=4, edgecolor='none')
        ax.plot(x, comb_mean, color=color_adapt, label="组合策略", zorder=5, linewidth=1.5)

        # 轴标签 
        ax.set_ylabel(ylabel, fontsize=10.5)
        ax.set_xlabel("循环圈数", fontsize=10.5)

        # 网格与图例
        ax.grid(True, linestyle=':', alpha=0.6, color='#CCCCCC')
        ax.legend(
            loc='best', 
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

        # 收尾导出标准
        ax.tick_params(axis='x', pad=2, length=3)
        ax.tick_params(axis='y', pad=2, length=3)
        
        for label in ax.get_xticklabels() + ax.get_yticklabels():
            label.set_fontname('Times New Roman')

        plt.tight_layout()

        out_png = out_dir / f"{prefix}_{suffix}.png"
        out_pdf = out_dir / f"{prefix}_{suffix}.pdf"
        
        plt.savefig(out_png, dpi=300, bbox_inches='tight')
        plt.savefig(out_pdf, format='pdf', bbox_inches='tight')
        
        plt.close(fig)
        logging.info(f"成功保存单图: {out_png.name} (含同名 PDF)")


def main():
    p = argparse.ArgumentParser(description="Load Experiment 2 CSV and plot separate standardized figures.")
    p.add_argument("--csv-path", default="runs/adaptive_experiment/result2/exp2_stage1_50_metrics.csv")
    p.add_argument("--output-dir", default="runs/adaptive_experiment/result2/eval_plots")
    args = p.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
    
    csv_path = Path(args.csv_path)
    if not csv_path.exists():
        logging.error(f"找不到数据文件: {csv_path}")
        sys.exit(1)

    logging.info(f"正在加载数据: {csv_path}")
    data_rows = []
    with csv_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            parsed_row = {k: float(v) for k, v in row.items()}
            data_rows.append(parsed_row)

    if not data_rows:
        logging.error("CSV 文件为空！")
        sys.exit(1)

    file_prefix = "exp2_stage1_50"
    logging.info("开始分别绘制独立的老化对比图...")
    plot_exp2_separate(data_rows, out_dir, file_prefix)
    
    logging.info("==== 绘图全部完成！====")

if __name__ == "__main__":
    main()