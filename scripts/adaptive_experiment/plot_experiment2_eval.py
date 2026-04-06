"""
仅用于实验2的离线绘图 (加载现有的 CSV 数据，绘制标准化曲线图)
尺寸: 8.0cm x 5.5cm，线宽: 0.5，仅温度图保留图例。
# 自定义放大镜：回报(左下)、时间(右下)、电压(右下)。
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


plt.rcParams.update({
   'font.family': ['Times New Roman', 'SimSun'],  # 关键：先英文字体，再中文
    'axes.unicode_minus': False,

    # 边框与刻度
    "axes.spines.top": True,
    "axes.spines.right": True,
    "axes.linewidth": 0.5,
    "xtick.major.width": 0.5,
    "ytick.major.width": 0.5,
    "xtick.direction": "in",
    "ytick.direction": "in",
    
    # 严格的 9pt 字号体系
    "font.size": 9,                     
    "axes.labelsize": 9,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9  
})

def plot_exp2_separate(data_rows: List[Dict], out_dir: Path, prefix: str) -> None:
    color_static = '#3C5488'
    color_adapt = '#E64B35'
    color_drift = '#7FB069' # 对应图例中绿色的参数漂移线
    
    cm_to_inch = 1 / 2.54
    figsize_single = (9.0 * cm_to_inch, 5.5 * cm_to_inch)

    # 提取横坐标：老化阶段 k (Cycle Number)
    x = np.array([r["stage"] for r in data_rows])
    
    # ==========================================
    # 核心修改：直接根据公式计算附加极化阻抗
    # R_contact = R_sei,0 + gamma_sei * k
    # ==========================================
    R_sei_0 = 0.005   # 初始附加阻抗 5 mΩ
    gamma_sei = 0.002 # 随阶段增长系数 2 mΩ
    resistance_data = R_sei_0 + gamma_sei * x  # 单位：欧姆 (Ω)

    # ==========================================
    # 定义配置：(CSV键名, Y轴标签, 文件后缀, 换算, 放大区域[x1,x2,y1,y2], 放大镜位置[x,y,w,h], 零线, 显示图例)
    # ==========================================
    plots_config = [
        ("total_reward", "累计回报(-)", "reward", 1.0, 
        [20, 50, -50, 10], [0.12, 0.12, 0.4, 0.32], 0, False),
        ("charge_time_min", "充电时间(s)", "charge_time", 60.0, 
        [20, 50, 0, 50], [0.52, 0.12, 0.4, 0.32], 0, False),
        ("voltage_violation", "电压越限(V)", "voltage_viol", 1.0, 
        [20, 50, -0.1, 0.1], [0.52, 0.12, 0.4, 0.32], 0, False),
        ("temperature_violation", "最高温度(°C)", "temperature", 1.0,  # 修改此处：K -> °C
        None, None, None, True)
    ]

    for base_key, ylabel, suffix, multiplier, zoom_limits, inset_pos, limit_line_y, show_legend in plots_config:
        fig, ax = plt.subplots(figsize=figsize_single)
        
        static_mean = np.array([r[f"static_{base_key}_mean"] for r in data_rows]) * multiplier
        static_min = np.array([r[f"static_{base_key}_min"] for r in data_rows]) * multiplier
        static_max = np.array([r[f"static_{base_key}_max"] for r in data_rows]) * multiplier
        
        comb_mean = np.array([r[f"combined_{base_key}_mean"] for r in data_rows]) * multiplier
        comb_min = np.array([r[f"combined_{base_key}_min"] for r in data_rows]) * multiplier
        comb_max = np.array([r[f"combined_{base_key}_max"] for r in data_rows]) * multiplier

        # 2. 修改温度转换逻辑
        if base_key == "temperature_violation":
            # 原逻辑是加 309.15 (Kelvin)，转换为摄氏度需减去 273.15
            # 309.15 - 273.15 = 36.0
            offset = 36.0 
            static_mean += offset
            static_min += offset
            static_max += offset
            comb_mean += offset
            comb_min += offset
            comb_max += offset

        if limit_line_y is not None:
            ax.axhline(y=limit_line_y, color='black', linestyle='--', linewidth=0.5, zorder=1)

        # 绘制主轴 (左侧: 累计回报、充电时间等) -------------------------
        ax.fill_between(x, static_min, static_max, color=color_static, alpha=0.15, zorder=2, edgecolor='none')
        l1, = ax.plot(x, static_mean, color=color_static, label="静态模型", zorder=3, linewidth=0.5)
        
        ax.fill_between(x, comb_min, comb_max, color=color_adapt, alpha=0.15, zorder=4, edgecolor='none')
        l2, = ax.plot(x, comb_mean, color=color_adapt, label="老化衰减模型", zorder=5, linewidth=0.5)

        # 绘制副轴 (右侧: 附加极化阻抗) -------------------------
        ax2 = ax.twinx()
        l3, = ax2.plot(x, resistance_data, color=color_drift, label="附加极化阻抗", zorder=1, linewidth=0.8)
        
        # 设置右轴的样式（绿色轴线和刻度）
        ax2.set_ylabel("附加极化阻抗 (Ω)", color=color_drift, fontsize=9)
        ax2.tick_params(axis='y', labelcolor=color_drift, direction="in", length=3, pad=2, width=0.5)
        ax2.spines["right"].set_color(color_drift)
        ax2.spines["right"].set_linewidth(0.5)
        for label in ax2.get_yticklabels():
            label.set_fontname('Times New Roman')
            
        # 设置 Y 轴范围使其更美观 (可选：让0.005到0.105的曲线在图表中处于合适的高度)
        ax2.set_ylim(0, 0.12) 

        # # 放大镜逻辑 -------------------------
        # if zoom_limits and inset_pos:
        #     axins = ax.inset_axes(inset_pos)
        #     if limit_line_y is not None:
        #         axins.axhline(y=limit_line_y, color='black', linestyle='--', linewidth=0.5, zorder=1)

        #     axins.fill_between(x, static_min, static_max, color=color_static, alpha=0.15, zorder=2, edgecolor='none')
        #     axins.plot(x, static_mean, color=color_static, zorder=3, linewidth=0.5)
        #     axins.fill_between(x, comb_min, comb_max, color=color_adapt, alpha=0.15, zorder=4, edgecolor='none')
        #     axins.plot(x, comb_mean, color=color_adapt, zorder=5, linewidth=0.5)
            
        #     x1, x2, y1, y2 = zoom_limits
        #     axins.set_xlim(x1, x2)
        #     axins.set_ylim(y1, y2)
            
        #     axins.tick_params(axis='both', which='major', labelsize=7, pad=1, length=1.5, width=0.4)
        #     for label in axins.get_xticklabels() + axins.get_yticklabels():
        #         label.set_fontname('Times New Roman')
                
        #     rect_patch, connect_lines = ax.indicate_inset_zoom(axins, edgecolor="black", alpha=0.6, linewidth=0.5)
        #     for line in connect_lines:
        #         line.set_linestyle('--')
        #         line.set_linewidth(0.4)

        # 轴标签和刻度设置
        ax.set_ylabel(ylabel, fontsize=9)
        ax.set_xlabel("老化阶段", fontsize=9) 
        ax.tick_params(axis='both', pad=2, length=3)
        for label in ax.get_xticklabels() + ax.get_yticklabels():
            label.set_fontname('Times New Roman')

        # 合并左右两轴的图例 -------------------------
        if show_legend:
            handles = [l1, l2, l3]
            labels = [h.get_label() for h in handles]
            
            ax.legend(handles, labels, loc='upper center', ncol=1, frameon=True, 
                      facecolor='white', framealpha=0.8,
                      edgecolor=(0.7, 0.7, 0.7, 1.0), borderpad=0.3, 
                      handletextpad=0.3, labelspacing=0.3)

        plt.tight_layout()
        plt.savefig(out_dir / f"{prefix}_{suffix}.png", dpi=600, bbox_inches='tight', pad_inches=0.1)
        plt.savefig(out_dir / f"{prefix}_{suffix}.pdf", format='pdf', bbox_inches='tight', pad_inches=0.1)
        plt.close(fig)
# def plot_exp2_separate(data_rows: List[Dict], out_dir: Path, prefix: str) -> None:
#     color_static = '#3C5488'
#     color_adapt = '#E64B35'
    
#     cm_to_inch = 1 / 2.54
#     figsize_single = (8.0 * cm_to_inch, 5.5 * cm_to_inch)

#     x = np.array([r["stage"] for r in data_rows])

#     # ==========================================
#     # 定义配置：(CSV键名, Y轴标签, 文件后缀, 换算, 放大区域[x1,x2,y1,y2], 放大镜位置[x,y,w,h], 零线, 显示图例)
#     # ==========================================
#     plots_config = [
#         # 1. 累计回报：放大 20-50 阶段组合策略，位置左下角
#         ("total_reward", "累计回报(-)", "reward", 1.0, 
#          [20, 50, -50, 10], [0.12, 0.12, 0.4, 0.32], 0, False),
         
#         # 2. 充电时间：放大 20-50 阶段静态策略，位置右下角
#         ("charge_time_min", "充电时间(s)", "charge_time", 60.0, 
#          [20, 50, 0, 50], [0.52, 0.12, 0.4, 0.32], 0, False),
         
#         # 3. 电压越限：放大 20-50 阶段双策略对比，位置右下角
#         ("voltage_violation", "电压越限(V)", "voltage_viol", 1.0, 
#          [20, 50, -0.1, 0.1], [0.52, 0.12, 0.4, 0.32], 0, False),
         
#         # 4. 温度：最高温度显示，无放大镜，保留图例
#         ("temperature_violation", "最高温度(K)", "temperature", 1.0, 
#          None, None, None, True)
#     ]

#     for base_key, ylabel, suffix, multiplier, zoom_limits, inset_pos, limit_line_y, show_legend in plots_config:
#         fig, ax = plt.subplots(figsize=figsize_single)
        
#         static_mean = np.array([r[f"static_{base_key}_mean"] for r in data_rows]) * multiplier
#         static_min = np.array([r[f"static_{base_key}_min"] for r in data_rows]) * multiplier
#         static_max = np.array([r[f"static_{base_key}_max"] for r in data_rows]) * multiplier
        
#         comb_mean = np.array([r[f"combined_{base_key}_mean"] for r in data_rows]) * multiplier
#         comb_min = np.array([r[f"combined_{base_key}_min"] for r in data_rows]) * multiplier
#         comb_max = np.array([r[f"combined_{base_key}_max"] for r in data_rows]) * multiplier

#         # 温度换算逻辑
#         if base_key == "temperature_violation":
#             static_mean += 309.15
#             static_min += 309.15
#             static_max += 309.15
#             comb_mean += 309.15
#             comb_min += 309.15
#             comb_max += 309.15

#         # 绘制基准虚线
#         if limit_line_y is not None:
#             ax.axhline(y=limit_line_y, color='black', linestyle='--', linewidth=0.5, zorder=1)

#         # 绘制主线 (lw=0.5)
#         ax.fill_between(x, static_min, static_max, color=color_static, alpha=0.15, zorder=2, edgecolor='none')
#         ax.plot(x, static_mean, color=color_static, label="静态模型", zorder=3, linewidth=0.5)
#         ax.fill_between(x, comb_min, comb_max, color=color_adapt, alpha=0.15, zorder=4, edgecolor='none')
#         ax.plot(x, comb_mean, color=color_adapt, label="老化衰减模型", zorder=5, linewidth=0.5)

#         # 放大镜逻辑
#         if zoom_limits and inset_pos:
#             axins = ax.inset_axes(inset_pos)
#             if limit_line_y is not None:
#                 axins.axhline(y=limit_line_y, color='black', linestyle='--', linewidth=0.5, zorder=1)

#             axins.fill_between(x, static_min, static_max, color=color_static, alpha=0.15, zorder=2, edgecolor='none')
#             axins.plot(x, static_mean, color=color_static, zorder=3, linewidth=0.5)
#             axins.fill_between(x, comb_min, comb_max, color=color_adapt, alpha=0.15, zorder=4, edgecolor='none')
#             axins.plot(x, comb_mean, color=color_adapt, zorder=5, linewidth=0.5)
            
#             x1, x2, y1, y2 = zoom_limits
#             axins.set_xlim(x1, x2)
#             axins.set_ylim(y1, y2)
            
#             axins.tick_params(axis='both', which='major', labelsize=7, pad=1, length=1.5, width=0.4)
#             for label in axins.get_xticklabels() + axins.get_yticklabels():
#                 label.set_fontname('Times New Roman')
                
#             rect_patch, connect_lines = ax.indicate_inset_zoom(axins, edgecolor="black", alpha=0.6, linewidth=0.5)
#             for line in connect_lines:
#                 line.set_linestyle('--')
#                 line.set_linewidth(0.4)

#         ax.set_ylabel(ylabel, fontsize=9)
#         ax.set_xlabel("老化阶段", fontsize=9)

#         if show_legend:
#             ax.legend(loc='best', ncol=1, frameon=True, facecolor='white', framealpha=0.8,
#                       edgecolor=(0.7, 0.7, 0.7, 0.5), borderpad=0.3, handletextpad=0.3, labelspacing=0.3)

#         ax.tick_params(axis='both', pad=2, length=3)
#         for label in ax.get_xticklabels() + ax.get_yticklabels():
#             label.set_fontname('Times New Roman')

#         plt.tight_layout()
#         plt.savefig(out_dir / f"{prefix}_{suffix}.png", dpi=600, bbox_inches='tight')
#         plt.savefig(out_dir / f"{prefix}_{suffix}.pdf", format='pdf', bbox_inches='tight')
#         plt.close(fig)
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