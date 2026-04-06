import csv
import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List, Dict
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

def plot_metrics(rows: List[Dict[str, float]], out_dir: Path) -> None:
    from matplotlib.ticker import MultipleLocator, ScalarFormatter
    setup_plot_style()
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
    x_min, x_max = min(stages), max(stages)
    x_padding = (x_max - x_min) * 0.05 if x_max > x_min else 2


    fig_comb, ax_mse_c = plt.subplots(figsize=figsize_single)
    ax_mae_c = ax_mse_c.twinx()
    
    line1, = ax_mse_c.plot(stages, mse, marker="o", markersize=5, linewidth=1.5, 
                           color=color_mse, markeredgecolor="white", markeredgewidth=0.8, label="均方误差")
    line2, = ax_mae_c.plot(stages, mae, marker="s", markersize=5, linewidth=1.5, 
                           color=color_mae, markeredgecolor="white", markeredgewidth=0.8, label="平均绝对误差")
    
    ax_mse_c.set_xlabel(r"老化阶段")
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
    leg = ax_mse_c.legend(lines, labels, loc='upper left')
    leg.get_frame().set_linewidth(0.5)

    fig_comb.tight_layout()
    fig_comb.savefig(out_dir / "stages_combined_mse_mae.pdf", format='pdf', bbox_inches="tight")
    fig_comb.savefig(out_dir / "stages_combined_mse_mae.png", dpi=300, bbox_inches="tight")
    plt.close(fig_comb)
# 这里请保留您原本代码中的 setup_plot_style 和 plot_metrics 函数
# def setup_plot_style(): ...
# def plot_metrics(rows: List[Dict[str, float]], out_dir: Path): ...

def main_plot_only() -> None:
    p = argparse.ArgumentParser(description="直接读取CSV文件进行绘图")
    # 默认路径指向您之前保存 CSV 的位置
    p.add_argument("--csv-path", default="runs/adaptive_experiment/result1/exp1_stage_cv_metrics.csv")
    p.add_argument("--output-dir", default="runs/adaptive_experiment/result1")
    args = p.parse_args()

    csv_path = Path(args.csv_path)
    out_dir = Path(args.output_dir)

    if not csv_path.exists():
        raise FileNotFoundError(f"找不到指定的 CSV 文件: {csv_path}。请检查路径是否正确。")

    rows: List[Dict[str, float]] = []
    
    # 1. 读取 CSV 文件
    print(f"正在读取数据: {csv_path}")
    with csv_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            # 将读取到的字符串转换为浮点数
            parsed_row = {
                "stage": float(row["stage"]),
                "r2": float(row["r2"]),
                "mse": float(row["mse"]),
                "mae": float(row["mae"])
            }
            # 如果 CSV 中有 samples 也可以读出来，但画图目前不需要
            rows.append(parsed_row)

    # 2. 确保按照 stage 排序（以防 CSV 顺序乱了）
    rows = sorted(rows, key=lambda x: x["stage"])

    # 3. 确保输出目录存在
    out_dir.mkdir(parents=True, exist_ok=True)

    # 4. 调用画图函数
    print("开始绘制评估指标图表...")
    plot_metrics(rows, out_dir)
    print(f"[DONE] 绘图完成！图表已保存至: {out_dir}")


if __name__ == "__main__":
    # 执行纯绘图逻辑
    main_plot_only()