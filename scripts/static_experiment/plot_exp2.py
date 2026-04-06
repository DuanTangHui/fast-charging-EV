import sys
import argparse
import os
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.font_manager as font_manager
import matplotlib.ticker as ticker

# ==============================================================================
# 终极 PDF 中英文字体混合与学术标准设置
# ==============================================================================
simsun_path = r'C:\Windows\Fonts\simsun.ttc'
if os.path.exists(simsun_path):
    font_manager.fontManager.addfont(simsun_path)
plt.rcParams.update({
    'font.family': ['Times New Roman', 'SimSun'],
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
    "legend.fontsize": 9  # 图例字号调小，防止越界
})

current_dir = Path(__file__).resolve().parent
if str(current_dir) not in sys.path:
    sys.path.append(str(current_dir))

from common import (
    build_env_agent_reward,
    evaluate_policy_trajectory,
)
from mpl_toolkits.axes_grid1.inset_locator import mark_inset  # 确保导入放大镜连接线工具

CM_TO_INCH = 1 / 2.54
SUBPLOT_WIDTH_CM = 8.0
SUBPLOT_HEIGHT_CM = 5.5

def plot_debug_trajectories(real_trajs: list, mix_trajs: list, output_dir: Path) -> None:
    # 使用你指定的尺寸规范
    sub_size = (SUBPLOT_WIDTH_CM * CM_TO_INCH, SUBPLOT_HEIGHT_CM * CM_TO_INCH)
    
    # 纯文本表示
    panels = [
        ("current_a", "充电电流 (A)", "current"),
        ("soc", "电池荷电状态 (-)", "soc"),
        ("voltage_v", "单体最大电压 (V)", "voltage"),
        ("temperature_k", "最高温度 (°C)", "temperature"),
    ]

    output_dir.mkdir(parents=True, exist_ok=True)

    # 去掉 LaTeX 公式，直接返回数值字符串
    def times_formatter(x, pos):
        return f"{x:g}"

    for key, ylabel, filename in panels:
        fig, ax = plt.subplots(figsize=sub_size)
        
        # 1. 绘制真实仿真环境 (蓝色)
        for i, (name, traj) in enumerate(real_trajs):
            label = "真实仿真环境" if i == 0 else "_nolegend_"
            if key == "temperature_k":
                y_data = traj[key] - 273.15
            elif key == "current_a":
                y_data = abs(traj[key])  # 取绝对值
            else:
                y_data = traj[key]
            ax.plot(traj["time_s"], y_data, color="tab:blue", alpha=0.8, linewidth=0.5, label=label)
            
        # 2. 绘制静态模型 (橙色)
        for i, (name, traj) in enumerate(mix_trajs):
            label = "静态模型" if i == 0 else "_nolegend_"
            if key == "temperature_k":
                y_data = traj[key] - 273.15
            elif key == "current_a":
                y_data = abs(traj[key])  # 取绝对值
            else:
                y_data = traj[key]
            ax.plot(traj["time_s"], y_data, color="tab:orange", linestyle="-", alpha=0.8, linewidth=0.5, label=label)
        
        # 纯文本表示
        ax.set_xlabel("时间 (s)")
        ax.set_ylabel(ylabel)
        
        # --- 基准虚线与局部放大镜 ---
        if key == "soc":
            ax.axhline(0.8, color="k", linestyle="--", linewidth=1.0)
        elif key == "voltage_v":
            ax.axhline(4.2, color="k", linestyle="--", linewidth=1.0)
            
            # 电压局部放大镜
            axins = ax.inset_axes([0.4, 0.15, 0.55, 0.4]) 
            for name, traj in real_trajs:
                axins.plot(traj["time_s"], traj[key], color="tab:blue", alpha=0.8, linewidth=0.5)
            for name, traj in mix_trajs:
                axins.plot(traj["time_s"], traj[key], color="tab:orange", linestyle="-", alpha=0.8, linewidth=0.5)
            
            axins.axhline(4.2, color="k", linestyle="--", linewidth=1.0)
            axins.set_xlim(500, 2000)
            axins.set_ylim(4.0, 4.21)
            mark_inset(ax, axins, loc1=1, loc2=3, fc="none", ec="0.5", linewidth=0.5)
        
        # 坐标轴格式化
        ax.xaxis.set_major_formatter(ticker.FuncFormatter(times_formatter))
        ax.yaxis.set_major_formatter(ticker.FuncFormatter(times_formatter))
        
        # --- 核心修改：精细化带框图例，仅在温度图显示 ---
        if key == "temperature_k":
            leg = ax.legend(
                loc="upper right",
                frameon=True,
                facecolor='white',
                framealpha=0.9,
                edgecolor=(0.7, 0.7, 0.7, 0.5),
                borderpad=0.3,
                handletextpad=0.3,
                labelspacing=0.2
            )
            if leg:
                leg.get_frame().set_linewidth(0.5)
            
        # --- 核心修改：固定子图边距，防止导出图片的主绘图区变形 ---
        fig.subplots_adjust(left=0.18, right=0.82, top=0.92, bottom=0.18)
            
        save_path = output_dir / f"{filename}.pdf"
        plt.savefig(save_path, format="pdf", dpi=300, bbox_inches="tight")
        plt.close(fig)

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/pack_3p6s_spme.yaml"))
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--real-agents-dir", type=Path, default=Path("runs/real_agents_batch"))
    parser.add_argument("--mix-agents-dir", type=Path, default=Path("runs/static_agents_batch"))
    parser.add_argument("--output-dir", type=Path, default=Path("runs/static_experiment/exp1_policy_shape"))
    args = parser.parse_args()

    # 收集真实环境模型 (保存为元组: (文件名, 轨迹))
    real_trajs = []
    real_ckpts = list(args.real_agents_dir.glob("*.pt"))
    for ckpt in real_ckpts:
        print(f"正在加载真实环境智能体: {ckpt.name}")
        cfg, env_real, agent_real, _ = build_env_agent_reward(args.config, args.seed)
        agent_real.load(str(ckpt), map_location="cpu")
        traj = evaluate_policy_trajectory(env_real, agent_real, seed=args.seed)
        real_trajs.append((ckpt.name, traj))  # 存入文件名

    # 收集静态代理模型 (保存为元组: (文件名, 轨迹))
    mix_trajs = []
    mix_ckpts = list(args.mix_agents_dir.glob("*.pt"))
    for ckpt in mix_ckpts:
        print(f"正在加载静态代理模型智能体: {ckpt.name}")
        _, env_mix, agent_mix, _ = build_env_agent_reward(args.config, args.seed)
        agent_mix.load(str(ckpt), map_location="cpu")
        traj = evaluate_policy_trajectory(env_mix, agent_mix, seed=args.seed)
        mix_trajs.append((ckpt.name, traj))  # 存入文件名

    print("开始绘制带文件名的 Debug PDF 图像...")
    plot_debug_trajectories(real_trajs, mix_trajs, args.output_dir)
    print(f"[Done] 排查用 PDF 图片已保存至: {args.output_dir}")

if __name__ == "__main__":
    main()