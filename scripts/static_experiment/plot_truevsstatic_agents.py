import sys
import argparse
import os
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.font_manager as font_manager
import matplotlib.ticker as ticker

# ==============================================================================
# 1. 终极 PDF 中英文字体混合与学术标准设置
# ==============================================================================
simsun_path = r'C:\Windows\Fonts\simsun.ttc'
if os.path.exists(simsun_path):
    font_manager.fontManager.addfont(simsun_path)

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["SimSun"], 
    
    # 核心：将数学文本内的常规字体 (rm) 映射为 Times New Roman
    "mathtext.fontset": "custom",
    "mathtext.rm": "Times New Roman",
    "mathtext.it": "Times New Roman:italic",
    
    # 学术期刊强制要求！确保字体以 TrueType (Type 42) 格式嵌入 PDF
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "axes.unicode_minus": False,
    
    # 开启全包围边框
    "axes.spines.top": True,
    "axes.spines.right": True,
    
    # 将边框和刻度线微调变细，显得更精致
    "axes.linewidth": 1.0,
    "xtick.major.width": 1.0,
    "ytick.major.width": 1.0,
    "xtick.direction": "in",
    "ytick.direction": "in",
    
    # 【全局字号设定为 9】
    "font.size": 9,                     
    "axes.labelsize": 9,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9  
})

current_dir = Path(__file__).resolve().parent
if str(current_dir) not in sys.path:
    sys.path.append(str(current_dir))

from common import (
    build_env_agent_reward,
    evaluate_policy_trajectory,
)

def plot_separate_trajectories(real_traj, mix_traj, output_dir: Path) -> None:
    # 尺寸与颜色设置 (一排限制宽度 7.5cm，长度 5.0cm)
    cm_to_inch = 1 / 2.54
    fig_width = 7.5 * cm_to_inch
    fig_height = 5.0 * cm_to_inch

    # 注意这里的英文和符号，必须用 $\mathrm{}$ 包裹，才能触发 Times New Roman
    panels = [
        ("current_a", "施加电流 $\\mathrm{(A)}$", "current"),
        ("soc", "电池 $\\mathrm{SOC\ (-)}$", "soc"),
        ("voltage_v", "端电压 $\\mathrm{(V)}$", "voltage"),
        ("temperature_k", "电池温度 $\\mathrm{(K)}$", "temperature"),
    ]

    output_dir.mkdir(parents=True, exist_ok=True)

    # 定义一个格式化函数，强制将坐标轴的数字也用 $\mathrm{}$ 包裹
    def times_formatter(x, pos):
        return f"$\\mathrm{{{x:g}}}$"

    for key, ylabel, filename in panels:
        fig, ax = plt.subplots(figsize=(fig_width, fig_height))
        
        # 绘制曲线
        ax.plot(real_traj["time_s"], real_traj[key], label="真实仿真环境", linewidth=1.5)
        ax.plot(mix_traj["time_s"], mix_traj[key], "--", label="静态代理模型", linewidth=1.5)
        
        # 设置标签，X轴同样应用 mathtext
        ax.set_xlabel("时间 $\\mathrm{(s)}$")
        ax.set_ylabel(ylabel)
        
        # 强制坐标轴刻度数字使用 Times New Roman
        ax.xaxis.set_major_formatter(ticker.FuncFormatter(times_formatter))
        ax.yaxis.set_major_formatter(ticker.FuncFormatter(times_formatter))
        
        # 仅在电流图上显示图例，或者你可以根据需要调整
        if key == "current_a":
            ax.legend(frameon=True, loc="best")
            
        plt.tight_layout()
        
        save_path = output_dir / f"{filename}.pdf"
        plt.savefig(save_path, format="pdf", dpi=300, bbox_inches="tight")
        plt.close(fig)

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/pack_3p6s_spme.yaml"))
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--real-agent-ckpt", type=Path, default=Path("runs/static_experiment/exp1_policy_shape/real_trained_agent_ckpt.pt"))
    parser.add_argument("--mix-agent-ckpt", type=Path, default=Path("runs/cycle0-td3/agent_ckpt.pt"))
    parser.add_argument("--output-dir", type=Path, default=Path("runs/static_experiment/exp1_policy_shape/pdf_plots"))
    args = parser.parse_args()

    print(f"正在加载真实环境智能体: {args.real_agent_ckpt}")
    cfg, env_real, agent_real, _ = build_env_agent_reward(args.config, args.seed)
    agent_real.load(str(args.real_agent_ckpt), map_location="cpu")
    real_traj = evaluate_policy_trajectory(env_real, agent_real, seed=args.seed)

    print(f"正在加载混合训练智能体: {args.mix_agent_ckpt}")
    _, env_mix, agent_mix, _ = build_env_agent_reward(args.config, args.seed)
    agent_mix.load(str(args.mix_agent_ckpt), map_location="cpu")
    mix_traj = evaluate_policy_trajectory(env_mix, agent_mix, seed=args.seed)

    print("开始绘制独立 PDF 图像...")
    plot_separate_trajectories(real_traj, mix_traj, args.output_dir)
    print(f"[Done] 4张 PDF 图片已保存至: {args.output_dir}")

if __name__ == "__main__":
    main()