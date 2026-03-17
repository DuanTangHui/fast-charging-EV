import argparse
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# 设置中文字体，防止 matplotlib 图表中的中文显示为方块
plt.rcParams['font.sans-serif'] = ['SimHei', 'Songti SC', 'Arial Unicode MS'] # 兼容 Windows 和 Mac
plt.rcParams['axes.unicode_minus'] = False 

def plot_metric_curves(df_real: pd.DataFrame, df_mix: pd.DataFrame, output: Path, warmup_episodes: int) -> None:
    # 设置滑动窗口大小
    window_size = 30 
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), sharex=True)
    settings = [
        ("total_reward", "累积回报 (Total Reward)"),
        ("charge_time_s", "充电时间 (s)"),
        ("voltage_violation", "电压违规程度 (max(V) - 4.2)"),
        ("temperature_violation", "温度违规程度 (max(T) - 309.15)"),
    ]
    
    for ax, (col, title) in zip(axes.ravel(), settings):
        # 检查列是否存在，防止报错
        if col not in df_real.columns or col not in df_mix.columns:
            print(f"[警告] 数据中找不到列: {col}，请检查 CSV 文件！")
            continue

        # 计算滑动平均
        real_smooth = df_real[col].rolling(window=window_size, min_periods=1).mean()
        mix_smooth = df_mix[col].rolling(window=window_size, min_periods=1).mean()
        
        # 1. 绘制真实基线
        ax.plot(df_real["episode"], df_real[col], alpha=0.2, color="tab:blue")
        ax.plot(df_real["episode"], real_smooth, label="真实基线 (Real)", linewidth=2.5, color="tab:blue")
        
        # 2. 绘制混合方案
        ax.plot(df_mix["episode"], df_mix[col], alpha=0.2, color="tab:orange")
        ax.plot(df_mix["episode"], mix_smooth, label="混合方案 (Hybrid)", linewidth=2.5, linestyle="--", color="tab:orange")
        
        # 3. 添加 Warmup 分界线
        ax.axvline(x=warmup_episodes, color="red", linestyle=":", linewidth=1.5, label="Surrogate 介入点")
        
        # 4. 图表美化
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.grid(alpha=0.3, linestyle="--")
        
        # 获取最大 episode 数量以对齐 X 轴
        max_ep = max(df_real["episode"].max(), df_mix["episode"].max())
        ax.set_xlim(0, max_ep)
        
        # 添加 y=0 的安全基准线（仅针对违规指标）
        if "violation" in col:
            ax.axhline(y=0, color="gray", linestyle="-", linewidth=1)

    axes[1, 0].set_xlabel("Episode Number", fontsize=11)
    axes[1, 1].set_xlabel("Episode Number", fontsize=11)
    
    axes[0, 0].legend(loc="best", fontsize=10)
    
    plt.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[Done] 图表已成功保存至: {output}")

def main() -> None:
    parser = argparse.ArgumentParser(description="单独绘制强化学习实验的训练过程对比图")
    # 这里默认指向你之前跑出来的结果目录，你可以根据实际情况修改
    parser.add_argument("--data-dir", type=Path, default=Path("runs/static_experiment/exp2_training_process_1000"))
    parser.add_argument("--warmup", type=int, default=50, help="真实环境预热的 episode 数量")
    args = parser.parse_args()

    real_csv = args.data_dir / "real_baseline_metrics.csv"
    hybrid_csv = args.data_dir / "hybrid_metrics.csv"
    output_img = args.data_dir / "training_process_1000_compare_standalone.png"

    if not real_csv.exists() or not hybrid_csv.exists():
        print(f"[错误] 找不到 CSV 文件，请确认路径是否正确: {args.data_dir}")
        return

    # 读取 CSV 数据
    df_real = pd.read_csv(real_csv)
    df_mix = pd.read_csv(hybrid_csv)

    # 调用绘图函数
    plot_metric_curves(df_real, df_mix, output_img, args.warmup)

if __name__ == "__main__":
    main()