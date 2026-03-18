from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, List

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from src.rl.agent_factory import build_agent_from_config
from src.utils.config import load_config
from src.utils.seeds import set_global_seed
from scripts.other_method.common import (
    # 注意：这里移除了原始的 plot_comparison，我们改用自己规范化手写的
    rollout_one_episode,
    save_traj_csv,
    stage_policy_from_schedule,
)


def _build_pack_env(env_cfg: Dict[str, Any]):
    # 延迟导入：允许在未安装 liionpack 的环境下也能执行 --help。
    from src.envs.liionpack_spme_pack_env import build_pack_env
    return build_pack_env(env_cfg)


def build_policy_from_manifest_item(env, item: Dict[str, Any], cfg: Dict[str, Any]):
    if item["type"] == "schedule":
        schedule = json.loads(Path(item["path"]).read_text(encoding="utf-8"))
        return stage_policy_from_schedule(
            schedule=schedule,
            action_low=float(env.action_space.low[0]),
            action_high=float(env.action_space.high[0]),
        )

    if item["type"] == "rl_agent":
        algorithm = item["algorithm"]
        cfg_rl = json.loads(json.dumps(cfg["rl"]))
        cfg_rl["algorithm"] = algorithm

        agent = build_agent_from_config(
            state_dim=env.observation_space.shape[0],
            action_dim=1,
            rl_config=cfg_rl,
        )
        agent.load(item["path"])

        low = float(env.action_space.low[0])
        high = float(env.action_space.high[0])

        def _policy(obs, _info):
            action = agent.act(obs)
            return np.array([float(np.clip(action[0], low, high))], dtype=np.float32)

        return _policy

    raise ValueError(f"Unknown item type: {item['type']}")


def plot_publication_comparison(method_to_traj: Dict[str, List[Dict]], output_dir: Path):
    """
    顶刊规范化的 2x2 矩阵子图：绘制电流、SOC、电压、温度的时间对比曲线。
    """
    # 1. 强制屏蔽字体内部元数据警告
    logging.getLogger('fontTools.subset').level = logging.ERROR

    # 2. 加载宋体
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

    # 4. Nature 经典学术配色 (映射不同方法)
    # 你可以根据实际 manifest 里的名字调整匹配关键字
    color_dict = {
        "ddpg": '#3C5488',   # 深蓝色
        "td3": '#E64B35',    # 红色
        "mcc": '#00A087',    # 蓝绿色 (用于多阶段恒流, 根据你的具体缩写匹配)
        "default": '#A9A9A9' # 灰色兜底
    }

    # 物理尺寸规范：2x2 矩阵子图，单图 7.5cm x 6.0cm -> 整体约 15cm x 12cm
    cm_to_inch = 1 / 2.54
    fig, axs = plt.subplots(2, 2, figsize=(15.0 * cm_to_inch, 12.0 * cm_to_inch))
    axs = axs.flatten()

    # 提取轨迹字典的常用键（如果你的 traj key 不叫这些，请在此处修改）
    # 通常的轨迹键可能为："t", "current", "voltage", "soc", "temperature"
    metrics = [
        ("current", r"充电电流 ($\mathrm{A}$)","(a) 充电电流"),
        ("soc", r"电池荷电状态","(b) 电池荷电状态"),
        ("vmax", r"端电压 ($\mathrm{V}$)","(c) 端电压"),
        ("tmax", r"最高温度 ($\mathrm{K}$)","(d) 最高温度")
    ]

    for name, traj in method_to_traj.items():
        name_lower = name.lower()
        
        # 寻找对应的颜色
        c = color_dict["default"]
        for k_color, v_color in color_dict.items():
            if k_color in name_lower:
                c = v_color
                break

        if "ddpg" in name_lower:
            display_name = "DDPG方法"
        elif "td3" in name_lower:
            display_name = "本章所提方法"
        elif any(k in name_lower for k in ["mcc", "mscc", "multi"]):
            display_name = "多阶段恒流方法"
        else:
            display_name = name.upper()
        # 转换时间轴单位（假设环境是以秒记录，画图时转换为分钟更直观。如果不需转换则去掉 /60.0）
        time_seq = np.array([step.get("t", step.get("time", 0)) for step in traj]) 

        for i, (key, ylabel) in enumerate(metrics):
            # 获取 y 轴数据，兼容常见的键命名
            # 如果你的字典里叫 'V' 或 'T_cell' 等，请相应调整 'step.get' 的备选项
            y_seq = np.array([step.get(key, step.get(key.capitalize(), 0)) for step in traj])
            
            axs[i].plot(time_seq, y_seq, label=display_name, color=c, linewidth=1.5)

    for i, (key, ylabel) in enumerate(metrics):
        ax = axs[i]
        ax.set_ylabel(ylabel)
        if i >= 2:
            ax.set_xlabel(r"时间 ($\mathrm{s}$)")
        
        # 网格与刻度强制规范
        ax.grid(True, linestyle=':', alpha=0.6, color='#CCCCCC')
        ax.tick_params(axis='x', pad=2, length=3)
        ax.tick_params(axis='y', pad=2, length=3)
        for label in ax.get_xticklabels() + ax.get_yticklabels():
            label.set_fontname('Times New Roman')
        
    fig.align_ylabels()
    handles, labels = axs[0].get_legend_handles_labels()
    fig.legend(
        handles, labels,
        loc='lower center',       # 定位点为下居中
        bbox_to_anchor=(0.5, 0.02),  # 锚点放在画布底端中间
        ncol=3,                   # 强制分为3列（一行展示完）
        frameon=True,              
        facecolor='white',         
        framealpha=0.9,            
        edgecolor=(0.7, 0.7, 0.7, 0.5),
        borderpad=0.5,       
        handletextpad=0.4,   
        columnspacing=2.0         # 增加列之间的间距，显得更舒展
    )

    # 【核心修改】：调整布局，把底部的 8% 空出来留给图例
    plt.tight_layout(rect=[0, 0.05, 1, 1])

    
    output_pdf = output_dir / "real_env_comparison.pdf"
    output_png = output_dir / "real_env_comparison.png"
    plt.savefig(output_pdf, format='pdf', bbox_inches='tight')
    plt.savefig(output_png, dpi=300, bbox_inches='tight')
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run one real-environment episode for specific comparison methods.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--manifest", required=True, help="Path of manifest.json generated by train script.")
    parser.add_argument("--output", default="runs/other_method/eval")
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()

    cfg = load_config(args.config).data
    set_global_seed(args.seed)

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))

    method_to_traj = {}
    summary_rows = []
    
    # 允许保留的方法关键词（确保 "多阶段恒流" 在你的 manifest 中是用 "mcc", "mscc" 或 "multi" 命名的）
    # 如果你的多阶段恒流拼写不同，请在这里补充！
    allowed_keywords = ["ddpg", "td3", "mcc", "mscc", "multi"]

    for name, item in manifest.items():
        name_lower = name.lower()
        
        # 核心过滤逻辑：屏蔽 ppo 和 ga，且只允许在 allowed_keywords 中的方法
        if "ppo" in name_lower or "ga" in name_lower:
            print(f"[Skip] {name} 触发屏蔽规则 (ppo/ga)")
            continue
            
        if not any(k in name_lower for k in allowed_keywords):
            print(f"[Skip] {name} 不在白名单中 ({allowed_keywords})")
            continue

        env = _build_pack_env(cfg["env"])
        policy = build_policy_from_manifest_item(env=env, item=item, cfg=cfg)
        traj, summary = rollout_one_episode(env, policy, seed=args.seed)
        method_to_traj[name] = traj
        summary_rows.append({"method": name, **summary})

        save_traj_csv(output_dir / f"traj_{name}.csv", traj)
        print(
            f"[{name}] steps={summary['steps']:.0f} soc_end={summary['soc_end']:.4f} "
            f"vmax_peak={summary['vmax_peak']:.4f} tmax_peak={summary['tmax_peak']:.2f} "
            f"i_mean={summary['current_mean']:.3f}"
        )

    # 调用我们手写的、符合规范的绘图函数
    plot_publication_comparison(method_to_traj, output_dir)
    
    (output_dir / "summary.json").write_text(
        json.dumps(summary_rows, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"\n[DONE] Saved outputs and publication-ready plots to: {output_dir}")


if __name__ == "__main__":
    main()