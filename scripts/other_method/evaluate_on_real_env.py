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
    顶刊规范化：独立保存电流、SOC、电压、温度曲线为单独的PDF。
    已根据用户要求：
    1. 修改配色：GA 和 Other 颜色改为清晰亮色。
    2. 图例顺序：将‘本文所提方法’排在第一位。
    3. 标签修改：端电压改为“单体最大电压 (V)”。
    4. 单位修改：温度单位改为摄氏度 (°C)，并自动进行 K 转 °C 的计算。
    """
    # 1. 屏蔽字体元数据警告
    logging.getLogger('fontTools.subset').level = logging.ERROR

    # 2. 加载宋体 (针对Windows)
    simsun_path = r'C:\Windows\Fonts\simsun.ttc'
    if os.path.exists(simsun_path):
        font_manager.fontManager.addfont(simsun_path)

    # 3. 顶刊全局标准配置
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["SimSun", "Times New Roman"], 
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
        
        "font.size": 9,             
        "axes.labelsize": 9,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9  
    })

    # 4. 配色方案：清晰、高对比度的亮色
    color_dict = {
        "ddpg": '#3C5488',   # 深蓝色 (DDPG)
        "td3": '#E64B35',    # 红色 (本文所提方法)
        "mcc": '#00A087',    # 蓝绿色 (MCC)
        "ga": '#FF9E1C',     # 清晰亮丽的橙色 (GA)
        "default": '#A040A0' # 清晰亮丽的紫色 (Other)
    }

    # 5. 【核心修改】指标配置更新标签和单位
    # 格式：(键, 纵轴标签, 文件名后缀)
    metrics = [
        ("current", "充电电流 (A)", "Current"),
        ("soc", "电池荷电状态 (-)", "SOC"),
        ("vmax", "单体最大电压 (V)", "Voltage"), # 修改为单体最大电压
        ("tmax", "最高温度 (℃)", "Temperature") # 修改为摄氏度
    ]

    cm_to_inch = 1 / 2.54
    # 设定单图尺寸，例如 8cm x 6cm
    fig_size = (8.0 * cm_to_inch, 6.0 * cm_to_inch)

    for key, ylabel, filename in metrics:
        fig, ax = plt.subplots(figsize=fig_size)
        
        # 创建绘图项目列表，用于排序和统一处理
        plot_info_list = []

        for name, traj in method_to_traj.items():
            name_lower = name.lower()
            
            # 颜色匹配
            c = color_dict["default"]
            for k_color, v_color in color_dict.items():
                if k_color in name_lower:
                    c = v_color
                    break

            # 命名转换
            if "ddpg" in name_lower:
                display_name = "DDPG"
            elif "td3" in name_lower:
                display_name = "本文所提方法"
            elif "ga" in name_lower:
                display_name = r"GA-MSCC$^{[83]}$"
            elif any(k in name_lower for k in ["mcc", "mscc", "multi"]):
                display_name = r"MSCC$^{[82]}$"
            else:
                display_name = name.upper()

            # 确定图例顺序优先级 (数字越小越靠前)
            order_priority = 5
            if display_name == "本文所提方法":
                order_priority = 0
            elif display_name == r"GA-MSCC$^{[83]}$":
                order_priority = 1
            elif display_name ==  r"MSCC$^{[82]}$":
                order_priority = 2
            elif display_name == "DDPG":
                order_priority = 3
            elif display_name == name.upper():
                order_priority = 4

            # 时间轴处理
            time_seq = np.array([step.get("t", step.get("time", 0)) for step in traj]) 
            
            # 数据提取与处理
            y_seq = np.array([step.get(key, step.get(key.capitalize(), 0)) for step in traj], dtype=float)
            
            # 电流取绝对值
            if key == "current":
                y_seq = np.abs(y_seq)
            
            # 【核心修改】温度开尔文转摄氏度
            if key == "tmax":
                # 简单判定：如果温度均值大于200，说明原数据大概率是开尔文(K)
                if len(y_seq) > 0 and np.mean(y_seq) > 200:
                    y_seq = y_seq - 273.15
            
            # 将绘图信息存入列表
            plot_info_list.append({
                "time": time_seq,
                "y": y_seq,
                "label": display_name,
                "color": c,
                "order": order_priority,
                "name": name 
            })

        # 按自定义优先级排序
        plot_info_list.sort(key=lambda x: (x["order"], x["name"]))

        # 按排序后的顺序绘图
        for plot_info in plot_info_list:
            ax.plot(plot_info["time"], plot_info["y"], label=plot_info["label"], color=plot_info["color"], linewidth=0.5)

        # 修饰每个独立的图
        ax.set_ylabel(ylabel)
        ax.set_xlabel("时间 (s)")
        
        # 强制刻度字体为 Times New Roman
        for label in ax.get_xticklabels() + ax.get_yticklabels():
            label.set_fontname('Times New Roman')
        
        # 图例内置到每个子图
        ax.legend(loc='best', frameon=True, framealpha=0.8)

        # 保存独立文件
        plt.tight_layout()
        pdf_path = output_dir / f"comparison_{filename.lower()}.pdf"
        plt.savefig(pdf_path, format='pdf', bbox_inches='tight')
        plt.close(fig)

    print(f"所有独立指标图已保存至: {output_dir}")

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
    allowed_keywords = ["ddpg", "td3", "mcc", "mscc", "multistage_cc","ga_cc"]

    for name, item in manifest.items():
        name_lower = name.lower()
        
        # 核心过滤逻辑：屏蔽 ppo 和 ga，且只允许在 allowed_keywords 中的方法
        # if "ppo" in name_lower or "ga" in name_lower:
        #     print(f"[Skip] {name} 触发屏蔽规则 (ppo/ga)")
        #     continue
            
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