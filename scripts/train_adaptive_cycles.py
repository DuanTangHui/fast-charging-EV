"""Run adaptive cycles with differential surrogate updates."""
from __future__ import annotations

import argparse # 解析命令行参数 
import sys
import os
from dataclasses import replace
from pathlib import Path

# 获取当前脚本的绝对路径
current_dir = Path(__file__).resolve().parent
# 获取项目根目录 (即 scripts 的上一级)
root_dir = current_dir.parent
# 将根目录添加到 Python 搜索路径中
sys.path.append(str(root_dir))
 
import torch

from src.envs.liionpack_spme_pack_env import build_pack_env
from src.rewards.paper_reward import PaperRewardConfig
from src.rl.agent_factory import build_agent_from_config
from src.rl.trainers.trainer_adaptive_gp import AdaptiveConfig, train_adaptive_cycles
from src.surrogate.dataset import build_dataset
from src.surrogate.gp_combined import CombinedSurrogate
from src.surrogate.gp_differential import DifferentialSurrogate
from src.surrogate.gp_static import StaticSurrogate
from src.utils.config import load_config
from src.utils.logging import ensure_dir
from src.utils.seeds import set_global_seed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    config = load_config(args.config).data
    set_global_seed(config.get("seed"))

    env = build_pack_env(config["env"])
    reward_cfg = PaperRewardConfig(**config["reward"])

    static_ckpt = Path(config["logging"]["runs_dir"]) / "cycle0" / "static_surrogate.pt"
    if static_ckpt.exists():
        static_surrogate = torch.load(static_ckpt, weights_only=False)
    else:
        static_surrogate = StaticSurrogate(
            input_dim=env.observation_space.shape[0] + 1,
            output_dim=env.observation_space.shape[0] - 1,
            hidden_sizes=config["surrogate"]["hidden_sizes"],
            ensemble_size=config["surrogate"]["ensemble_size"],
            lr=config["surrogate"]["learning_rate"],
        )
        transitions = []
        for _ in range(2):
            state, info = env.reset()
            done = False
            while not done:
                action = env.action_space.sample()
                next_state, _, terminated, truncated, _ = env.step(action)
                transitions.append((state, action, next_state[:6] - state[:6]))
                state = next_state
                done = terminated or truncated
        dataset = build_dataset(transitions)
        static_surrogate.fit(dataset, epochs=5)
    
    run_dir = ensure_dir(Path(config["logging"]["runs_dir"]) / "adaptive")
    adaptive_cfg = AdaptiveConfig(**config["trainer"]["adaptive"])

    # 每个老化阶段只执行一次“真实采样->差分模型训练->组合模型微调 agent”流程
    adaptive_cfg = replace(adaptive_cfg, cycles=1)
    aging_cfg = config.get("aging", {})
    aging_stages = int(aging_cfg.get("stages", 100))

    cycle0_dir = Path(config["logging"]["runs_dir"]) / "cycle0"
    ckpt_path = cycle0_dir / "agent_ckpt.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Missing checkpoint: {ckpt_path}")
    
    # 热启动：stage_0 从 cycle0 开始，后续 stage 从前一 stage 继续（仅网络参数）
    warm_start_ckpt = ckpt_path

    for stage_idx in range(aging_stages):
        if hasattr(env, "set_aging_stage"):
            env.set_aging_stage(stage_idx)

        stage_dir = ensure_dir(run_dir / f"stage_{stage_idx:03d}")

        agent = build_agent_from_config(
            state_dim=env.observation_space.shape[0],
            action_dim=1,
            rl_config=config["rl"],
        )
        agent.load(str(warm_start_ckpt), map_location="cpu")

        # Stage 切换后清空 replay，避免旧阶段经验污染当前阶段分布
        if hasattr(agent, "buffer") and hasattr(agent.buffer, "buffer"):
            agent.buffer.buffer.clear()

        diff_surrogate = DifferentialSurrogate(
            input_dim=env.observation_space.shape[0] + 1,
            output_dim=env.observation_space.shape[0] - 1,
            hidden_sizes=config["surrogate"]["hidden_sizes"],
            ensemble_size=config["surrogate"]["ensemble_size"],
            lr=config["surrogate"]["learning_rate"],
        )
        combined = CombinedSurrogate(static_surrogate, diff_surrogate)

        train_adaptive_cycles(
            env,
            agent,
            static_surrogate,
            diff_surrogate,
            combined,
            reward_cfg,
            adaptive_cfg,
            str(stage_dir),
            soh_enabled=config["soh_prior"]["enabled"],
            lambda_prior=config["soh_prior"]["lambda_prior"],
            theta_dim=config["soh_prior"]["theta_dim"],
            dummy_soh=config["soh_prior"]["dummy_soh"],
        )

        stage_agent_ckpt = stage_dir / "agent_ckpt.pt"
        agent.save(str(stage_agent_ckpt))
        torch.save(diff_surrogate, stage_dir / "diff_surrogate.pt")
        warm_start_ckpt = stage_agent_ckpt


if __name__ == "__main__":
    main()
