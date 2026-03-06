"""Run adaptive cycles with differential surrogate updates."""
from __future__ import annotations

import argparse # 解析命令行参数 
import sys
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

def test_aging_reasonableness(env, increment=0.0005, cycles=100, test_current=-15.0):
    """
    用于标定老化电阻增量是否合理的测试函数（基于 aging_stage）
    """
    import numpy as np
    print("\n" + "="*50)
    print("🚀 开始进行 Cycle 100 极限老化物理边界测试...")

    # 1) 先做一致性检查（可选）
    # 如果 env 配置里的 sei_resistance_per_cycle 不是你传入的 increment，提醒一下
    env_inc = getattr(env, "_sei_resistance_per_cycle", None)
    if env_inc is not None and abs(float(env_inc) - float(increment)) > 1e-12:
        print(f"⚠️ 注意: env 内配置增量={env_inc}, 你传入 increment={increment}")

    # 2) 设置到目标老化阶段（例如 cycles=100 -> stage=100）
    prev_stage = getattr(env, "_aging_stage", 0)
    env.set_aging_stage(cycles)

    # 3) 初始化深度老化环境
    state, info = env.reset(options={"soc_low": 0.8, "soc_high": 0.8})
    print(f"[测试] Cycle {cycles} 初始状态: SOC={info['SOC_pack']:.3f}, 初始电压={info['V_cell_max']:.3f}V")

    # 4) 施加大电流动作
    action = np.array([test_current], dtype=np.float32)
    for step_idx in range(1, 4):
        next_state, reward, terminated, truncated, next_info = env.step(action)    
        print(f"[测试] 施加 {test_current}A 后的第一步响应:")
        print(f"  -> 最大电压: {next_info['V_cell_max']:.4f} V")
        print(f"  -> 最高温度: {next_info['T_cell_max']:.2f} K")
        print(f"  -> 是否越限死掉: {next_info['violation']}")

        if next_info["violation"]:
            print("✅ 结论: 成功拦截！静态大电流策略在该老化阶段会失效。")
            break
    if not next_info['violation']:
        print("⚠️ 结论: 未越限，可考虑加大每周期增量或测试电流。")

    print("="*50 + "\n")

    # 5) 恢复原阶段，避免影响后续流程
    env.set_aging_stage(prev_stage)

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    config = load_config(args.config).data
    set_global_seed(config.get("seed"))

    env = build_pack_env(config["env"])
    test_aging_reasonableness(env, increment=0.0005, cycles=100, test_current=-15.0)
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
