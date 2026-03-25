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

# 手动修改这里来控制从哪个老化阶段开始训练（包含该阶段）
START_AGING_STAGE = 30
# 手动修改这里来控制训练到哪个老化阶段结束（包含该阶段）
END_AGING_STAGE = 30
def _load_network_weights_only(agent: object, ckpt_path: Path) -> None:
    """Load model weights only (no replay/optimizer/update_step) for adaptation experiments."""
    ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)

    # TD3
    for key, attr in [
        ("actor", "actor"),
        ("actor_target", "actor_target"),
        ("critic1", "critic1"),
        ("critic2", "critic2"),
        ("critic1_target", "critic1_target"),
        ("critic2_target", "critic2_target"),
        # DDPG
        ("target_actor", "target_actor"),
        ("target_critic", "target_critic"),
        ("critic", "critic"),
        # PPO/common
        ("actor", "actor"),
    ]:
        module = getattr(agent, attr, None)
        if module is not None and key in ckpt:
            module.load_state_dict(ckpt[key])

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
    
    adaptive_cfg = AdaptiveConfig(**config["trainer"]["adaptive"])
    adaptive_cfg.cycles = 1

    initial_ckpt = Path(config["logging"]["runs_dir"]) / "cycle0" / "agent_ckpt.pt"
    if not initial_ckpt.exists():
        raise FileNotFoundError(f"Missing checkpoint: {initial_ckpt}")
    
    # agent = build_agent_from_config(
    #     state_dim=env.observation_space.shape[0],
    #     action_dim=1,
    #     rl_config=config["rl"],
    # )
    # agent.load(str(initial_ckpt))

    previous_agent_ckpt = initial_ckpt
    if START_AGING_STAGE > 1:
        resume_ckpt = (
            Path(config["logging"]["runs_dir"])
            / f"adaptive/adaptive_cycle{START_AGING_STAGE - 1}"
            / "agent_ckpt.pt"
        )
        if not resume_ckpt.exists():
            raise FileNotFoundError(
                "Missing resume checkpoint for requested start stage: "
                f"{resume_ckpt}. Run earlier stages first or set START_AGING_STAGE = 1."
            )
        previous_agent_ckpt = resume_ckpt
    for aging_stage in range(START_AGING_STAGE, END_AGING_STAGE + 1):
        agent = build_agent_from_config(
            state_dim=env.observation_space.shape[0],
            action_dim=1,
            rl_config=config["rl"],
        )
        _load_network_weights_only(agent, previous_agent_ckpt)
        
        if hasattr(env, "set_aging_stage"):
            env.set_aging_stage(aging_stage)

        diff_surrogate = DifferentialSurrogate(
            input_dim=env.observation_space.shape[0] + 1,
            output_dim=env.observation_space.shape[0] - 1,
            hidden_sizes=config["surrogate"]["hidden_sizes"],
            ensemble_size=config["surrogate"]["ensemble_size"],
            lr=config["surrogate"]["learning_rate"],
        )
        combined = CombinedSurrogate(static_surrogate, diff_surrogate)

        stage_run_dir = ensure_dir(Path(config["logging"]["runs_dir"]) / f"adaptive/adaptive_cycle{aging_stage}")

        train_adaptive_cycles(
            env,
            agent,
            static_surrogate,
            diff_surrogate,
            combined,
            reward_cfg,
            adaptive_cfg,
            str(stage_run_dir),
            soh_enabled=config["soh_prior"]["enabled"],
            lambda_prior=config["soh_prior"]["lambda_prior"],
            theta_dim=config["soh_prior"]["theta_dim"],
            dummy_soh=config["soh_prior"]["dummy_soh"],
        )

        agent_ckpt = stage_run_dir / "agent_ckpt.pt"
        if hasattr(agent, "save"):
            agent.save(str(agent_ckpt))
            previous_agent_ckpt = agent_ckpt

        torch.save(diff_surrogate, stage_run_dir / "diff_surrogate.pt")
        print(f"Completed adaptive cycle {aging_stage}, results saved to {stage_run_dir}")


if __name__ == "__main__":
    main()
