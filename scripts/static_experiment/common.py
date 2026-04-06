"""静态代理实验通用工具。"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Tuple

import csv
import sys
import numpy as np
import torch

current_dir = Path(__file__).resolve().parent
root_dir = current_dir.parent.parent
if str(root_dir) not in sys.path:
    sys.path.append(str(root_dir))

from src.envs.liionpack_spme_pack_env import build_pack_env
from src.evaluation.episode_rollout import rollout_surrogate
from src.rewards.paper_reward import PaperRewardConfig, reward_from_info, compute_paper_reward
from src.rl.agent_factory import build_agent_from_config
from src.rl.noise import GaussianNoise
from src.rl.trainers.trainer_static_gp import Cycle0Config, get_chen2020_ocv_func
from src.surrogate.dataset import build_dataset
from src.surrogate.gp_static import StaticSurrogate
from src.utils.config import load_config
from src.utils.seeds import set_global_seed


@dataclass
class EpisodeMetrics:
    episode: int
    total_reward: float
    charge_time_s: float
    voltage_violation: float
    temperature_violation: float
    sim_crash: int = 0


def policy_with_noise(agent: Any, sigma: float, low: float, high: float) -> Callable[[np.ndarray], np.ndarray]:
    def _policy(state: np.ndarray) -> np.ndarray:
        action = float(agent.act(state)[0])
        if sigma > 0:
            action += float(np.random.normal(0.0, sigma))
        action = float(np.clip(action, low, high))
        return np.array([action], dtype=np.float32)

    return _policy


def _info_to_state(info: Dict[str, Any], fallback: np.ndarray) -> np.ndarray:
    return np.array([
        float(info.get("SOC_pack", fallback[0])),
        float(info.get("std_SOC", fallback[1])),
        float(info.get("V_cell_max", fallback[2])),
        float(info.get("dV", fallback[3])),
        float(info.get("T_cell_max", fallback[4])),
        float(info.get("T_cell_min", fallback[5])),
        float(info.get("I_prev", fallback[6])),
    ], dtype=np.float32)


def _episode_stats(infos: List[Dict[str, Any]], env, fallback_time_s: float = 0.0) -> Tuple[float, float, float]:
    if not infos:
        return float(fallback_time_s), 0.0, 0.0
    charge_time_s = float(infos[-1].get("t", fallback_time_s))
    vmax = max(float(x.get("V_cell_max", env.v_max)) for x in infos)
    tmax = max(float(x.get("T_cell_max", env.t_max)) for x in infos)
    return charge_time_s, float(vmax - env.v_max), float(tmax - env.t_max)


def rollout_real_episode(
    env,
    agent,
    reward_cfg: PaperRewardConfig,
    sigma: float = 0.0,
    crash_penalty: float = -5.0,
) -> Tuple[float, List[Dict], List[Tuple[np.ndarray, np.ndarray, np.ndarray, bool]], bool, str]:
    transitions: List[Tuple[np.ndarray, np.ndarray, np.ndarray, bool]] = []
    infos: List[Dict[str, Any]] = []
    total_reward = 0.0
    crash_msg = ""

    try:
        state, info = env.reset()
    except Exception as exc:  # noqa: BLE001
        return float(crash_penalty), infos, transitions, True, f"reset_crash: {exc}"

    infos = [info]
    done = False
    policy = policy_with_noise(agent, sigma, float(agent.config.action_low), float(agent.config.action_high))

    while not done:
        action = policy(state)
        try:
            next_state, _, terminated, truncated, next_info = env.step(action)
        except Exception as exc:  # noqa: BLE001
            total_reward += float(crash_penalty)
            crash_msg = f"step_crash: {exc}"
            if transitions:
                s0, a0, n0, _ = transitions[-1]
                transitions[-1] = (s0, a0, n0, True)
            crash_info = dict(info)
            crash_info["reward"] = float(crash_penalty)
            crash_info["terminated_reason"] = "sim_crash"
            crash_info["sim_crash"] = True
            infos.append(crash_info)
            return total_reward, infos, transitions, True, crash_msg

        reward = reward_from_info(info, next_info, reward_cfg, env.v_max, env.t_max)
        next_info["reward"] = reward
        done_flag = bool(terminated or truncated)
        agent.observe(state, action, reward, next_state, done_flag)
        transitions.append((state.copy(), action.copy(), next_state.copy(), done_flag))

        total_reward += reward
        infos.append(next_info)
        state = next_state
        info = next_info
        done = done_flag

    return total_reward, infos, transitions, False, crash_msg


def run_real_training(
    env,
    agent,
    reward_cfg: PaperRewardConfig,
    episodes: int,
    updates_per_step: int = 1,
    sigma: float = 0.3,
    crash_penalty: float = -5.0,
) -> Tuple[List[EpisodeMetrics], List[Tuple[np.ndarray, np.ndarray, np.ndarray]]]:
    metrics: List[EpisodeMetrics] = []
    dataset: List[Tuple[np.ndarray, np.ndarray, np.ndarray]] = []

    for ep in range(1, episodes + 1):
        ep_reward, infos, transitions, crashed, crash_msg = rollout_real_episode(
            env,
            agent,
            reward_cfg,
            sigma=sigma,
            crash_penalty=crash_penalty,
        )
        if crashed:
            print(f"[WARN] real episode {ep} crashed and skipped. penalty={crash_penalty}, detail={crash_msg}")

        for _ in range(max(1, updates_per_step) * len(transitions)):
            agent.update()

        for s, a, n, _ in transitions:
            dataset.append((s[:], a[:], (n[:6] - s[:6]).copy()))

        charge_time_s, v_vio, t_vio = _episode_stats(infos, env)
        metrics.append(EpisodeMetrics(ep, float(ep_reward), charge_time_s, v_vio, t_vio, int(crashed)))
    return metrics, dataset


def run_real_training_collect_style(
    env,
    agent,
    reward_cfg: PaperRewardConfig,
    episodes: int,
    hold_steps: int = 1,
    eps_random_start: float = 0.85,
    eps_random_end: float = 0.25,
    noise_sigma_start: float = 0.20,
    noise_sigma_end: float = 0.05,
) -> Tuple[List[EpisodeMetrics], List[Tuple[np.ndarray, np.ndarray, np.ndarray]]]:
    """真实环境训练采用 train_cycle0/collect_real_data 同一套交互逻辑。"""
    low = float(agent.config.action_low)
    high = float(agent.config.action_high)
    cfg = Cycle0Config(
        real_episodes=episodes,
        surrogate_epochs=1,
        policy_epochs=1,
        hold_steps=hold_steps,
        eps_random_start=eps_random_start,
        eps_random_end=eps_random_end,
        noise_sigma_start=noise_sigma_start,
        noise_sigma_end=noise_sigma_end,
    )
    v_soft = float(getattr(cfg, "v_soft_max", env.v_max - 0.03))
    t_soft = float(getattr(cfg, "t_soft_max", env.t_max - 1.5))
    get_ocv = get_chen2020_ocv_func()

    def physics_limit_action(action_from_agent: float, info: dict) -> float:
        soc_curr = float(info.get("SOC_cell_max", info.get("SOC_pack")))
        _ = v_soft, t_soft
        u_ocv = float(get_ocv(soc_curr))
        v_limit = 4.2
        r_internal = 0.025
        i_bound_single = max(0.0, (v_limit - u_ocv) / r_internal)
        i_bound_pack = i_bound_single * 3.0
        safe_action = np.clip(action_from_agent, -i_bound_pack, 0.0)
        return float(safe_action)

    metrics: List[EpisodeMetrics] = []
    dataset: List[Tuple[np.ndarray, np.ndarray, np.ndarray]] = []
    crash_penalty = -5.0
    for ep in range(episodes):
        frac = ep / max(1, episodes - 1)
        sigma = ((1 - frac) * cfg.noise_sigma_start + frac * cfg.noise_sigma_end) * (high - low)
        noise = GaussianNoise(sigma=float(sigma))

        try:
            state, info = env.reset(options={"soc_low": 0.1, "soc_high": 0.8})
        except Exception as exc:  # noqa: BLE001
            print(f"[WARN] real collect-style reset failed at ep={ep + 1}: {exc}")
            metrics.append(EpisodeMetrics(ep + 1, float(crash_penalty), 0.0, 0.0, 0.0, 1))
            continue

        done = False
        crashed = False
        crash_msg = ""
        ep_reward = 0.0
        current_soc = float(info.get("SOC_pack", state[0]))
        infos: List[Dict[str, Any]] = [dict(info)]

        while not done:
            raw_action = float(agent.act(state)[0])
            eps_random = (1.0 - frac) * cfg.eps_random_start + frac * cfg.eps_random_end
            if np.random.rand() < eps_random:
                raw_action = float(np.random.uniform(low, high))
            a_noisy = raw_action + noise.sample()
            a_clipped = float(np.clip(a_noisy, low, high))
            safe_action = physics_limit_action(a_clipped, info)
            safe_action = float(np.clip(safe_action, low, high))
            action_to_exec = np.array([safe_action], dtype=np.float32)

            start_state = state.copy()
            accumulated_reward = 0.0
            for _ in range(cfg.hold_steps):
                prev_soc = current_soc
                try:
                    next_state, _, terminated, truncated, next_info = env.step(action_to_exec)
                except Exception as exc:  # noqa: BLE001
                    crashed = True
                    crash_msg = f"step failed: {exc}"
                    accumulated_reward += float(crash_penalty)
                    done = True
                    crash_info = dict(info)
                    crash_info["reward"] = float(crash_penalty)
                    crash_info["terminated_reason"] = "sim_crash"
                    crash_info["sim_crash"] = True
                    infos.append(crash_info)
                    break
                current_soc = float(next_info["SOC_pack"])
                v_max = float(next_info["V_cell_max"])
                t_max = float(next_info["T_cell_max"])
                std_soc = float(next_info.get("std_SOC", 0.0))
                i_exec = float(next_info.get("I_pack_true", safe_action))

                r_step, _, _, _, _, _, _ = compute_paper_reward(
                    soc_prev=prev_soc,
                    soc_next=current_soc,
                    v_max_next=v_max,
                    t_max_next=t_max,
                    std_soc_next=std_soc,
                    action_current=i_exec,
                    v_limit=env.v_max,
                    t_limit=env.t_max,
                    config=reward_cfg,
                )
                accumulated_reward += r_step
                state = next_state
                info = next_info
                infos.append(dict(next_info))
                if terminated or truncated:
                    done = True
                    break

            agent.observe(start_state, action_to_exec, accumulated_reward, state, done)
            if len(getattr(agent, "buffer", [])) >= int(getattr(agent.config, "batch_size", 1)):
                agent.update()

            final_delta = state[:6] - start_state[:6]
            dataset.append((start_state.copy(), action_to_exec.copy(), final_delta.copy()))
            ep_reward += accumulated_reward
            if crashed:
                print(f"[WARN] real collect-style episode {ep + 1} crashed. {crash_msg}")
                break

        charge_time_s, v_vio, t_vio = _episode_stats(infos, env)
        metrics.append(EpisodeMetrics(ep + 1, float(ep_reward), charge_time_s, v_vio, t_vio, int(crashed)))

    return metrics, dataset


def run_surrogate_training(
    env,
    agent,
    surrogate: StaticSurrogate,
    reward_cfg: PaperRewardConfig,
    episodes: int,
    rollouts_per_episode: int = 3,
    updates_per_episode: int = 50,
    crash_penalty: float = -5.0,
) -> List[EpisodeMetrics]:
    metrics: List[EpisodeMetrics] = []

    for ep in range(1, episodes + 1):
        ep_reward, ep_time, ep_vio, ep_tvio = [], [], [], []
        crash_cnt = 0
        for ridx in range(rollouts_per_episode):
            try:
                if ridx == 0:
                    state0, _ = env.reset()
                else:
                    state0, _ = env.reset(seed=ep * 1000 + ridx)
            except Exception as exc:  # noqa: BLE001
                crash_cnt += 1
                ep_reward.append(float(crash_penalty))
                ep_time.append(0.0)
                ep_vio.append(0.0)
                ep_tvio.append(0.0)
                print(f"[WARN] surrogate init reset failed at ep={ep}, rollout={ridx}: {exc}")
                continue

            total_reward, infos = rollout_surrogate(
                state=state0,
                surrogate=surrogate.predict,
                policy=policy_with_noise(agent, 0.2, float(agent.config.action_low), float(agent.config.action_high)),
                horizon=env.max_steps,
                reward_cfg=reward_cfg,
                dt=env.dt,
                v_max=env.v_max,
                t_max=env.t_max,
            )

            for t in range(len(infos) - 1):
                s = np.array([
                    infos[t]["SOC_pack"], infos[t]["std_SOC"], infos[t]["V_cell_max"], infos[t]["dV"], infos[t]["T_cell_max"], infos[t]["T_cell_min"], infos[t]["I_prev"],
                ], dtype=np.float32)
                s_next = np.array([
                    infos[t + 1]["SOC_pack"], infos[t + 1]["std_SOC"], infos[t + 1]["V_cell_max"], infos[t + 1]["dV"], infos[t + 1]["T_cell_max"], infos[t + 1]["T_cell_min"], infos[t + 1]["I_prev"],
                ], dtype=np.float32)
                a = np.array([float(infos[t + 1]["I"])], dtype=np.float32)
                r = float(infos[t + 1]["reward"])
                done = bool(infos[t + 1]["violation"] or (t == len(infos) - 2))
                agent.observe(s, a, r, s_next, done)

            ep_reward.append(total_reward)
            ep_time.append(float(infos[-1].get("t", len(infos) * env.dt)))
            ep_vio.append(max(float(x["V_cell_max"]) for x in infos) - env.v_max)
            ep_tvio.append(max(float(x["T_cell_max"]) for x in infos) - env.t_max)

        for _ in range(updates_per_episode):
            agent.update()

        if not ep_reward:
            ep_reward, ep_time, ep_vio, ep_tvio = [float(crash_penalty)], [0.0], [0.0], [0.0]
            crash_cnt = max(1, crash_cnt)
        metrics.append(EpisodeMetrics(ep, float(np.mean(ep_reward)), float(np.mean(ep_time)), float(np.mean(ep_vio)), float(np.mean(ep_tvio)), int(crash_cnt > 0)))

    return metrics


def evaluate_policy_trajectory(env, agent, seed: int) -> Dict[str, np.ndarray]:
    try:
        s, info = env.reset(seed=seed)
    except Exception as exc:  # noqa: BLE001
        print(f"[WARN] evaluate trajectory reset failed: {exc}")
        return {
            "time_s": np.asarray([0.0], dtype=np.float32),
            "current_a": np.asarray([0.0], dtype=np.float32),
            "soc": np.asarray([np.nan], dtype=np.float32),
            "voltage_v": np.asarray([np.nan], dtype=np.float32),
            "temperature_k": np.asarray([np.nan], dtype=np.float32),
        }

    cur = [float(info.get("I_pack_true", s[6]))]
    soc = [float(info.get("SOC_pack", s[0]))]
    vol = [float(info.get("V_cell_max", s[2]))]
    temp = [float(info.get("T_cell_max", s[4]))]
    ts = [0.0]
    done = False
    while not done:
        a = agent.act(s)
        try:
            s, _, term, trunc, info = env.step(a)
        except Exception as exc:  # noqa: BLE001
            print(f"[WARN] evaluate trajectory step failed: {exc}")
            break
        cur.append(float(info.get("I_pack_true", float(a[0]))))
        soc.append(float(info.get("SOC_pack", s[0])))
        vol.append(float(info.get("V_cell_max", s[2])))
        temp.append(float(info.get("T_cell_max", s[4])))
        ts.append(float(info.get("t", ts[-1] + env.dt)))
        done = bool(term or trunc)
    return {
        "time_s": np.asarray(ts),
        "current_a": np.asarray(cur),
        "soc": np.asarray(soc),
        "voltage_v": np.asarray(vol),
        "temperature_k": np.asarray(temp),
    }


def save_metrics_csv(metrics: List[EpisodeMetrics], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["episode", "total_reward", "charge_time_s", "voltage_violation", "temperature_violation", "sim_crash"])
        for m in metrics:
            writer.writerow([m.episode, m.total_reward, m.charge_time_s, m.voltage_violation, m.temperature_violation, m.sim_crash])


def build_env_agent_reward(config_path: Path, seed: int):
    cfg = load_config(str(config_path)).data
    set_global_seed(seed)
    env = build_pack_env(cfg["env"])
    agent = build_agent_from_config(env.observation_space.shape[0], 1, cfg["rl"])
    reward_cfg = PaperRewardConfig(**cfg["reward"])
    return cfg, env, agent, reward_cfg


def fit_static_surrogate(env, transitions: List[Tuple[np.ndarray, np.ndarray, np.ndarray]], cfg: Dict[str, Any], epochs: int) -> StaticSurrogate:
    surrogate = StaticSurrogate(
        input_dim=env.observation_space.shape[0] + 1,
        output_dim=env.observation_space.shape[0] - 1,
        hidden_sizes=cfg["surrogate"]["hidden_sizes"],
        ensemble_size=cfg["surrogate"]["ensemble_size"],
        lr=cfg["surrogate"]["learning_rate"],
    )
    if len(transitions) == 0:
        print("[WARN] no real transitions collected (likely due simulator crashes). Use one neutral fallback transition to keep flow running.")
        s = np.zeros(env.observation_space.shape[0], dtype=np.float32)
        a = np.array([0.0], dtype=np.float32)
        d = np.zeros(env.observation_space.shape[0] - 1, dtype=np.float32)
        transitions = [(s, a, d)]
    dataset = build_dataset(transitions)
    surrogate.fit(dataset, epochs=epochs)
    return surrogate