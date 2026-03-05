"""Episode rollout utilities for environments and surrogates."""
from __future__ import annotations

from functools import lru_cache
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

from ..envs.base_env import BasePackEnv
from ..rewards.paper_reward import PaperRewardConfig, reward_from_info, compute_paper_reward


def rollout_env(
    env: BasePackEnv,
    policy: Callable[[np.ndarray], np.ndarray],
    reward_cfg: PaperRewardConfig,
    *,
    hold_steps: int = 1,
    reset_options: Optional[Dict[str, float]] = None,
    action_low: Optional[float] = None,
    action_high: Optional[float] = None,
    use_physics_limit: bool = False,
) -> Tuple[float, List[Dict]]:
    """Rollout in a real environment."""
    
    @lru_cache(maxsize=1)
    def _get_chen2020_ocv_func():
        import pybamm
        from scipy.interpolate import interp1d

        param = pybamm.ParameterValues("Chen2020")
        U_n = param["Negative electrode OCP [V]"]
        U_p = param["Positive electrode OCP [V]"]
        try:
            sto_n_0 = param["Lower stoichiometric limit in negative electrode"]
            sto_n_1 = param["Upper stoichiometric limit in negative electrode"]
            sto_p_0 = param["Upper stoichiometric limit in positive electrode"]
            sto_p_1 = param["Lower stoichiometric limit in positive electrode"]
        except KeyError:
            sto_n_0, sto_n_1 = 0.0279, 0.9014
            sto_p_0, sto_p_1 = 0.9077, 0.2661

        soc_range = np.linspace(0, 1, 100)
        ocv_values: List[float] = []
        for soc in soc_range:
            curr_sto_n = sto_n_0 + soc * (sto_n_1 - sto_n_0)
            curr_sto_p = sto_p_0 - soc * (sto_p_0 - sto_p_1)
            v = param.evaluate(U_p(pybamm.Scalar(curr_sto_p))) - param.evaluate(U_n(pybamm.Scalar(curr_sto_n)))
            ocv_values.append(float(v))
        return interp1d(soc_range, ocv_values, kind="linear", fill_value="extrapolate")

    def _physics_limit_action(action_from_agent: float, info: Dict) -> float:
        soc_curr = float(info.get("SOC_cell_max", info.get("SOC_pack", 0.0)))
        try:
            # 1. 计算当前 OCV
            u_ocv = float(_get_chen2020_ocv_func()(soc_curr))
        except Exception:
            return float(action_from_agent)
        # 2. 设定物理边界：截止电压 4.2V，估算内阻 0.025 Ohm
        v_limit = 4.2
        r_internal = 0.025
        # 3. 计算物理安全电流边界 I_bound 
        i_bound_single = max(0.0, (v_limit - u_ocv) / r_internal)
        # 4. 适配 3p6s 电池组总电流 (充电为负值)
        i_bound_pack = i_bound_single * 3.0
        # 5. 动作裁剪：Agent 可以尝试物理极限内的任何电流，但禁止导致瞬时超压 [cite: 243, 272]
        return float(np.clip(action_from_agent, -i_bound_pack, 0.0))
    
    try:
        if reset_options is None:
            state, info = env.reset()
        else:
            state, info = env.reset(options=reset_options)

    except Exception as exc:  # noqa: BLE001 - reset 阶段兜底，避免初始化求解失败中断训练
        failed_info: Dict = {
            "t": 0.0,
            "I": 0.0,
            "I_prev": 0.0,
            "SOC_pack": 0.0,
            "std_SOC": 0.0,
            "V_cell_max": float(getattr(env, "v_max", 0.0)),
            "V_cell_min": 0.0,
            "T_cell_max": float(getattr(env, "t_max", 0.0)),
            "T_cell_min": 0.0,
            "violation": True,
            "terminated_reason": "reset_solver_error",
            "solver_error": repr(exc),
            "reward": -50.0,
        }
        print("episode_reset_failed:", failed_info["terminated_reason"], failed_info["solver_error"])
        return -50.0, [failed_info]

    infos: List[Dict] = [info]
    total_reward = 0.0
    prev_info = info
    done = False
    
    while not done:
        action_raw = policy(state)
        action_val = float(action_raw[0]) if np.ndim(action_raw) > 0 else float(action_raw)
        if action_low is not None and action_high is not None:
            action_val = float(np.clip(action_val, action_low, action_high))
        if use_physics_limit:
            action_val = _physics_limit_action(action_val, prev_info)
            if action_low is not None and action_high is not None:
                action_val = float(np.clip(action_val, action_low, action_high))

        action = np.array([action_val], dtype=np.float32)

        macro_reward = 0.0
        macro_info = prev_info
        episode_done = False
        for _ in range(max(1, int(hold_steps))):
            try:
                next_state, _, terminated, truncated, next_info = env.step(action)
            except Exception as exc:  # noqa: BLE001 - 兜底保护，防止单次仿真异常中断训练
                next_state = state
                terminated = False
                truncated = True
                next_info = dict(macro_info)
                next_info.update(
                    {
                        "I": action_val,
                        "terminated_reason": "rollout_exception",
                        "violation": True,
                        "solver_error": repr(exc),
                    }
                )

            step_reward = reward_from_info(macro_info, next_info, reward_cfg, env.v_max, env.t_max)
            macro_reward += step_reward
            state = next_state
            macro_info = next_info
            if terminated or truncated:
                episode_done = True
                break

        macro_info["reward"] = macro_reward
        infos.append(macro_info)
        total_reward += macro_reward
        prev_info = macro_info
        done = episode_done

    print(
        "episode_end:",
        "steps=", len(infos)-1,
        "R=", round(total_reward, 2),
        "SOC_end=", round(infos[-1]["SOC_pack"], 4),
        "Vmax=", round(max(i["V_cell_max"] for i in infos), 4),
        "Tmax=", round(max(i["T_cell_max"] for i in infos), 2),
        "viol=", infos[-1].get("violation", None),
        "reason=", infos[-1].get("terminated_reason", None),
        "I_mean=", round(np.mean([i["I"] for i in infos[1:]]), 2) if len(infos) > 1 else None,
    )

    return total_reward, infos


def rollout_surrogate(
    state: np.ndarray,
    surrogate: Callable, 
    policy: Callable,
    horizon: int,
    reward_cfg: PaperRewardConfig,
    dt: float,
    v_max: float,
    t_max: float
) -> Tuple[float, List[Dict]]:
    """
    用静态代理模型进行 rollout：
    - 状态 7 维: [SOCmean, stdSOC, Vmax, dV, Tmax, Tmin, Iprev]
    - surrogate 预测 delta 6 维（不含 Iprev）
    - next_state[:6] = state[:6] + delta6
    - next_state[6] = action (近似 I_true)
    同时加入不确定性悲观惩罚（Hao et al. 风格）。
    """
    # 状态索引定义 
    IDX_SOC = 0
    IDX_STD_SOC = 1
    IDX_VMAX = 2
    IDX_DV = 3
    IDX_TMAX = 4
    IDX_TMIN = 5
    IDX_IPREV = 6
    # ---------- delta 索引----------
    D_VMAX = 2

    curr_state = state.copy()
    total_reward = 0.0
    infos = []
    
    # 初始化 info
    curr_info = {
        "t": 0.0,
        "SOC_pack": float(curr_state[IDX_SOC]),
        "std_SOC": float(curr_state[IDX_STD_SOC]),
        "V_cell_max": float(curr_state[IDX_VMAX]),
        "dV": float(curr_state[IDX_DV]),
        "T_cell_max": float(curr_state[IDX_TMAX]),
        "T_cell_min": float(curr_state[IDX_TMIN]),
        "I": float(curr_state[IDX_IPREV]),       # 当前步采用的电流（这里等于初始 Iprev）
        "I_prev": float(curr_state[IDX_IPREV]),  # 明确写出口径
        "reward": 0.0,
        "violation": False,
        "viol_hard": False,
        "is_risky": False,
        "terminated_reason": "reset",
    }
    BASE_TERMS = {
        "delta_soc": 0.0,
        "v_soft": False,
        "i_abs": 0.0,
        "r_soc": 0.0,
        "r_track": 0.0,
        "r_finish": 0.0,
        "r_v": 0.0,
        "r_t": 0.0,
        "r_const": 0.0,
        "r_action": 0.0,
        "r_time": 0.0,
    }
    curr_info.update(BASE_TERMS)
    infos.append(curr_info)

    for k in range(horizon):
        # 1. 策略动作
        action = policy(curr_state) 
        a_val = float(action[0])
        
        # 2. 预测 (Mean, Std)
        delta_mean, delta_std = surrogate(curr_state, action)
       
        # 3. 状态更新 只更新前6维；Iprev=action）
        next_state = curr_state.copy()
        # 强制 SOC (索引0) 的增量非负
        # 这样即使代理模型预测了负值，物理层也会将其修正为 0
        delta_mean[0] = max(0.0, delta_mean[0])
        next_state[:6] = curr_state[:6] + delta_mean
        next_state[IDX_IPREV] = a_val

        # 4. 安全检查 (Hao et al. 核心)
        v_mean = next_state[IDX_VMAX]
        v_std = delta_std[D_VMAX]
        v_risk = v_mean + 3.0 * v_std
        
        # 硬约束检查 (Hard Constraint)
        viol_hard = (next_state[IDX_VMAX] > v_max) or (next_state[IDX_TMAX] > t_max)
        is_risky = (v_risk > v_max)
        
        violation = bool(viol_hard)

        # 5. 计算奖励
        r_phys,r_soc ,r_time ,r_v , r_t , r_const , r_action = compute_paper_reward(
            soc_prev=curr_state[IDX_SOC],
            soc_next=next_state[IDX_SOC],
            v_max_next=next_state[IDX_VMAX],
            t_max_next=next_state[IDX_TMAX],
            std_soc_next=next_state[IDX_STD_SOC], # 参数名对应 paper_reward定义
            action_current=a_val,                 # 传入当前动作用于惩罚
            v_limit=v_max,
            t_limit=t_max,
            config=reward_cfg
        )

        # 6. 【关键修复】补充代理模型独有的不确定性风险惩罚
        risk_penalty = 0.0
        # 如果预测均值没超，但考虑了模型的不确定性 (mean + 3*std) 后超标了
        if is_risky and not viol_hard:
            # 说明 Agent 正在利用模型不自信的区域，必须给予足够痛的惩罚
            # 这个值建议设置得比单步 r_soc 的正常收益大，比如 -20.0
            risk_penalty = -20.0
        
       

        step_reward = r_phys + risk_penalty
        # if v_mean > 4.15: # 软约束阈值
        #     # 随着电压接近 4.2，惩罚呈指数增长
        #     # 4.15V -> -0.2
        #     # 4.18V -> -2.0
        #     # 4.20V -> -20.0 (加上下面的 safety_penalty，总计 -70)
        #     barrier_penalty = -2.0 * np.exp(30.0 * (v_mean - 4.18))
        
        # # 违规惩罚 (Violation Penalty)
        # safety_penalty = 0.0
        # if violation:
        #     safety_penalty = -50.0 # 给予重罚

        # step_reward = r_phys + barrier_penalty + safety_penalty
        total_reward += step_reward

        # 6) violation 与终止条件（尽量贴近真实环境口径）
        soc_done = float(next_state[IDX_SOC]) >= 0.80

        terminated = soc_done or viol_hard or (k + 1 >= horizon)
        
        if violation:
            reason = "violation"
        elif soc_done:
            reason = "soc_full"
        elif k + 1 >= horizon:
            reason = "horizon"
        else:
            reason = "running"

        # 加入soc变化量
        delta_soc = next_state[IDX_SOC] - curr_state[IDX_SOC]
        v_soft = next_state[IDX_VMAX] > (v_max - 0.05)
        infos.append({
            "t": float((k + 1) * dt),
            "SOC_pack": float(next_state[IDX_SOC]),
            "std_SOC": float(next_state[IDX_STD_SOC]),
            "V_cell_max": float(next_state[IDX_VMAX]),
            "dV": float(next_state[IDX_DV]),
            "T_cell_max": float(next_state[IDX_TMAX]),
            "T_cell_min": float(next_state[IDX_TMIN]),
            "I": a_val,            
            "I_prev": float(curr_state[IDX_IPREV]),   
            "reward": step_reward,  
            "violation": violation,
            "viol_hard": bool(viol_hard),
            "is_risky": bool(is_risky),
            "v_risk": float(v_risk),
            "terminated_reason": reason,
            "delta_soc": delta_soc,
            "v_soft": v_soft,
            "i_abs": float(abs(a_val)),    
        })
        
        curr_state = next_state
        if terminated:
            break
        # if len(infos) < 12:
        #     print("a_val", a_val)
    return total_reward, infos
