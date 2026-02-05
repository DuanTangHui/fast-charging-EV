"""
观测构造与曲线提取工具。

- build_observation: 将 cell 级数组聚合为 PackObservation（RL 输入）
- curve_from_infos:  将每步 info 列表转为“可画曲线”的 dict
"""

from __future__ import annotations

from typing import Dict, Sequence

import numpy as np

from .base_env import PackObservation

"""
    构造 RL 状态量（7维）：
    - SOCmean: 能量水平
    - std_SOC: 容量不一致性
    - V_cell_max: 过充安全边界
    - dV: 电压离散度（末端更敏感）
    - T_cell_max: 热安全
    - T_cell_min: 低温析锂预警
    - I_prev: 上一步真实执行电流（I_pack_true），帮助区分 OCV vs 极化/IR
"""
def build_observation(
    soc: np.ndarray,
    voltage: np.ndarray,
    temperature: np.ndarray,
    soh: np.ndarray,
    I_pack_true: float,
) -> PackObservation:
    """
    将 cell 级数组聚合为 pack 级观测。

    参数：
    - soc:         (n_cells,) 每个 cell 的 SOC
    - voltage:     (n_cells,) 每个 cell 的端电压
    - temperature: (n_cells,) 每个 cell 的温度
    - soh:         (n_cells,) 每个 cell 的 SOH（目前不进观测，但保留接口方便扩展）
    - i_prev:      上一步电流（真实执行电流）
    
    返回：
    - PackObservation：包含 pack 平均 SOC、电压/温度极值、不一致性指标等
    """
    # pack 平均 SOC
    soc_pack = float(np.mean(soc))

    # 电压统计
    v_max = float(np.max(voltage))
    v_min = float(np.min(voltage))

    # 温度统计
    t_max = float(np.max(temperature))
    t_min = float(np.min(temperature))

    # 注意：soh 当前未直接使用，但先留着接口，方便未来把 SOH_pack 等加入观测向量
    _ = soh  # 显式占位，避免“未使用变量”警告（你也可以直接删掉这个变量）

    return PackObservation(
        SOC_pack=soc_pack,
        std_SOC=float(np.std(soc)),
        V_cell_max=v_max,
        dV=v_max - v_min,
        T_cell_max=t_max,
        T_cell_min=t_min,
        I_prev=float(I_pack_true),
    )


def curve_from_infos(infos: Sequence[Dict]) -> Dict[str, list[float]]:
    """
    将一段 episode 的 info 列表，转换为绘图用的曲线数据。

    约定：
    - infos: 每一步 env.step 返回的 info 字典组成的列表
    - 输出 dict 的每个 key 对应一条曲线（list[float]），长度=步数

    注意：
    - 如果某个 info 缺少某个字段，会用 0.0 兜底（避免 KeyError）
    - 你可以根据项目需要扩展 curves keys，例如加入 I_pack_true、I_cell_std 等
    """
    curves: Dict[str, list[float]] = {
        "t": [],
        "I": [],
        "SOC_pack": [],
        "V_cell_max": [],
        "V_cell_min": [],
        "T_cell_max": [],
        "T_cell_min": [],
        "dV": [],
        "dT": [],
        "reward": [],
        "r_soc": [],
        "r_v": [],
        "r_action": [],

    }

    # infos[0] 是 reset 行，不包含真正的 transition
    for info in infos[1:]:
        for key in curves:
            # info中没有dT和V_cell_min，需要特殊处理
            if key == "dT":
                t_max = float(info.get("T_cell_max", 0.0))
                t_min = float(info.get("T_cell_min", 0.0))
                curves["dT"].append(t_max - t_min)
            elif key == "V_cell_min":
                v_max = float(info.get("V_cell_max", 0.0))
                dV = float(info.get("dV", 0.0))
                curves["V_cell_min"].append(v_max - dV)
            else:
                curves[key].append(float(info.get(key, 0.0)))

    return curves
