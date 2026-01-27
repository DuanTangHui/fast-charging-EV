# Battery Fast-Charge Pack MBRL

This repository provides a runnable, paper-style project scaffold inspired by
"Adaptive Model-Based Reinforcement Learning for Fast-Charging Optimization of
Lithium-Ion Batteries" and extended to a 3P6S pack with aging, cell heterogeneity,
and an external SOH prior (STAPINN-like) for fast online calibration.

## Highlights
- Pack-level environment with max-cell voltage/temperature constraints.
- Static, differential, and combined surrogate models (NN-backed for now).
- DDPG actor-critic with replaceable trainer logic for Cycle0/Adaptive cycles.
- SOH prior pipeline with a fast calibration stage that regularizes to prior.
- Fully runnable scripts with toy backend when liionpack/pybamm are unavailable.

## Quick Start
```bash
pip install -r requirements.txt
python scripts/train_cycle0_build_static_gp.py --config configs/pack_3p6s_spme.yaml
python scripts/train_adaptive_cycles.py --config configs/pack_3p6s_spme_with_soh_prior.yaml
python scripts/evaluate_policy.py --config configs/pack_3p6s_spme.yaml --ckpt runs/cycle0/policy.pt
```

## Project Layout
See the config files in `configs/` for key hyperparameters, sampling step,
reward weights, and environment settings.

## 1.从脚本入口开始

-`scripts/train_cycle0_build_static_gp.py`：跑 Cycle0（真实环境采样 → 训练静态代理 → 代理上训练策略）。这是最核心的训练入口。

-`scripts/train_adaptive_cycles.py`：跑论文 Algorithm 2 的自适应循环（少量真实采样 → 更新差分代理 → 组合代理上训练）。

-`scripts/evaluate_policy.py`：加载策略并评估/画图。

## 2.环境与观测（相当于“仿真器”）

-环境在 `src/envs/liionpack_spme_pack_env.py`：包含 pack 级别的 SOC/V/T 统计量与动作电流 I，默认 toy fallback 也能跑。

-观测构造在 `src/envs/observables.py`：将 cell-level 汇总为 pack-level 的 max/min/std 等。

## 3.奖励函数（严格贴论文式(24)）

`src/rewards/paper_reward.py`：SOC 增益、时间惩罚、电压越界惩罚、温度越界惩罚都在这里计算。

## 4.代理模型（核心创新点之一）

-静态代理 StaticSurrogate（G0）在 `src/surrogate/gp_static.py`。

-差分代理 DifferentialSurrogate（G^）在 `src/surrogate/gp_differential.py`（仍保留接口），组合代理 CombinedSurrogate 在 src/surrogate/gp_combined.py。

-底层用 NN ensemble (`src/surrogate/nn_delta_model.py`) 输出均值+不确定性。

-训练数据与归一化逻辑在 `src/surrogate/dataset.py`（你遇到的 overflow 已修复）。

## 5.SOH 先验与快速标定

-SOH Predictor（目前 Dummy）在 `src/soh_prior/stapinn_predictor.py`，特征提取在    `src/soh_prior/feature_extraction.py `，SOH→参数先验在 `src/soh_prior/soh2param_mapper.py`。

-快速标定在 `src/calibration/fast_calibrator.py`，用于对齐先验并更新老化参数。


## 思路
我现在想做一个电池组的快速充电策略优化研究。 第一点：想利用liionpack的spme模型搭建一个3P6S 电池组老化电池的真实环境，根据文章中的方法，我打算利用神经网络或高斯过程方法来拟合一个代理模型，以此提升后续强化学习的训练过程。 第二点：前期的工作做电池组的soh估计，将电池组看作一个整体，输入充电过程中一些特征的变化曲线做soh预测，想将此前的研究融入到快速充电策略优化研究中。我的想法是建立一个简单的映射关系（或者一个小网络），将 STAPINN 输出的 SOH 值映射为差分模型的初始参数猜测（Initial Guess）。这样，差分模型只需要极少量的真实数据（比如 1-2 个充电片段）就能迅速收敛。SOH 模型在这里起到了**“先验知识提供者（Prior Knowledge Provider）”**的作用。 第三点 强化学习方法的选择，奖励函数的设计，环境的状态定义，动作定义，初步思路要考虑电池组的单体电池间的差异，所以需要一些表征差异的状态，不能单纯是soc、v、T。