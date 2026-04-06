# other_method 对比实验说明

本目录用于完成“多阶段恒流、遗传算法、基于静态代理模型的 TD3，以及 DDPG/PPO”在同一真实环境下的一回合对比。

## 目录结构

- `train_other_methods.py`：训练/生成所有对比方法，并把模型（或策略配置）统一落盘到同一个 `models/` 目录。
- `evaluate_on_real_env.py`：读取 `models/manifest.json`，把所有方法在真实环境逐个跑 1 回合，并导出轨迹和对比图。
- `common.py`：公共工具（轨迹采集、CSV 导出、多方法曲线绘图、分段恒流策略构造）。
- `run_full_experiment.py`：跨平台一键执行（Linux / macOS / Windows 都可用）。

---

## 方法定义

### 1) 多阶段恒流（`multistage_cc`）

默认策略按 SOC 分段输出固定电流：

- SOC ≤ 0.40：-20A
- 0.40 < SOC ≤ 0.70：-12A
- 0.70 < SOC ≤ 0.90：-7A
- SOC > 0.90：-3A

该策略会写入 `models/multistage_cc_schedule.json`。

### 2) 遗传算法分段恒流（`ga_cc`）

`train_other_methods.py` 中实现了一个轻量 GA：

- 个体：4 段 SOC 区间对应的 4 个恒流值。
- 适应度：`soc_end - 0.002*steps - 约束惩罚`。
- 进化：精英保留 + 交叉 + 高斯变异。

训练后得到 `models/ga_schedule.json`。

### 3) 基于静态代理模型的 RL（`td3` / `ddpg` / `ppo`）

- 直接复用仓库已有 `train_cycle0(...)` 流程：
  1. 真实环境采样。
  2. 静态代理模型训练。
  3. 在代理上训练策略网络。
- 输出：
  - `models/<algo>_static_gp/agent_ckpt.pt`
  - `models/<algo>_static_gp/static_surrogate.pt`

其中 `td3` 即“基于静态代理模型的 TD3”对比方法。

---

## 运行方式

> 建议先确认已有依赖（`requirements.txt`）和可用算力。

### A. 一键运行（跨平台，推荐）

```bash
python scripts/other_method/run_full_experiment.py configs/pack_3p6s_spme.yaml runs/other_method 7
```

### B. 分步运行

1. 训练/生成方法

```bash
python scripts/other_method/train_other_methods.py --config configs/pack_3p6s_spme.yaml --output runs/other_method --seed 7
```

2. 在真实环境跑一回合并对比

```bash
python scripts/other_method/evaluate_on_real_env.py --config configs/pack_3p6s_spme.yaml --manifest runs/other_method/models/manifest.json --output runs/other_method/eval --seed 7
```

---

## 输出说明

训练阶段会产出：

- `runs/other_method/models/manifest.json`：所有方法注册清单（类型、路径、算法）。
- `runs/other_method/models/*.json`：分段恒流策略定义。
- `runs/other_method/models/<algo>_static_gp/*.pt`：RL 模型和静态代理。

评估阶段会产出：

- `runs/other_method/eval/traj_<method>.csv`：每个方法一回合轨迹（时间、电流、SOC、最大电压、最大温度）。
- `runs/other_method/eval/real_env_comparison.png`：4 条指标对比图（充电电流、SOC、最大电压、最大温度）。
- `runs/other_method/eval/summary.json`：汇总指标（末端 SOC、峰值电压、峰值温度、平均电流等）。

---

## 可调参数建议

- `train_other_methods.py`
  - `--methods`：只跑子集方法（例如只跑 `td3 ddpg ppo`）。
  - `--population / --generations`：GA 搜索强度。
- `evaluate_on_real_env.py`
  - `--seed`：统一回合初始种子，保证方法间可比性。

---

## 备注

- 该目录代码默认将不同方法统一放到一个 `models/` 下，便于批量管理和后处理。
- 真实环境评估目前是“每种方法 1 回合”；如需统计稳定性，可在外层循环不同 seed，拼接多个 `summary.json` 后做均值和方差分析。