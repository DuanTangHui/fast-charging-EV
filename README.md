# Battery Fast-Charge Pack MBRL

本项目是一个可运行的“论文风格”电池快充优化框架，参考论文 *Adaptive Model-Based Reinforcement Learning for Fast-Charging Optimization of Lithium-Ion Batteries*，并扩展到：

- 3P6S 电池包场景（Liionpack + PyBaMM SPME）
- 单体不一致性与老化
- 静态/差分/组合代理模型
- 可切换 RL 策略（DDPG / TD3 / PPO）
- 可接入 SOH 先验（STAPINN-like）

---

## 1. 你可以用它做什么？

- 在真实环境（或 toy fallback）中采样充电轨迹
- 训练静态代理模型（Cycle0）
- 在代理模型上训练充电策略
- 做自适应循环（Algorithm 2）：真实少量增量数据 + 差分模型更新 + 策略再训练
- 对策略在真实环境中进行测试、可视化与指标记录

---

## 2. 环境准备（必看）

### 2.1 Python 版本建议

建议 Python 3.10+。

### 2.2 安装依赖

```bash
pip install -r requirements.txt
```

> 若你本机缺少 `liionpack/pybamm` 或求解器环境不完整，项目支持 fallback/toy backend（可运行，但物理精度会降低）。

### 2.3 项目目录（核心）

- `configs/`：配置文件（环境、奖励、RL、trainer 等）
- `scripts/`：训练/评估入口脚本
- `src/envs/`：环境定义
- `src/surrogate/`：代理模型
- `src/rl/`：RL 算法实现与训练器
- `src/soh_prior/`：SOH 先验相关模块

---

## 3. 一次完整训练流程（推荐）

## Step 1：训练 Cycle0（真实采样 + 静态代理 + 策略初训）

```bash
python scripts/train_cycle0_build_static_gp.py --config configs/pack_3p6s_spme.yaml
```

默认产物会在 `runs/cycle0/` 下，包括：

- `agent_ckpt.pt`：agent 全量 checkpoint
- `static_surrogate.pt`：静态代理模型
- 训练日志与图像（若启用）

## Step 2：训练 Adaptive Cycles（差分代理 + 组合代理 + 策略迭代）

```bash
python scripts/train_adaptive_cycles.py --config configs/pack_3p6s_spme_with_soh_prior.yaml
```

默认产物在 `runs/adaptive/` 下，包括：

- `diff_surrogate.pt`：差分代理模型
- metrics 与可视化图

## Step 3：评估策略（离线评估）

```bash
python scripts/evaluate_policy.py --config configs/pack_3p6s_spme.yaml --ckpt runs/cycle0/policy.pt
```

## Step 4：真实环境回放测试（建议）

```bash
python scripts/test_real_policy.py --config configs/pack_3p6s_spme.yaml --policy runs/cycle0/agent_ckpt.pt
python scripts/test_real_policy.py --config configs/pack_3p6s_spme.yaml --runs-root runs/adaptive
```

可选参数：

- `--no-guard`：关闭安全守卫（仅建议调试时使用）

---

## 4. 配置文件怎么改（重点）

以 `configs/pack_3p6s_spme.yaml` 为例，常改动区域：

- `env`: 环境参数（`dt`, `max_steps`, `v_max`, `t_max`, 并串联数等）
- `reward`: 奖励权重（`w_soc`, `w_time`, `w_v`, `w_t`, `w_const`）
- `rl`: 算法及超参数
- `trainer.cycle0` / `trainer.adaptive`: 训练轮次与每轮更新策略

### 4.1 RL 算法切换

在配置中修改：

```yaml
rl:
  algorithm: td3  # 可选: ddpg | td3 | ppo
```

### 4.2 TD3 专属参数（仅 `algorithm: td3` 生效）

```yaml
rl:
  td3:
    policy_noise: 0.2
    noise_clip: 0.5
    policy_delay: 2
```

### 4.3 PPO 专属参数（仅 `algorithm: ppo` 生效）

```yaml
rl:
  ppo:
    clip_ratio: 0.2
    ppo_epochs: 4
    gae_lambda: 0.95
    entropy_coef: 0.0
```

---

## 5. 三种算法在本项目中的训练语义

为避免“换算法但训练流程没换”的问题，当前训练器已区分：

- **DDPG / TD3（off-policy）**
  - 使用 replay buffer 语义
  - 保留 `updates_per_epoch` 多次更新策略
  - 允许外部探索噪声

- **PPO（on-policy）**
  - 不再叠加额外外部高斯噪声（策略本身已随机）
  - 按 on-policy 节奏在 rollout 后触发更新

---

## 6. 常见运行命令（速查）

### 6.1 使用 DDPG

1) 改配置：`rl.algorithm: ddpg`
2) 运行：

```bash
python scripts/train_cycle0_build_static_gp.py --config configs/pack_3p6s_spme.yaml
python scripts/train_adaptive_cycles.py --config configs/pack_3p6s_spme_with_soh_prior.yaml
```

### 6.2 使用 TD3

1) 改配置：`rl.algorithm: td3`
2) 检查 `rl.td3` 参数
3) 同上运行训练脚本

### 6.3 使用 PPO

1) 改配置：`rl.algorithm: ppo`
2) 检查 `rl.ppo` 参数
3) 同上运行训练脚本

---

## 7. 输出文件说明

- `runs/cycle0/agent_ckpt.pt`：Cycle0 结束的 agent
- `runs/cycle0/static_surrogate.pt`：静态代理
- `runs/adaptive/diff_surrogate.pt`：差分代理
- `runs/**/metrics.jsonl`：训练日志（可用于后处理画图）
- `runs/**/episode_*.png` / `cycle_*_epoch_*.png`：轨迹图

---

## 8. 常见问题（FAQ）

### Q1：报错找不到 checkpoint？

请先运行 Cycle0，确认 `runs/cycle0/agent_ckpt.pt` 已生成，再运行 adaptive。

### Q2：为什么策略输出经常接近 0A？

- 检查 `action_low/action_high` 是否合理
- 检查奖励中越界惩罚是否过强
- 检查训练是否正确加载了 normalizer 与完整 checkpoint

### Q3：PPO 切换后效果不稳定？

先从以下参数调优：

- `ppo_epochs`
- `clip_ratio`
- `gae_lambda`
- `batch_size`

并适当增加 `policy_rollouts_per_epoch`。

### Q4：liionpack 环境跑不起来怎么办？

可先使用 toy backend 验证流程通不通，再回到真实环境排查求解器与参数配置。

---

## 9. 开发建议

- 先固定一个配置跑通端到端流程，再逐项改超参
- 每次只改一个维度（奖励 / 算法 / env 边界）
- 对比实验时固定随机种子并保存配置快照

---

## 10. 入口脚本索引

- `scripts/train_cycle0_build_static_gp.py`：Cycle0 训练
- `scripts/train_adaptive_cycles.py`：Adaptive 训练
- `scripts/evaluate_policy.py`：评估
- `scripts/test_real_policy.py`：真实环境策略测试

如果你希望，我下一步可以继续补一份“**实验对照模板**”（如 DDPG vs TD3 vs PPO 的统一实验表格与推荐超参初值）。
