# static_experiment

本目录包含静态代理模型相关实验。


- `experiment_static_surrogate_validity.py`（实验1）
静态代理模型有效性分析实验一：验证静态代理模型的拟合有效性

1) 基于统计指标的交叉验证：
   - 使用带 episode 列的数据文件（dataset_with_episode.csv）
   - 对 E=20..50，每个 E 采用累计数据 (episode<=E)
   - 在每个 E 上执行 5-fold，并重复 2 次，得到平均 R² / MSE / MAE 曲线

2) 完整环境仿真对比：
   - 同一 RL 智能体分别在真实物理环境 (liionpack+PyBaMM) 与静态代理上交互
   - 单回合，max_steps=720，SOC>=0.8 终止
   - 输出电流/电压/SOC/温度四条曲线与耗时对比

- `experiment_policy_shape_comparison.py`（实验2）
  - 对比两种训练方式最终策略在**真实物理环境**下的一次完整充电轨迹：电流/SOC/电压/温度-时间曲线。
  - 方案A：真实环境按 `train_cycle0 -> collect_real_data` 同逻辑训练 650 episodes。
  - 方案B：直接加载已训练混合策略 `runs/cycle0/agent_ckpt.pt`。
  - 会默认保存真实环境训练后的 agent 到 `runs/static_experiment/exp1_policy_shape/real_trained_agent_ckpt.pt`。

- `experiment_training_process_1000.py`（实验3）
  - 对比真实基线与混合方案在 1000 episode 全过程中的四项指标：
    - 累积回报
    - 充电时间
    - 电压违规程度（max(V)-4.2）
    - 温度违规程度（max(T)-309.15）
  - 输出原始 CSV 与总图。
  - 会默认保存：真实基线 agent、混合方案 agent、拟合后的 surrogate 模型。

- `scripts/experiment_static_surrogate_validity.py`（实验3，仓库已有实现）
  - 静态模型回归指标评估（R2/MSE/MAE）。
  - 在真实环境与静态代理环境中各模拟一次完整充电，输出曲线对比与原始轨迹 CSV。


> 说明：真实物理环境训练已切换为与 `collect_real_data` 一致的交互逻辑，
> 若物理仿真崩溃将按原始行为直接报错中断（不再吞错继续）。

## 运行示例

```bash
python scripts/static_experiment/experiment_policy_shape_comparison.py --config configs/pack_3p6s_spme.yaml --real-episodes 650 --agent-ckpt runs/cycle0/agent_ckpt.pt

python scripts/static_experiment/experiment_training_process_1000.py  --config configs/pack_3p6s_spme.yaml

python scripts/experiment_static_surrogate_validity.py  
```

## 输出

默认输出在 `runs/static_experiment/` 或实验脚本默认输出目录下，所有实验均会保存原始 CSV 以支持复现。