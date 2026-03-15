# Adaptive / Differential Experiments

## 实验1：差分模型交叉验证（2x5-fold）

```bash
python scripts/adaptive_experiment/experiment1_diff_cv.py
```

默认会自动扫描：
- `runs/adaptive/adaptive_cycle*/cycle_1/episodes/episode_30/dataset_with_episode.csv`

也可通过 `--adaptive-root` 和 `--dataset-relpath` 调整自动扫描路径，
或继续通过 `--datasets ...` 手动指定数据集列表。

输出：
- `runs/adaptive_experiment/result1/exp1_stage_cv_metrics.csv`
- `runs/adaptive_experiment/result1/exp1_r2_vs_stage.png`
- `runs/adaptive_experiment/result1/exp1_mse_mae_vs_stage.png`

---

## 实验2：25°C下老化阶段1~100，静态代理 vs 组合代理

```bash
python scripts/adaptive_experiment/experiment2_stage100_comparison.py \
  --runs-dir runs --start-stage 1 --end-stage 100
```

输出：
- `runs/adaptive_experiment/result2/exp2_stage1_100_metrics.csv`
- `runs/adaptive_experiment/result2/exp2_stage1_100_comparison.png`

---

## 实验3：第10老化阶段，真实物理训练630ep vs 组合代理

```bash
python scripts/adaptive_experiment/experiment3_stage10_real_vs_combined.py \
  --runs-dir runs --stage 10 --real-train-episodes 630
```

输出：
- `runs/adaptive_experiment/result3/real_env_stage10_ep630_agent_ckpt.pt`
- `runs/adaptive_experiment/result3/exp3_stage10_summary.csv`
- `runs/adaptive_experiment/result3/exp3_real_training_metrics.csv`
- `runs/adaptive_experiment/result3/exp3_stage10_real_vs_combined_curves.png`