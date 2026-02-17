# import pandas as pd
# import numpy as np

# # Load data
# df = pd.read_csv('dataset1.csv')

# # Define problematic condition
# cond_problematic = (df['s_0'] > 0.6) & (df['action'].abs() < 2)

# # Identify indices
# problematic_indices = df[cond_problematic].index

# # Randomly select 80% of these to drop
# np.random.seed(42)
# drop_indices = np.random.choice(problematic_indices, size=int(len(problematic_indices) * 0.8), replace=False)

# # Create cleaned dataset
# df_cleaned = df.drop(drop_indices)

# # Check new stats
# print(f"Original samples: {len(df)}")
# print(f"Removed samples: {len(drop_indices)}")
# print(f"Cleaned samples: {len(df_cleaned)}")

# # Save to new csv
# df_cleaned.to_csv('dataset_cleaned.csv', index=False)

# # Comparison of action distribution in SOC > 0.6
# print("\nAction Mean in SOC > 0.6 (Original):", df[df['s_0'] > 0.6]['action'].mean())
# print("Action Mean in SOC > 0.6 (Cleaned):", df_cleaned[df_cleaned['s_0'] > 0.6]['action'].mean())


# import pandas as pd
# import numpy as np

# # 加载之前清洗过的数据集（已剔除高SOC静态样本）
# df = pd.read_csv('dataset_cleaned.csv')

# # 识别降温样本 (d_4 < 0 或 d_5 < 0)
# # 我们主要关注温度最大值的变化 d_4
# cooling_samples = df[df['d_4'] < 0]
# other_samples = df[df['d_4'] >= 0]

# print(f"原始降温样本数: {len(cooling_samples)}")
# print(f"其他样本数: {len(other_samples)}")

# # 执行过采样：将降温样本复制 3 倍
# # 这样在训练的每一个 Epoch 中，模型看到降温信号的频率会大大增加
# df_oversampled = pd.concat([other_samples, cooling_samples, cooling_samples, cooling_samples])

# # 随机打乱数据，防止模型学习到序列偏差
# df_oversampled = df_oversampled.sample(frac=1).reset_index(drop=True)

# print(f"过采样后总样本数: {len(df_oversampled)}")
# print(f"过采样后降温样本占比: {len(cooling_samples)*3 / len(df_oversampled) * 100:.2f}%")

# # 保存新数据集
# df_oversampled.to_csv('dataset_oversampled.csv', index=False)

import pandas as pd
import numpy as np

# 1. 加载你现在的过采样数据集
df = pd.read_csv('dataset_oversampled.csv')

# 2. 自动获取环境温度 (假设数据集开始时的温度就是环境温度)
t_ambient = 298.15
# 3. 注入关键物理特征：温差 (s_7)
# 这一步非常关键！它给了模型一个“散热参考点”
df['s_7'] = df['s_4'] - t_ambient

# 4. 调整列顺序，确保 action 还在状态特征之后（如果你的模型输入是 s0-s7, action）
# 我们把 s_7 放到 s_6 后面
cols = list(df.columns)
# 原列顺序: s_0...s_6, action, d_0...d_5
# 我们要在 action 之前插入 s_7
new_cols = cols[:7] + ['s_7'] + [cols[7]] + cols[8:-1] 
df_final = df[new_cols]

# 5. 保存最终版
df_final.to_csv('dataset_final_physics.csv', index=False)
print("成功生成 dataset_final_physics.csv！该文件包含 s_0 到 s_7 共 8 个状态输入。")