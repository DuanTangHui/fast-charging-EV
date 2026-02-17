# import pandas as pd
# import matplotlib.pyplot as plt
# # Load the dataset
# df = pd.read_csv('dataset.csv')

# # Inspect the first few rows and summary information
# print(df.head())
# print(df.info())
# print(df.describe())

# df['episode_id'] = (df['s_0'].diff() < -0.1).cumsum()

# # Quantify action smoothness
# action_diff_std = df['action'].diff().std()
# print(f"Standard deviation of action diff: {action_diff_std}")

# # Check action smoothness per episode
# action_diff_std_per_ep = df.groupby('episode_id')['action'].apply(lambda x: x.diff().std())
# print("\nAction diff std per episode (first 10):")
# print(action_diff_std_per_ep.head(10))

# # Scatter plot of action vs d_4 to see the trend
# plt.figure(figsize=(8, 6))
# plt.scatter(df['action'], df['d_4'], alpha=0.1)
# plt.xlabel('Action (Current)')
# plt.ylabel('d_4 (Delta T_max)')
# plt.title('Relationship between Current and Temp Change')
# plt.axhline(0, color='red', linestyle='--')
# plt.savefig('action_vs_d4.png')

# # Examine the "Surrogate Model training" issue.
# # Could s_6 (I_prev) and action be very similar?
# print("\nCorrelation between s_6 and action:", df['s_6'].corr(df['action']))
# import pandas as pd
# import numpy as np
# from sklearn.linear_model import LinearRegression

# # Load the dataset
# df = pd.read_csv('dataset.csv')

# # Constants
# T_env = 298.15

# # Prepare features and target
# # Equation: d_4 = alpha * action^2 - beta * (s_4 - T_env)
# # Let Y = d_4, X1 = action^2, X2 = -(s_4 - T_env)
# # Then Y = alpha * X1 + beta * X2

# Y = df['d_4'].values
# X1 = df['action'].values**2
# X2 = -(df['s_4'].values - T_env)

# X = np.column_stack((X1, X2))

# # Perform linear regression without intercept
# model = LinearRegression(fit_intercept=False)
# model.fit(X, Y)

# alpha, beta = model.coef_

# print(f"Estimated alpha (Joule heating): {alpha}")
# print(f"Estimated beta (Cooling coefficient): {beta}")

# # Calculate R-squared to see how well the physical prior fits the data
# r_sq = model.score(X, Y)
# print(f"R-squared of the physical prior: {r_sq}")

import pandas as pd

# 1. 加载原始数据集
df = pd.read_csv('dataset.csv')

# 2. 设定回归得到的物理参数
alpha = 2.181e-4  # 焦耳生热系数
beta = 2.600e-2   # 散热系数
T_env = 298.15    # 环境温度 (K)

# 3. 计算物理主项: alpha * I^2 - beta * (T_max - T_env)
# 在你的数据集中：s_4 是 T_max，action 是 I
physics_term = alpha * (df['action']**2) - beta * (df['s_4'] - T_env)

# 4. 将 d_4 替换为残差 r_theta
# r_theta = 实际Delta_T - 物理预测Delta_T
df['d_4'] = df['d_4'] - physics_term

# 5. 保存处理后的数据集，用于训练残差网络
df.to_csv('dataset_with_residual.csv', index=False)

print("处理完成！d4 列已成功替换为物理残差 r_theta。")
print(df[['s_4', 'action', 'd_4']].head())