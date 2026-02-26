import matplotlib.pyplot as plt
import numpy as np

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']  # 使用黑体
plt.rcParams['axes.unicode_minus'] = False    # 解决负号显示问题

# 六个维度
labels = ["SOC奖励", "时间惩罚", "电压惩罚", "温度惩罚", "一致性惩罚", "动作惩罚"]
weights = [20, 10, 50, 30, 15, 10]  # 推荐权重配置表（归一化版）

# 角度设置
angles = np.linspace(0, 2*np.pi, len(labels), endpoint=False).tolist()
weights += weights[:1]  # 闭合多边形
angles += angles[:1]

# 绘制雷达图
fig, ax = plt.subplots(figsize=(6,6), subplot_kw=dict(polar=True))

# 使用彩色填充
ax.plot(angles, weights, 'o-', linewidth=2, color='red', label="奖励函数偏好")
ax.fill(angles, weights, color='red', alpha=0.25)

# 设置标签
ax.set_xticks(angles[:-1])
ax.set_xticklabels(labels, fontsize=12, color="blue")  # 中文标签，蓝色字体
ax.set_yticklabels([])  # 隐藏半径刻度
ax.set_title("奖励函数偏好雷达图", fontsize=16, color="darkgreen")

# 添加网格线和图例
ax.grid(color="gray", linestyle="--", linewidth=0.5)
plt.legend(loc="upper right")

plt.show()