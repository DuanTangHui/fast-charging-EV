import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
df = pd.read_csv('dataset.csv')

# Set up the figure with subplots
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Plot 1: Histogram of Actions
sns.histplot(df['action'], bins=30, kde=True, ax=axes[0])
axes[0].set_title('Distribution of Actions (Current)')
axes[0].set_xlabel('Current (A)')
axes[0].set_ylabel('Frequency')

# Plot 2: Action vs Delta Temperature (d_4)
sns.scatterplot(x='action', y='d_4', data=df, ax=axes[1], alpha=0.5)
axes[1].set_title('Action vs Delta Temperature (d_4)')
axes[1].set_xlabel('Current (A)')
axes[1].set_ylabel('Delta Temperature (K)')

# Plot 3: Action vs Delta Voltage (d_2)
sns.scatterplot(x='action', y='d_2', data=df, ax=axes[2], alpha=0.5)
axes[2].set_title('Action vs Delta Voltage (d_2)')
axes[2].set_xlabel('Current (A)')
axes[2].set_ylabel('Delta Voltage (V)')

plt.tight_layout()
plt.savefig('data_analysis.png')