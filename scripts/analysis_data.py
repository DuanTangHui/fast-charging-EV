# import matplotlib.pyplot as plt
# import seaborn as sns
# import pandas as pd
# df = pd.read_csv('dataset.csv')

# # Set up the figure with subplots
# fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# # Plot 1: Histogram of Actions
# sns.histplot(df['action'], bins=30, kde=True, ax=axes[0])
# axes[0].set_title('Distribution of Actions (Current)')
# axes[0].set_xlabel('Current (A)')
# axes[0].set_ylabel('Frequency')

# # Plot 2: Action vs Delta Temperature (d_4)
# sns.scatterplot(x='action', y='d_4', data=df, ax=axes[1], alpha=0.5)
# axes[1].set_title('Action vs Delta Temperature (d_4)')
# axes[1].set_xlabel('Current (A)')
# axes[1].set_ylabel('Delta Temperature (K)')

# # Plot 3: Action vs Delta Voltage (d_2)
# sns.scatterplot(x='action', y='d_2', data=df, ax=axes[2], alpha=0.5)
# axes[2].set_title('Action vs Delta Voltage (d_2)')
# axes[2].set_xlabel('Current (A)')
# axes[2].set_ylabel('Delta Voltage (V)')

# plt.tight_layout()
# plt.savefig('data_analysis.png')

import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('dataset.csv')

# Identify episodes
df['episode'] = (df['s_0'].diff() < -0.1).cumsum()
num_episodes = df['episode'].nunique()

plt.figure(figsize=(15, 12))

# Subplot 1: Current (Action)
plt.subplot(3, 2, 1)
for i in range(num_episodes):
    ep_data = df[df['episode'] == i].reset_index()
    plt.plot(ep_data.index, ep_data['action'], alpha=0.3, color='blue')
plt.xlabel('Step')
plt.ylabel('Current')
plt.title('Current Profiles (50 Episodes)')

# Subplot 2: Temperature (s_4)
plt.subplot(3, 2, 2)
for i in range(num_episodes):
    ep_data = df[df['episode'] == i].reset_index()
    plt.plot(ep_data.index, ep_data['s_4'], alpha=0.3, color='red')
plt.xlabel('Step')
plt.ylabel('Temperature (K)')
plt.title('Temperature Profiles (50 Episodes)')

# Subplot 3: SOC (s_0)
plt.subplot(3, 2, 3)
for i in range(num_episodes):
    ep_data = df[df['episode'] == i].reset_index()
    plt.plot(ep_data.index, ep_data['s_0'], alpha=0.3, color='green')
plt.xlabel('Step')
plt.ylabel('SOC')
plt.title('SOC Profiles (50 Episodes)')

# Subplot 4: Current vs SOC
plt.subplot(3, 2, 4)
for i in range(num_episodes):
    ep_data = df[df['episode'] == i]
    plt.plot(ep_data['s_0'], ep_data['action'], alpha=0.3, color='purple')
plt.xlabel('SOC')
plt.ylabel('Current')
plt.title('Current vs SOC')

# Subplot 5: Delta Temperature (d_4)
plt.subplot(3, 2, 5)
for i in range(num_episodes):
    ep_data = df[df['episode'] == i].reset_index()
    plt.plot(ep_data.index, ep_data['d_4'], alpha=0.3, color='orange')
plt.axhline(0, color='black', linestyle='--', linewidth=1)
plt.xlabel('Step')
plt.ylabel('Delta Temp (d_4)')
plt.title('Temperature Change Rate (d_4)')

# Subplot 6: Close up of Current at high SOC (> 0.7)
plt.subplot(3, 2, 6)
high_soc_data = df[df['s_0'] > 0.7]
for i in range(num_episodes):
    ep_data = df[(df['episode'] == i) & (df['s_0'] > 0.7)]
    if not ep_data.empty:
        plt.plot(ep_data['s_0'], ep_data['action'], alpha=0.3)
plt.xlabel('SOC (>0.7)')
plt.ylabel('Current')
plt.title('Current Tapering at High SOC')

plt.tight_layout()
plt.savefig('charging_curves_detailed.png')

# Statistical check for tapering
tapering_count = 0
for i in range(num_episodes):
    ep_data = df[df['episode'] == i]
    max_curr_mag = ep_data['action'].abs().max()
    last_curr_mag = abs(ep_data['action'].iloc[-1])
    # Tapering defined as final current being less than 20% of max current in that episode
    if last_curr_mag < 0.2 * max_curr_mag:
        tapering_count += 1

print(f"Total episodes: {num_episodes}")
print(f"Episodes with significant current tapering at end (<20% max): {tapering_count}")