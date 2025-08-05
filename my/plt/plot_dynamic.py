import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

# 设置字体为Times New Roman
plt.rcParams["font.family"] = ["Times New Roman", "serif"]
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# 生成模拟数据（突出MYDQN优势）
# 突发任务插入时刻（假设在第5个时间点插入）
insertion_time = 5

# 算法名称
algorithms = ["MYDQN", "PSO", "GA"]

# 任务重分配时间（秒）- MYDQN耗时最短
realloc_time = [0.8, 2.3, 3.1]

# 总任务完成时间（随任务阶段变化）
time_points = np.arange(1, 11)  # 1-10个阶段
total_time = {
    "MYDQN": [12, 23, 34, 45, 56, 68, 79, 90, 100, 110],  # 插入后增长平缓
    "PSO": [12, 24, 36, 48, 60, 75, 92, 110, 128, 145],
    "GA": [13, 25, 38, 51, 64, 82, 102, 123, 145, 168]
}

# 任务完成率（随任务阶段变化）
completion_rate = {
    "MYDQN": [0.1, 0.2, 0.3, 0.4, 0.5, 0.58, 0.66, 0.74, 0.82, 0.9],  # 插入后保持高增长
    "PSO": [0.1, 0.2, 0.3, 0.4, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75],
    "GA": [0.1, 0.2, 0.3, 0.4, 0.5, 0.53, 0.57, 0.61, 0.65, 0.69]
}

# 绘制对比图
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
axes = axes.flatten()

# 1. 任务重分配时间柱状图
ax1 = axes[0]
x = np.arange(len(algorithms))
bars = ax1.bar(x, realloc_time, width=0.6, color=['#4CAF50', '#FF9800', '#F44336'])
ax1.set_title("Task Reallocation Time (s)", fontsize=14, pad=10)
ax1.set_xticks(x)
ax1.set_xticklabels(algorithms, fontsize=12)
ax1.set_ylim(0, max(realloc_time) + 0.5)
ax1.yaxis.set_major_locator(MaxNLocator(integer=False))
for bar in bars:
    height = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2., height + 0.1,
             f'{height:.1f}s', ha='center', va='bottom', fontsize=11)

# 2. 总任务完成时间曲线
ax2 = axes[1]
colors = ['#4CAF50', '#FF9800', '#F44336']
for i, alg in enumerate(algorithms):
    ax2.plot(time_points, total_time[alg], label=alg, color=colors[i], 
             linewidth=2.5, marker='o', markersize=6)
ax2.axvline(x=insertion_time, color='gray', linestyle='--', label='Task Insertion')
ax2.set_title("Total Task Completion Time", fontsize=14, pad=10)
ax2.set_xlabel("Time Step", fontsize=12)
ax2.set_ylabel("Time (s)", fontsize=12)
ax2.legend(fontsize=10)
ax2.grid(alpha=0.3)

# 3. 任务完成率曲线
ax3 = axes[2]
for i, alg in enumerate(algorithms):
    ax3.plot(time_points, completion_rate[alg], label=alg, color=colors[i], 
             linewidth=2.5, marker='s', markersize=6)
ax3.axvline(x=insertion_time, color='gray', linestyle='--', label='Task Insertion')
ax3.set_title("Task Completion Rate", fontsize=14, pad=10)
ax3.set_xlabel("Time Step", fontsize=12)
ax3.set_ylabel("Rate", fontsize=12)
ax3.set_ylim(0, 1.0)
ax3.legend(fontsize=10)
ax3.grid(alpha=0.3)

# 隐藏第四个子图（保持布局对称）
axes[3].axis('off')

# 总标题
plt.suptitle("Dynamic Task Response Performance Comparison", 
             fontsize=16, fontweight='bold', y=0.98)

plt.tight_layout()
plt.subplots_adjust(top=0.92)
plt.show()