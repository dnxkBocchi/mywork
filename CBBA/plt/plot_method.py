import matplotlib.pyplot as plt
import numpy as np
import os
import sys

# 路径配置（确保文件读取和模块导入正常）
current_script_dir = os.path.dirname(os.path.abspath(__file__))
plt_dir = current_script_dir
my_dir = os.path.dirname(plt_dir)
sys.path.append(my_dir)


# 方法名称
methods = ["AUCTION", "CONTRACTNET", "CCBA_KMEANS", "CCBA_MY"]

# 数据
distance = [265.95, 286.18, 253.64, 253.64]
time = [208.82, 201.97, 218.20, 196.46]
messages = [120, 110, 90, 74]
success_rate = [1.0, 1.0, 1.0, 1.0]

# 成功率转百分比
success_rate_percent = [x * 100 for x in success_rate]

x = np.arange(len(methods))
width = 0.35

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# =========================
# 左图：Distance 和 Time
# =========================
ax1 = axes[0]
bars1 = ax1.bar(x - width / 2, distance, width, label="Distance")
bars2 = ax1.bar(x + width / 2, time, width, label="Time")

ax1.set_title("Distance and Time Comparison")
ax1.set_xlabel("Methods")
ax1.set_ylabel("Value")
ax1.set_xticks(x)
ax1.set_xticklabels(methods)
ax1.legend()

# 让左图 y 轴不从 0 开始，而是根据数据自动缩放
left_values = distance + time
ymin = min(left_values)
ymax = max(left_values)
margin = (ymax - ymin) * 0.25  # 留一点边距，让差异更明显
ax1.set_ylim(ymin - margin, ymax + margin)

# =========================
# 右图：Messages 和 Success Rate
# =========================
ax2 = axes[1]
bars3 = ax2.bar(x - width / 2, messages, width, label="Messages")
bars4 = ax2.bar(x + width / 2, success_rate_percent, width, label="Success Rate (%)")

ax2.set_title("Messages and Success Rate Comparison")
ax2.set_xlabel("Methods")
ax2.set_ylabel("Value")
ax2.set_xticks(x)
ax2.set_xticklabels(methods)
ax2.legend()


# =========================
# 给柱子加标签
# =========================
def add_labels(ax, bars, fmt="{:.2f}", offset=1.0):
    for bar in bars:
        h = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            h + offset,
            fmt.format(h),
            ha="center",
            va="bottom",
            fontsize=9,
        )


add_labels(ax1, bars1, "{:.2f}", offset=1.0)
add_labels(ax1, bars2, "{:.2f}", offset=1.0)
add_labels(ax2, bars3, "{:.0f}", offset=1.0)
add_labels(ax2, bars4, "{:.0f}%", offset=1.0)

save_dir = os.path.join(my_dir, "outputs")
os.makedirs(save_dir, exist_ok=True)
save_path = os.path.join(save_dir, "method_comparison.pdf")

plt.tight_layout()
plt.savefig(save_path, dpi=300, bbox_inches="tight", format="pdf")
plt.show()
