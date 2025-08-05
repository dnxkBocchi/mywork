import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.ticker import MaxNLocator

# 设置中文显示
plt.rcParams["font.family"] = ["SimHei", "WenQuanYi Micro Hei", "Heiti TC"]
plt.rcParams["axes.unicode_minus"] = False  # 解决负号显示问题

# 实验数据整理
algorithms = ["RANDOM", "RR", "GA", "PSO", "MOPSO", "DQN", "GMP-DQN"]
metrics = ["Total Fitness", "Total Distance", "Total Time", "Total Success"]

# 小型场景数据
small_data = {
    "Total Distance": [673.79, 726.45, 748.42, 726.04, 719.27, 681.30, 663.87],
    "Total Time": [524.81, 518.38, 518.38, 412.18, 391.46, 382.53, 395.32],
    "Total Fitness": [0.24, 0.94, 0.94, 0.69, 0.67, 0.82, 0.81],
    "Total Success": [0.20, 0.73, 0.80, 0.73, 0.73, 0.87, 0.87]
}

# 中型场景数据
middle_data = {
    "Total Distance": [1399.88, 1389.09, 1382.14, 1326.96, 1361.79, 1333.34, 1218.58],
    "Total Time": [688.00, 813.78, 753.85, 775.86, 771.74, 755.62, 724.89],
    "Total Fitness": [0.22, 0.95, 0.95, 0.69, 0.87, 0.95, 0.95],
    "Total Success": [0.07, 0.73, 0.87, 0.70, 0.83, 1.00, 0.97]
}

# 大型场景数据
large_data = {
    "Total Distance": [2951.82, 2827.78, 2861.04, 2772.75, 2814.38, 2752.66, 2429.19],
    "Total Time": [1754.32, 1656.29, 1616.42, 1615.50, 1670.03, 1576.28, 1461.04],
    "Total Fitness": [0.21, 0.94, 0.94, 0.57, 0.81, 0.90, 0.92],
    "Total Success": [0.08, 0.75, 0.78, 0.53, 0.85, 0.85, 0.90]
}

# 绘制柱状图
def plot_metrics(data, title, figsize=(10, 8)):
    # 设置中文字体支持，选用更美观的字体
    plt.rcParams["font.family"] = ["Times New Roman", "serif"]
    plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
    
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    axes = axes.flatten()
    
    # 调整后的淡色系颜色（降低饱和度）
    light_colors = [
        '#FFA0A0',  # 淡红
        '#7EE8E0',  # 淡青
        '#79C7E3',  # 淡蓝
        '#FFC09F',  # 淡橙
        '#C5E8D7',  # 淡绿
        '#FFE8A3',  # 淡黄
        '#BDA0CB'   # 淡紫
    ]
    
    for i, metric in enumerate(metrics):
        ax = axes[i]
        x = np.arange(len(algorithms))
        # 设置width=1实现柱间距为0，通过align='edge'确保无间隙
        bars = ax.bar(x, data[metric], width=1, align='edge', 
                      color=light_colors[:len(algorithms)])
        
        # 设置标题字体（加粗+适中大小）
        ax.set_title(metric, fontsize=13, fontweight='bold', pad=10)
        # 设置x轴刻度和标签（调整字体大小和旋转避免重叠）
        ax.set_xticks(x + 0.5)  # 居中显示标签
        ax.set_xticklabels(algorithms, fontsize=11, rotation=30, ha='right')
        # 设置y轴刻度格式
        ax.yaxis.set_major_locator(MaxNLocator(integer=False))
        # 优化刻度标签字体
        ax.tick_params(axis='y', labelsize=10)
        
        # 添加数据标签（调整位置和字体）
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{height:.2f}', ha='center', va='bottom', 
                    fontsize=10, fontweight='medium')
    
    # 设置总标题（加大字号+加粗）
    plt.suptitle(title, fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    return fig

# 生成三个场景的图表
fig_small = plot_metrics(small_data, "小型场景（5×5）各算法性能对比")
# fig_middle = plot_metrics(middle_data, "中型场景（10×10）各算法性能对比")
# fig_large = plot_metrics(large_data, "大型场景（20×20）各算法性能对比")

# 显示图表
plt.show()