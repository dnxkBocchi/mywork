import numpy as np
import matplotlib.pyplot as plt
import os
import sys
from matplotlib.ticker import MaxNLocator

plt.rcParams["font.family"] = ["Times New Roman", "serif"]
plt.rcParams["axes.unicode_minus"] = False

# 从文件读取数据
def read_data_from_file(file_path):
    # 读取文件内容并按行处理
    with open(file_path, "r") as f:
        data_lines = [line.strip() for line in f.readlines() if line.strip()]

    algorithms = ['RANDOM', 'RR', 'GA', 'PSO', 'MOPSO', 'DRL', 'GMP-DRL']
    metrics = [
        "Total Reward","Total Fitness", "Total Distance", "Total Time", "Total Success",
    ]

    # 分割数据块（每个场景有6行数据，与算法数量对应）
    scenario_size = len(algorithms)  # 每个场景的数据行数等于算法数量
    scenarios_data = []

    # 按场景大小分割数据
    for i in range(0, len(data_lines), scenario_size):
        scenario_lines = data_lines[i : i + scenario_size]
        scenario_data = {metric: [] for metric in metrics}

        for line in scenario_lines:
            values = [float(val.strip()) for val in line.split(",") if val.strip()]
            for idx, metric in enumerate(metrics):
                scenario_data[metric].append(values[idx])

        scenarios_data.append(scenario_data)

    return algorithms, metrics, scenarios_data

# 绘制两个指标的分组柱状图（黑白风格）
def plot_two_metrics(data, title, algorithms, metric_pair, location, save_path, figsize=(6, 4)):
    x = np.arange(len(metric_pair))  # 两个指标的位置
    width = 0.13  # 每个柱子的宽度
    
    # 黑白风格设置：不同的线条样式和填充模式
    line_styles = ['-', '--', '-.', ':', '-', '--', '-.']  # 不同线条样式区分不同算法
    fill_patterns = ['', '/', '\\', 'x', '+', '|', '/']  # 填充图案区分不同算法
    edge_widths = [1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 2.0]  # GMP-DRL线条更粗以突出显示

    fig, ax = plt.subplots(figsize=figsize)
    
    for i, alg in enumerate(algorithms):
        values = [data[metric_pair[0]][i], data[metric_pair[1]][i]]
        # 使用黑白样式：无填充色，仅用边框和图案区分
        ax.bar(x + (i - len(algorithms)/2) * width + width/2,
               values, width, label=alg, 
               color='none',  # 无填充色
               edgecolor='black',  # 黑色边框
               linestyle=line_styles[i],  # 不同线条样式
               linewidth=edge_widths[i],  # 线条宽度
               hatch=fill_patterns[i])  # 填充图案
        
        # 添加数值标签
        for j, val in enumerate(values):
            ax.text(x[j] + (i - len(algorithms)/2) * width + width/2,
                    val + 0.01, f"{val:.2f}",
                    ha='center', va='bottom', fontsize=7)

    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(metric_pair, fontsize=12)
    ax.yaxis.set_major_locator(MaxNLocator(integer=False))
    ax.tick_params(axis="y", labelsize=10)
    ax.legend(fontsize=6, loc=location)
    plt.tight_layout()

    fig.savefig(save_path, dpi=300, bbox_inches='tight')
    return fig

# ===== 示例调用 =====
if __name__ == "__main__":
    current_script_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(current_script_dir, "..", "plt", "total_method.txt")
    algorithms, metrics, scenarios = read_data_from_file(file_path)
    save_dir = os.path.join(current_script_dir, "..", "pic")

    scale = 5
    for i in range(len(scenarios)):
        # 图1: Total Fitness + Total Success
        fig1 = plot_two_metrics(scenarios[i], f"{scale}*{scale}: Fitness and Success",
                                algorithms, ["Total Fitness", "Total Success"], location='upper left',
                                save_path=os.path.join(save_dir, f"fitness_success_{scale}.svg"))
        # 图2: Total Distance + Total Time
        fig2 = plot_two_metrics(scenarios[i], f"{scale}*{scale}: Distance and Time",
                            algorithms, ["Total Distance", "Total Time"], location='upper right',
                            save_path=os.path.join(save_dir, f"distance_time_{scale}.svg"))
        scale *= 2

    plt.show()
    