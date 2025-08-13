import numpy as np
import matplotlib.pyplot as plt
import re
import sys
import os

# 设置中文显示
plt.rcParams["font.family"] = ["SimHei", "WenQuanYi Micro Hei", "Heiti TC"]
plt.rcParams["axes.unicode_minus"] = False  # 解决负号显示问题


# 从文件读取耗时数据
def read_time_data(file_path):
    with open(file_path, "r") as f:
        content = f.read()

    data_blocks = re.split(r"\n\s*\n", content.strip())
    algorithms = ["RANDOM", "RR", "GA", "PSO", "MOPSO", "GMP-DRL"]
    scenarios = []
    for block in data_blocks:
        times = [
            float(line.strip()) * 1000 for line in block.split("\n") if line.strip()
        ]
        scenarios.append(times)
    return algorithms, scenarios


# 绘制合并折线图
def plot_time_comparison_all(algorithms, scenarios, title, figsize=(7, 6)):
    plt.rcParams["font.family"] = ["Times New Roman", "serif"]
    plt.figure(figsize=figsize)

    colors = ["#358DA1", "#FF6B6B", "#4E9A06"]
    markers = ["o", "s", "^"]
    labels = ["5*5 scale", "10*10 scale", "20*20 scale"]
    offsets = [200, 400, 600]  # 三条线的标签错开高度
    for i, times in enumerate(scenarios):
        plt.plot(
            algorithms,
            times,
            marker=markers[i],
            color=colors[i],
            markersize=8,
            linewidth=2,
            linestyle="-",
            markerfacecolor=colors[i],
            label=labels[i],
        )

        # 数据标签
        for j, value in enumerate(times):
            plt.text(
                j,
                value + offsets[i] if j == 0 or j == 1 or j == 5 else value,
                f"{value:.2f}",
                ha="center",
                va="bottom",
                fontsize=10,
                color=colors[i]
            )

    plt.title(title, fontsize=15, pad=20)
    plt.xlabel("Methods", fontsize=12, labelpad=10)
    plt.ylabel("Spend time (milliseconds)", fontsize=12, labelpad=10)
    plt.xticks(rotation=30, ha="right", fontsize=10)
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.legend(fontsize=10, loc="best")
    plt.tight_layout()
    return plt.gcf()


if __name__ == "__main__":
    current_script_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(current_script_dir, "..", "plt", "spend_time.txt")
    algorithms, scenarios = read_time_data(file_path)

    fig = plot_time_comparison_all(
        algorithms, scenarios, "Spend Time Comparison for All Scales"
    )
    fig.savefig(
        os.path.join(current_script_dir, "..", "pic", "time_comparison_all.png"),
        dpi=300,
        bbox_inches="tight",
    )
    plt.show()
