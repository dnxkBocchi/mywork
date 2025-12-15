import matplotlib.pyplot as plt
import numpy as np

# 全局字体配置（符合期刊规范，避免中文依赖）
plt.rcParams["font.family"] = ["Times New Roman", "serif"]
plt.rcParams["axes.unicode_minus"] = False  # 解决负号显示异常
plt.rcParams["font.size"] = 10  # 基础字体大小，保证标签协调

# ---------------------- 1. 数据准备 ----------------------
models = ['w/o CGMP', 'w/o IGMP', 'w/o BAR', 'w/o ABR', 'DCMP']  # 模型名称
metrics = ['RMSE', 'MAE', 'R', 'SD']  # 评估指标

# CASF2013 各指标下的模型数值
casf2013_data = {
    'RMSE': [1.426, 1.504, 1.323, 1.559, 1.261],
    'MAE': [1.145, 1.166, 0.991, 1.189, 0.979],
    'R': [0.776, 0.757, 0.833, 0.756, 0.854],
    'SD': [1.459, 1.500, 1.248, 1.508, 1.212]
}

# CASF2016 各指标下的模型数值
casf2016_data = {
    'RMSE': [1.334, 1.416, 1.217, 1.437, 1.162],
    'MAE': [1.013, 1.055, 0.931, 1.163, 0.905],
    'R': [0.808, 0.724, 0.816, 0.699, 0.845],
    'SD': [1.334, 1.416, 1.217, 1.437, 1.162]
}

# 柱子颜色（与原图配色对应）
colors = ['#4575b4', '#fdae61', '#66bd63', '#d73027', '#938ed5']


# ---------------------- 2. 绘制分组柱状图 ----------------------
fig, axes = plt.subplots(1, 2, figsize=(15, 6))  # 1行2列的子图，设置画布大小

for i, (ax, data, title) in enumerate(zip(axes, [casf2013_data, casf2016_data], ['CASF2013', 'CASF2016'])):
    x = np.arange(len(metrics))  # 每个“指标组”的位置（共4组：RMSE、MAE、R、SD）
    total_width = 0.8  # 每个“指标组”的总宽度
    bar_width = total_width / len(models)  # 单个柱子的宽度

    # 为每个模型绘制柱子
    for j, (model, color) in enumerate(zip(models, colors)):
        # 计算每个柱子的水平位置，确保在“指标组”内均匀分布
        bar_positions = x - total_width/2 + bar_width/2 + j * bar_width
        ax.bar(
            bar_positions, 
            [data[metric][j] for metric in metrics],  # 该模型在各指标下的数值
            width=bar_width,
            color=color,
            label=model
        )

    # 设置子图的x轴刻度、标签和标题
    ax.set_xticks(x)
    ax.set_xticklabels(metrics, fontweight='bold')  # x轴刻度标签加粗
    ax.set_title(title, fontsize=14, fontweight='bold')  # 标题加粗
    ax.set_ylabel('Value', fontsize=12, fontweight='bold')  # y轴标签加粗

# 在第一个子图添加图例（两个子图图例一致）
axes[0].legend(prop={'weight': 'bold'})
axes[1].legend(prop={'weight': 'bold'})

# 调整布局，避免元素重叠
plt.tight_layout()

# ---------------------- 3. 保存为SVG矢量图 ----------------------
plt.savefig('casf_results.svg', format='svg', dpi=300)

# 显示图形（可选）
plt.show()