import matplotlib.pyplot as plt
import numpy as np


# 全局字体配置（符合期刊规范，避免中文依赖）
plt.rcParams["font.family"] = ["Times New Roman", "serif"]
plt.rcParams["axes.unicode_minus"] = False  # 解决负号显示异常
plt.rcParams["font.size"] = 10  # 基础字体大小，保证标签协调

# ---------------------- 数据准备 ----------------------
# 子图(a)：Number of Layers
x_layers = [1, 2, 3, 4]
y_casf2013_layers = [0.75, 0.805, 0.855, 0.82]   # CASF2013 各层的 R 值
y_casf2016_layers = [0.78, 0.82, 0.845, 0.833]  # CASF2016 各层的 R 值

# 子图(b)：Hidden Layer Dimension
x_dim = [64, 128, 256, 512]
y_casf2013_dim = [0.758, 0.81, 0.855, 0.805]    # CASF2013 各维度的 R 值
y_casf2016_dim = [0.775, 0.835, 0.845, 0.82]   # CASF2016 各维度的 R 值


# ---------------------- 绘制折线图 ----------------------
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))  # 1行2列子图，设置画布大小

# 子图(a)：Number of Layers
ax1.plot(x_layers, y_casf2013_layers, 'b-o', label='CASF2013', linewidth=2, markersize=6)
ax1.plot(x_layers, y_casf2016_layers, 'r--s', label='CASF2016', linewidth=2, markersize=6)
ax1.set_xlabel('(a) Number of Layers', fontweight='bold')     # X轴标签加粗
ax1.set_ylabel('R', fontweight='bold')                   # Y轴标签加粗
ax1.set_ylim(0.74, 0.86)                                 # Y轴范围匹配原图
ax1.grid(True, linestyle='--', alpha=0.7)                # 添加网格线
ax1.legend(prop={'weight': 'bold'})                      # 图例文字加粗
# 明确设置x轴刻度为数据值（确保每个点都显示刻度）
ax1.set_xticks(x_layers)
ax1.set_xticklabels(x_layers, fontweight='bold')  # 刻度值加粗

# 子图(b)：Hidden Layer Dimension
ax2.plot(x_dim, y_casf2013_dim, 'b-o', label='CASF2013', linewidth=2, markersize=6)
ax2.plot(x_dim, y_casf2016_dim, 'r--s', label='CASF2016', linewidth=2, markersize=6)
ax2.set_xlabel('(b) Hidden Layer Dimension', fontweight='bold')     # X轴标签加粗
ax2.set_ylabel('R', fontweight='bold')                         # Y轴标签加粗
ax2.set_ylim(0.74, 0.86)                                       # Y轴范围匹配原图
ax2.grid(True, linestyle='--', alpha=0.7)                      # 添加网格线
ax2.legend(prop={'weight': 'bold'})                            # 图例文字加粗
# 明确设置x轴刻度为数据值
ax2.set_xticks(x_dim)
ax2.set_xticklabels(x_dim, fontweight='bold')  # 刻度值加粗

# 调整布局，避免元素重叠
plt.tight_layout()

# ---------------------- 保存为SVG矢量图 ----------------------
plt.savefig('layer_dimension_results.svg', format='svg', dpi=300)

# 显示图形（可选）
plt.show()