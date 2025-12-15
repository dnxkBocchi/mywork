import matplotlib.pyplot as plt
import numpy as np
import sys
import os

# 路径配置（确保文件读取和保存路径正确）
# 获取当前脚本所在目录（plt文件夹）
plt_dir = os.path.dirname(os.path.abspath(__file__))
# 获取my目录（plt的父目录）并添加到搜索路径
my_dir = os.path.dirname(plt_dir)
sys.path.append(my_dir)

# 全局字体配置（符合期刊规范，避免中文依赖）
plt.rcParams["font.family"] = ["Times New Roman", "serif"]
plt.rcParams["axes.unicode_minus"] = False  # 解决负号显示异常
plt.rcParams["font.size"] = 10  # 基础字体大小，保证标签协调

# 1. 读取奖励数据（增加路径校验，避免文件找不到）
reward_file = os.path.join(plt_dir, "rewards_per10_episode.txt")  # 简化路径：plt目录下直接读取
if not os.path.exists(reward_file):
    print(f"错误：奖励数据文件不存在 → {reward_file}")
    sys.exit(1)

# 读取并解析数据
with open(reward_file, "r", encoding="utf-8") as f:
    lines = [line.strip() for line in f.readlines() if line.strip()]

# 校验数据格式（确保至少2行数据，对应两种算法）
if len(lines) < 2:
    print(f"错误：数据行数不足（需2行，实际{len(lines)}行）")
    sys.exit(1)

# 转换为数值数组（处理每行数据长度不一致的情况）
try:
    data = [list(map(float, line.split())) for line in lines[:2]]  # 只取前2行（对应DRL和GMP-DRL）
    # 确保两行数据长度一致（避免绘图时索引错误）
    min_len = min(len(data[0]), len(data[1]))
    data[0] = data[0][:min_len]
    data[1] = data[1][:min_len]
except ValueError as e:
    print(f"错误：数据格式无效 → {e}")
    sys.exit(1)

# 2. 数据预处理
y1 = np.array(data[0])  # DRL 原始奖励
y2 = np.array(data[1])  # GMP-DRL 原始奖励
x = np.arange(len(y1)) * 10  # x轴：每10个episode为间隔（与文件名对应）

# （可选）平滑处理函数（如需启用，取消下方调用注释即可）
def smooth(y, box_pts=5):
    box = np.ones(box_pts) / box_pts
    # 处理边界：避免平滑后数据长度变化
    y_smooth = np.convolve(y, box, mode='same')
    # 保持首尾数据不变（避免边界失真）
    y_smooth[0] = y[0]
    y_smooth[-1] = y[-1]
    return y_smooth

# （可选）启用平滑：若需要平滑曲线，替换下方y1/y2为平滑后的数据
# y1 = smooth(y1, box_pts=5)
# y2 = smooth(y2, box_pts=5)

# 3. 绘制黑白风格折线图
plt.figure(figsize=(8, 5))  # 图表尺寸（宽8英寸，高5英寸，符合期刊常用比例）

# 黑白风格核心配置：用「线条样式+标记形状+填充模式」区分两种算法
# DRL：实线 + 圆形空心标记
plt.plot(
    x, y1,
    label="DRL",
    color="black",          # 线条颜色：纯黑
    linestyle="-",          # 线条样式：实线
    linewidth=1.5,          # 线条宽度：适中（保证打印清晰）
    marker="o",             # 标记形状：圆形
    markersize=4,           # 标记大小：与原始一致
    markerfacecolor="none", # 标记填充：空心（区分于GMP-DRL）
    markeredgecolor="black",# 标记边框：纯黑
    markeredgewidth=1.2     # 标记边框宽度：增强轮廓（避免打印模糊）
)

# GMP-DRL：虚线 + 方形实心标记（与DRL形成明显对比）
plt.plot(
    x, y2,
    label="GMP-DRL",
    color="black",          # 线条颜色：纯黑（统一黑白风格）
    linestyle="--",         # 线条样式：虚线（与DRL实线区分）
    linewidth=1.5,          # 线条宽度：与DRL一致（保证视觉平衡）
    marker="s",             # 标记形状：方形（与DRL圆形区分）
    markersize=4,           # 标记大小：与DRL一致
    markerfacecolor="black",# 标记填充：实心（与DRL空心区分）
    markeredgecolor="black",# 标记边框：纯黑
    markeredgewidth=1.2     # 标记边框宽度：与DRL一致
)

# 4. 图表细节优化（符合期刊出版要求）
plt.xlabel("Episodes", fontsize=12, labelpad=10)  # x轴标签：字体12号，增加间距避免拥挤
plt.ylabel("Reward", fontsize=12, labelpad=10)    # y轴标签：同上
plt.title("Reward Comparison between DRL and GMP-DRL", fontsize=14, pad=20, fontweight="bold")  # 标题：加粗突出

# 网格线：仅y轴显示虚线网格（辅助读数，不干扰线条）
plt.grid(axis="y", linestyle="--", alpha=0.6, color="black")  # 网格颜色：黑色（弱化透明度，避免杂乱）

# 图例：位置自动优化，字体大小适配
plt.legend(fontsize=11, loc="best", frameon=True, framealpha=0.8)  # 带浅色边框，增强区分度

# 5. 保存与显示（确保高清且无截断）
save_dir = os.path.join(my_dir, "pic")  # 保存目录：my/pic（与之前代码逻辑一致）
if not os.path.exists(save_dir):
    os.makedirs(save_dir)  # 自动创建目录（避免路径不存在报错）

save_path = os.path.join(save_dir, "reward_comparison_20.svg")
plt.tight_layout()  # 自动调整布局：避免标签、标题被截断
plt.savefig(
    save_path,
    dpi=300,               # 分辨率：300dpi（期刊高清要求）
    bbox_inches="tight",   # 紧凑保存：去除多余空白
    format="svg"           # 格式：SVG（矢量图，放大不失真）
)
print(f"黑白风格图表已保存 → {save_path}")

# 可选：显示图表（若运行环境支持GUI）
try:
    plt.show()
except Exception:
    print("提示：当前环境不支持显示图表（仅保存文件）")