import matplotlib.pyplot as plt
import numpy as np
import sys
import os

# 获取当前脚本所在目录（plt文件夹）
plt_dir = os.path.dirname(os.path.abspath(__file__))
# 获取my目录（plt的父目录）
my_dir = os.path.dirname(plt_dir)
# 将my目录添加到搜索路径
sys.path.append(my_dir)

# 获取当前脚本所在的绝对目录
current_script_dir = os.path.dirname(os.path.abspath(__file__))
import matplotlib.pyplot as plt
import numpy as np
import sys
import os

# 路径配置（确保文件读取和模块导入正常）
# 获取当前脚本所在目录（plt文件夹）
plt_dir = os.path.dirname(os.path.abspath(__file__))
# 获取my目录（plt的父目录）并添加到搜索路径（确保能导入env模块）
my_dir = os.path.dirname(plt_dir)
sys.path.append(my_dir)

# 全局字体配置（符合期刊学术规范）
plt.rcParams["font.family"] = ["Times New Roman", "serif"]
plt.rcParams["axes.unicode_minus"] = False  # 解决负号显示异常
plt.rcParams["font.size"] = 10  # 基础字体大小，保证标签协调

# 1. 数据文件路径处理（增加容错校验）
# 拼接测试集数据路径（规范化路径，处理../等符号）
uav_csv = os.path.normpath(os.path.join(current_script_dir, "..", "data", "test", "uav.csv"))
task_csv = os.path.normpath(os.path.join(current_script_dir, "..", "data", "test", "task.csv"))

# 校验数据文件是否存在
if not os.path.exists(uav_csv):
    print(f"错误：无人机数据文件不存在 → {uav_csv}")
    sys.exit(1)
if not os.path.exists(task_csv):
    print(f"错误：任务数据文件不存在 → {task_csv}")
    sys.exit(1)

# 2. 加载数据（依赖env模块的load_different_scale_csv函数）
try:
    from env import load_different_scale_csv  # 导入环境模块
    # 加载中规模数据（10架无人机，可按需切换5/20规模）
    uavs, tasks, targets = load_different_scale_csv(uav_csv, task_csv, 10)
except ImportError:
    print("错误：无法导入env模块，请检查my目录是否已添加到搜索路径")
    sys.exit(1)
except Exception as e:
    print(f"错误：数据加载失败 → {e}")
    sys.exit(1)

# 3. 提取坐标数据（确保数据格式正确）
# 无人机坐标（按类型划分：侦察型3架、打击型3架、评估型4架）
try:
    uav_locations = [uav.location for uav in uavs]
    recon_uav = uav_locations[:3]    # 侦察型无人机（前3架）
    strike_uav = uav_locations[3:6]  # 打击型无人机（4-6架）
    eval_uav = uav_locations[6:10]   # 评估型无人机（7-10架）
    # 目标坐标
    target_locations = [target.location for target in targets]
except AttributeError as e:
    print(f"错误：数据对象属性异常（可能是env模块返回格式变化）→ {e}")
    sys.exit(1)

# 4. 绘制黑白风格散点图（核心：用标记形状+填充/边框区分类别）
plt.figure(figsize=(6, 6))  # 正方形画布，保证坐标比例不失真

# 4.1 绘制任务目标（黑色实心圆点，突出显示）
target_x, target_y = zip(*target_locations)
plt.scatter(
    target_x, target_y,
    color='black',          # 填充色：纯黑（目标需醒目）
    s=100,                  # 大小：与原始一致（100）
    marker='o',             # 形状：圆形（区分无人机）
    edgecolor='black',      # 边框：纯黑（增强轮廓）
    linewidth=1.2,          # 边框宽度：避免打印模糊
    label='Target'          # 图例标签
)

# 4.2 绘制侦察型无人机（三角形空心，与目标区分）
recon_x, recon_y = zip(*recon_uav)
plt.scatter(
    recon_x, recon_y,
    color='none',           # 填充色：空心（无颜色）
    s=120,                  # 大小：120（比目标略大，便于识别）
    marker='^',             # 形状：正三角形（唯一标识侦察型）
    edgecolor='black',      # 边框：纯黑
    linewidth=1.5,          # 边框宽度：比目标粗（增强区分度）
    label='Recon UAV'       # 图例标签
)

# 4.3 绘制打击型无人机（正方形实心，与其他类别区分）
strike_x, strike_y = zip(*strike_uav)
plt.scatter(
    strike_x, strike_y,
    color='black',          # 填充色：纯黑（实心，区分空心侦察型）
    s=120,                  # 大小：与侦察型一致（视觉平衡）
    marker='s',             # 形状：正方形（唯一标识打击型）
    edgecolor='black',      # 边框：纯黑
    linewidth=1.5,          # 边框宽度：与侦察型一致
    label='Strike UAV'      # 图例标签
)

# 4.4 绘制评估型无人机（菱形空心，与其他类别区分）
eval_x, eval_y = zip(*eval_uav)
plt.scatter(
    eval_x, eval_y,
    color='none',           # 填充色：空心（区分实心打击型）
    s=120,                  # 大小：与其他无人机一致
    marker='D',             # 形状：菱形（唯一标识评估型）
    edgecolor='black',      # 边框：纯黑
    linewidth=1.5,          # 边框宽度：与其他无人机一致
    label='Eval UAV'        # 图例标签
)

# 5. 图表细节优化（符合期刊出版要求）
# 坐标轴范围：固定100×100（与原始逻辑一致，保证位置分布准确）
plt.xlim(0, 100)
plt.ylim(0, 100)
# 坐标轴标签：12号字体，增加间距避免拥挤
plt.xlabel('X Coordinate', fontsize=12, labelpad=10)
plt.ylabel('Y Coordinate', fontsize=12, labelpad=10)
# 标题：14号加粗，突出主题
plt.title('Distribution of Task Targets and UAV Positions', fontsize=14, pad=20, fontweight='bold')
# 图例：自动优化位置，边框半透明（避免遮挡数据）
plt.legend(fontsize=11, loc='best', frameon=True, framealpha=0.8)
# 网格线：虚线+低透明度（辅助读坐标，不干扰数据点）
plt.grid(True, linestyle='--', alpha=0.6, color='black')

# 6. 保存与显示（确保高清无截断）
# 保存目录：my/pic（自动创建，避免路径不存在报错）
save_dir = os.path.join(my_dir, "pic")
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
save_path = os.path.join(save_dir, "uav_distribution.svg")

# 紧凑保存：去除多余空白，300dpi高清（期刊要求）
plt.tight_layout()
plt.savefig(save_path, dpi=300, bbox_inches='tight', format='svg')
print(f"黑白风格图表已保存 → {save_path}")

# 可选：显示图表（若环境支持GUI）
try:
    plt.show()
except Exception:
    print("提示：当前环境不支持显示图表（仅保存文件）")