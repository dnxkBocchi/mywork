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
# 拼接 uav.csv 的绝对路径（根据实际目录结构调整）
# 假设脚本在 "scripts/" ，uav.csv 在 "data/train/"，两者同级
uav_csv = os.path.join(current_script_dir, "..", "data", "train", "uav.csv")
task_csv = os.path.join(current_script_dir, "..", "data", "train", "task.csv")
# test
uav_csv = os.path.join(current_script_dir, "..", "data", "test", "uav.csv")
task_csv = os.path.join(current_script_dir, "..", "data", "test", "task.csv")
# 规范化路径（处理 .. 等相对路径符号）
uav_csv = os.path.normpath(uav_csv)
task_csv = os.path.normpath(task_csv)

# 现在可以导入env了
from env import *

plt.rcParams["font.family"] = ["Times New Roman", "serif"]
plt.rcParams["axes.unicode_minus"] = False

# 加载不同规模的无人机、任务和目标数据
# uavs, tasks, targets = load_different_scale_csv(uav_csv, task_csv, 5) # 小规模数据
uavs, tasks, targets = load_different_scale_csv(uav_csv, task_csv, 10) # 中规模数据
# uavs, tasks, targets = load_different_scale_csv(uav_csv, task_csv, 20) # 大规模数据

# 无人机坐标
uav_locations = [uav.location for uav in uavs]
# 目标坐标
target_locations = [target.location for target in targets]

# 划分无人机类型：3架侦察型，3架打击型，4架评估型
recon_uav = uav_locations[:3]    # 侦察型（前3架）
strike_uav = uav_locations[3:6]  # 打击型（4-6架）
eval_uav = uav_locations[6:10]     # 评估型（7-10架）

# 绘制图形
plt.figure(figsize=(6, 6))
# 绘制任务目标（红色圆点）
target_x, target_y = zip(*target_locations)
plt.scatter(target_x, target_y, c='red', s=100, marker='o', label='Target')

# 绘制不同类型无人机
recon_x, recon_y = zip(*recon_uav)
plt.scatter(recon_x, recon_y, c='blue', s=120, marker='^', label='Recon UAV')

strike_x, strike_y = zip(*strike_uav)
plt.scatter(strike_x, strike_y, c='green', s=120, marker='s', label='Strike UAV')

eval_x, eval_y = zip(*eval_uav)
plt.scatter(eval_x, eval_y, c='purple', s=120, marker='D', label='Eval UAV')

# 设置坐标轴范围（100×100）
plt.xlim(0, 100)
plt.ylim(0, 100)
plt.xlabel('X Coordinate')
plt.ylabel('Y Coordinate')
plt.title('Distribution of Task Targets and UAV Positions')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.7)
plt.savefig(os.path.join(current_script_dir, "..", "pic", "uav_distribution.png"), dpi=300, bbox_inches='tight')
plt.show()