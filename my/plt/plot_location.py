import matplotlib.pyplot as plt
import numpy as np

# 设置中文显示
plt.rcParams["font.family"] = ["SimHei", "WenQuanYi Micro Hei", "Heiti TC"]
plt.rcParams["axes.unicode_minus"] = False

# 任务目标坐标
target_coords = [
    (95.74, 96.10), (80.33, 88.26), (92.78, 90.41),
    (86.98, 91.84), (90.83, 96.13), (80.47, 81.44),
    (82.41, 80.78), (90.74, 95.29), (91.74, 96.32),
    (82.62, 83.64)
]

# 无人机坐标（10架）
uav_coords = [
    (12.89, 5.0), (5.96, 4.73), (1.62, 7.71),
    (9.16, 4.58), (3.23, 5.83), (5.64, 7.86),
    (6.32, 8.55), (6.82, 2.53), (6.86, 3.66),
    (8.58, 7.27)
]

# 划分无人机类型：3架侦察型，3架打击型，4架评估型
recon_uav = uav_coords[:3]    # 侦察型（前3架）
strike_uav = uav_coords[3:6]  # 打击型（4-6架）
eval_uav = uav_coords[6:]     # 评估型（7-10架）

# 绘制图形
plt.figure(figsize=(8, 8))
# 绘制任务目标（红色圆点）
target_x, target_y = zip(*target_coords)
plt.scatter(target_x, target_y, c='red', s=100, marker='o', label='任务目标')

# 绘制不同类型无人机
recon_x, recon_y = zip(*recon_uav)
plt.scatter(recon_x, recon_y, c='blue', s=120, marker='^', label='侦察型无人机')

strike_x, strike_y = zip(*strike_uav)
plt.scatter(strike_x, strike_y, c='green', s=120, marker='s', label='打击型无人机')

eval_x, eval_y = zip(*eval_uav)
plt.scatter(eval_x, eval_y, c='purple', s=120, marker='D', label='评估型无人机')

# 设置坐标轴范围（100×100）
plt.xlim(0, 100)
plt.ylim(0, 100)
plt.xlabel('X坐标')
plt.ylabel('Y坐标')
plt.title('任务目标与无人机位置分布')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.7)
plt.show()