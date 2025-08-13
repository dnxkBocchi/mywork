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
reward = os.path.join(current_script_dir, "..", "plt", "rewards_per10_episode.txt")

# 读取文件
with open(reward, "r") as f:
    lines = f.readlines()

# 转成二维数组
data = [list(map(float, line.split())) for line in lines]

# 每组数据
y1 = np.array(data[0])
y2 = np.array(data[1])

# x 轴坐标：0, 10, 20, ..., 2000
x = np.arange(len(y1)) * 10

def smooth(y, box_pts=5):
    box = np.ones(box_pts)/box_pts
    y_smooth = np.convolve(y, box, mode='same')
    return y_smooth

# 平滑处理
y1_smooth = smooth(y1, box_pts=5)
y2_smooth = smooth(y2, box_pts=5)

# 再画图
# plt.figure(figsize=(10, 6))
# plt.plot(x, y1_smooth, label="Algorithm 1", color='blue')
# plt.plot(x, y2_smooth, label="Algorithm 2", color='orange')

# plt.xlabel("Episodes")
# plt.ylabel("Reward")
# plt.title("Reward Comparison (Smoothed)")
# plt.legend()
# plt.grid(True)
# plt.show()

# 绘图
plt.figure(figsize=(8, 5))
plt.plot(x, y2, label="DRL", marker='o', markersize=4)
plt.plot(x, y1, label="GMP-DRL", marker='s', markersize=4)

plt.xlabel("Episodes")
plt.ylabel("Reward")
plt.title("Reward Comparison between DRL and GMP-DRL")
plt.legend()
plt.grid(True)

plt.show()
