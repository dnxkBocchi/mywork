from calculate import calculate_voyage_distance
from env import *
uav_csv = "D:\code\python_project\mywork\my\data/uav.csv"
task_csv = "D:\code\python_project\mywork\my\data/task.csv"
uavs = load_uavs(uav_csv)
target = initialize_targets(load_tasks(task_csv))

uavs[0].voyage -= calculate_voyage_distance(uavs[0], target[0].tasks[0])
print(uavs[0].voyage)

# 这是我的整体框架代码，用的是dqn算法，你能按照我的框架，去写一个粒子群算法吗？我想比较二者的算法优劣，