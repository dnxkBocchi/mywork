import sys
import os
import random

# 添加项目根目录到 sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from my.calculate import *


def random_allocation(env):
    """
    随机分配算法：为每个任务随机分配同类型的无人机
    """
    # 重置环境
    env.reset()
    done = False
    num_tasks = sum(len(target.tasks) for target in env.targets)
    uavs = env.uavs

    # 记录分配历史
    allocation_history = []
    # 记录每次实验的数据，判断优化程度
    total_reward = 0
    total_success = 0
    total_distance = 0
    total_fitness = 0

    while not done:
        # 获取当前任务
        target = env.targets[env.current_target_idx]
        task = target.tasks[env.task_step[target.id]]
        task_type = task.type

        # 随机选择当前类型的无人机
        uav = random.choice(uavs)

        # 记录分配
        allocation_history.append((task.id, uav.id))

        # 执行动作
        index = uavs.index(uav)
        next_state, reward, done, info = env.step(index)
        total_reward += reward
        total_fitness += calculate_fitness_r(task, uav)
        if reward > 0:
            total_success += 1

    total_reward /= num_tasks  # 平均每个任务的奖励
    total_success /= num_tasks  # 平均每个任务的成功率
    total_fitness /= num_tasks  # 平均适配度
    total_distance = calculate_all_voyage_distance(env.uavs)
    total_time = calculate_all_voyage_time(env.targets)
    print(
        f"Total Reward: {total_reward:.2f} | Total Fitness: {total_fitness:.2f} \
| Total Distance: {total_distance:.2f} | Total Time: {total_time:.2f} \
| Total Success : {total_success:.2f}"
    )
    log_all_voyage_time(env.uavs, env.targets)
    log_total_method(total_reward, total_fitness, total_distance, total_time, total_success)
    for task_id, uav_id in allocation_history:
        print(f"Task {task_id} assigned to UAV {uav_id}")
