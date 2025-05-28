import sys
import os

# 添加项目根目录到 sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from my.calculate import calculate_all_voyage_distance, calculate_all_voyage_time


def round_robin_allocation(env):
    """
    轮询算法：按顺序为每个任务分配同类型的无人机
    """
    # 重置环境
    env.reset()
    done = False
    num_tasks = sum(len(target.tasks) for target in env.targets)

    # 按类型分组无人机
    uav_groups = {}
    for uav in env.uavs:
        if uav.type not in uav_groups:
            uav_groups[uav.type] = []
        uav_groups[uav.type].append(uav)

    # 为每个类型的无人机维护一个指针，记录当前轮询位置
    uav_pointers = {uav_type: 0 for uav_type in uav_groups}

    # 记录分配历史
    allocation_history = []
    # 记录每次实验的数据，判断优化程度
    total_reward = 0
    total_success = 0
    total_distance = 0
    total_time = 0

    while not done:
        # 获取当前任务
        target = env.targets[env.current_target_idx]
        task = target.tasks[env.task_step[target.id]]
        task_type = task.type

        # 选择当前类型的下一个无人机（轮询）
        uav_list = uav_groups[task_type]
        uav_index = env.uavs.index(uav_list[uav_pointers[task_type]])

        # 记录分配
        allocation_history.append((task.id, uav_list[uav_pointers[task_type]].id))

        # 执行动作
        next_state, reward, done, info = env.step(uav_index)
        total_reward += reward
        if reward > 0:
            total_success += 1

        # 更新该类型无人机的指针，循环使用
        uav_pointers[task_type] = (uav_pointers[task_type] + 1) % len(uav_list)

    total_reward /= num_tasks  # 平均每个任务的奖励
    total_success /= num_tasks  # 平均每个任务的成功率
    total_distance = calculate_all_voyage_distance(env.uavs)
    total_time = calculate_all_voyage_time(env.targets)
    print(f"Total Reward: {total_reward:.2f} | Total Distance: {total_distance:.2f} | Total Success : {total_success:.2f} | Total Time: {total_time:.2f}")
    print("Allocation History:")
    for task_id, uav_id in allocation_history:
        print(f"Task {task_id} assigned to UAV {uav_id}")
