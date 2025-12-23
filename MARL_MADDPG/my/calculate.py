"""
多目标优化函数：
"""

import math

debug = False


def calculate_back_voyage(uav):
    """
    计算无人机返回基地的航程
    """
    x1, y1 = uav._init_location
    x2, y2 = uav.location
    distance = math.hypot(x2 - x1, y2 - y1)
    return distance


def calculate_all_voyage_distance(uavs):
    """
    计算所有无人机的航程
    """
    total_distance = 0
    for uav in uavs:
        distance = uav._init_voyage - uav.voyage
        total_distance += distance + calculate_back_voyage(uav)
    return total_distance


def calculate_voyage_distance(uav, task):
    """
    计算剩余航程
    """
    x1, y1 = uav.location
    x2, y2 = task.location
    distance = math.hypot(x2 - x1, y2 - y1)
    return distance


def calculate_all_voyage_time(targets):
    """
    计算所有无人机的航行时间
    """
    total_time = 0
    for target in targets:
        total_time += target.total_time
    return total_time


def calculate_voyage_time(uav, task):
    """
    计算航行时间
    time = distance / speed
    """
    distance = calculate_voyage_distance(uav, task)
    return distance / uav.speed


def calculate_max_possible_voyage_time(uavs, targets):
    """
    计算理论最大航程和最大完成时间：假设每个无人机从初始位置出发，依次访问所有任务点
    并返回基地（如果需要返回）
    """
    max_voyage = 0
    # 收集所有任务点的位置
    all_task_locations = []
    for target in targets:
        all_task_locations.append(target.location)
    # 对每个无人机，计算其可能的最大航程
    for uav in uavs:
        current_pos = uav.location
        current_voyage = 0
        # 计算从当前位置到所有任务点的往返距离
        for task_location in all_task_locations:
            # 到任务点的距离
            distance_to_task = math.hypot(
                task_location[0] - current_pos[0], task_location[1] - current_pos[1]
            )
            # 假设完成任务后不返回该位置（模拟最坏情况）
            current_voyage += distance_to_task
            current_pos = task_location  # 更新当前位置
        max_voyage = max(max_voyage, current_voyage)
    min_speed = min(uav.speed for uav in uavs) if uavs else 1
    max_total_time = max_voyage / min_speed  # 最小速度下的最大时间
    return max_voyage, max_total_time


def calculate_fitness_r(task, uav):
    """
    任务无人机适配度奖励：结合目标(任务)价值、需求与无人机能力
    reward = capability_match_factor
    capability_match_factor 考虑 uav 对任务需求的满足程度
    """
    # 能力匹配程度 = min( uav.capacity / task.requirement ) 取平均
    factors = []
    if task.type == 1 and uav.type == 1:  # 打击
        factors.append(min(uav.strike / task.strike, 1))
    if task.type == 2 and uav.type == 2:  # 侦察
        factors.append(min(uav.reconnaissance / task.reconnaissance, 1))
    if task.type == 3 and uav.type == 3:  # 评估
        factors.append(min(uav.assessment / task.assessment, 1))
    # 多能力取平均（兼顾多需求的任务）
    factor = sum(factors) / len(factors) if factors else 0
    return factor


def calculate_voyage_r(task, uav, max_total_voyage):
    """
    计算任务航程奖励：
    """
    voyage_distance = calculate_voyage_distance(uav, task)
    voyage_r = 1 - (voyage_distance / max_total_voyage)  # 航程越短奖励越高
    return voyage_r


def calculate_time_r(task, uav, max_total_time):
    """
    计算任务完成时间奖励：
    """
    voyage_time = calculate_voyage_time(uav, task)
    time_r = 1 - (voyage_time / max_total_time)  # 时间越短奖励越高
    return time_r


def check_constraints(uav, task):
    """
    检查各类约束
    1. 类型约束
    2. 弹药资源
    3. 时间资源
    4. 航程资源
    5. 时序要求
    """
    # 类型约束
    if task.type == 3 and uav.type != 3:
        return False
    if task.type == 2 and uav.type != 2:
        return False
    if task.type == 1 and uav.type != 1:
        return False
    # 弹药需求
    if task.ammunition > uav.ammunition:
        return False
    # 时间资源约束
    if task.time > uav.time:
        return False
    # 时序要求
    target = task.target
    for t in target.tasks:
        if t != task and t.flag == False:
            return False  # 前面任务未完成，不能执行
        elif t == task:
            break  # 前面任务已完成
    # 航程约束
    distance = calculate_voyage_distance(uav, task)
    if distance > uav.voyage:
        return False
    return True


def calculate_reward(
    uav,
    task,
    target,
    max_total_voyage,
    max_total_time,
    global_step=100,
    alpha=0.2,
    beta=0.4,
    gamma=0.4,
):
    """
    综合三项指标计算最终 reward:
      reward = alpha * 任务无人机适配度
             - beta * 总航程惩罚
             - gamma * 航行时间惩罚

    参数:
    alpha, beta, gamma: 权重系数，可根据策略重点调节
    """
    # 先检查约束，若不满足则给一个大负奖励
    if not check_constraints(uav, task):
        return 0

    fit_r = calculate_fitness_r(task, uav)
    voyage_r = calculate_voyage_r(task, uav, max_total_voyage)
    time_r = calculate_time_r(task, uav, max_total_time)
    if debug:
        print(
            f"fit_r: {fit_r:.2f}, voyage_r: {voyage_r:.2f}, time_r: {time_r:.2f}, "
            f"uav: {uav.id}, task: {task.id}"
        )

    return alpha * fit_r + beta * voyage_r + gamma * time_r


def log_all_voyage_time(uavs, targets):
    """
    计算所有无人机的航程和时间，并将每组数据保存到txt文件
    """
    # 打开文件准备写入，使用with语句确保文件正确关闭
    with open("plt/time_voyage.txt", "a", encoding="utf-8") as f:
        f.write("voyage: ")
        for i, uav in enumerate(uavs, 1):
            distance = uav._init_voyage - uav.voyage
            distance += calculate_back_voyage(uav)  # 加上返回基地的航程
            # 记录每组数据
            f.write(f"{distance:.2f}, ")
        f.write("\ntime: ")  # 每组数据后换行
        for i, target in enumerate(targets, 1):
            # 记录每组数据
            f.write(f"{target.total_time:.2f}, ")
        f.write("\ntasks: ")  # 每组数据后换行
        for i, uav in enumerate(uavs, 1):
            f.write(f"{uav.task_nums}, ")
        f.write("\n")


def log_total_method(
    total_reward, total_fitness, total_distance, total_time, total_success
):
    """
    记录每集数据到文件
    """
    with open("plt/total_method.txt", "a", encoding="utf-8") as f:
        f.write(
            f"{total_reward:.2f}, {total_fitness:.2f}, {total_distance:.2f}, {total_time:.2f}, {total_success:.2f}\n"
        )


def log_n():
    with open("plt/time_voyage.txt", "a", encoding="utf-8") as f:
        f.write(f"\n")
    with open("plt/total_method.txt", "a", encoding="utf-8") as f:
        f.write(f"\n")
    with open("plt/spend_time.txt", "a", encoding="utf-8") as f:
        f.write(f"\n")


def log_time(elapsed_time):
    """
    将算法耗时记录到txt文件
    """
    with open("plt/spend_time.txt", "a", encoding="utf-8") as f:
        f.write(f"{elapsed_time:.6f}\n")


def log_allocation(uav, task):
    """
    记录 UAV 和任务的分配情况
    """
    with open("plt/allocation.txt", "a", encoding="utf-8") as f:
        f.write(f"UAV {uav.id} allocated to Task {task.id}\n")
