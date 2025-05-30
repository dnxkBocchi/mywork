"""
多目标优化函数：
"""

import math

debug = False


def calculate_all_voyage_distance(uavs):
    """
    计算所有无人机的航程
    """
    total_distance = 0
    for uav in uavs:
        distance = uav._init_voyage - uav.voyage
        total_distance += distance
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
        for task in target.tasks:
            all_task_locations.append(task.location)
    # 对每个无人机，计算其可能的最大航程
    for uav in uavs:
        current_pos = uav.location
        # 计算从当前位置到所有任务点的往返距离
        for task_location in all_task_locations:
            # 到任务点的距离
            distance_to_task = math.hypot(
                task_location[0] - current_pos[0], task_location[1] - current_pos[1]
            )
            # 假设完成任务后不返回该位置（模拟最坏情况）
            max_voyage += distance_to_task
            current_pos = task_location  # 更新当前位置
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
    if task.type == 1:  # 打击
        factors.append(min(uav.strike / task.strike, 1))
    if task.type == 2:  # 侦察
        factors.append(min(uav.reconnaissance / task.reconnaissance, 1))
    if task.type == 3:  # 评估
        factors.append(min(uav.assessment / task.assessment, 1))
    # 多能力取平均（兼顾多需求的任务）
    factor = sum(factors) / len(factors)
    return factor


def calculate_voyage_r(task, uav, max_total_voyage):
    """
    计算任务航程奖励：
    """
    voyage_distance = calculate_voyage_distance(uav, task)
    voyage_r = 1 - (voyage_distance / max_total_voyage)  # 航程越短奖励越高
    return voyage_r


def calculate_time_r(task, uav, max_total_time):
    """ "
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
    5.
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
    uav, task, target, max_total_voyage, max_total_time, alpha=0.4, beta=0.3, gamma=0.3
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

    return alpha * fit_r + beta * voyage_r + gamma * time_r
