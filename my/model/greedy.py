import random
from calculate import *

def select_uav_by_matching(task, uavs) -> int:
    """
    基于任务类型与无人机能力匹配度的选择策略
    匹配规则：
    - 侦察任务 → 选择reconnaissance能力最高的无人机
    - 打击任务 → 选择strike能力最高的无人机
    - 评估任务 → 选择assessment能力最高的无人机
    """
    # 筛选类型匹配的无人机
    valid_uavs = []
    for idx, uav in enumerate(uavs):
        if check_constraints(uav, task):
            valid_uavs.append((idx, uav))
    if not valid_uavs:
        # return random.choice(uavs)  # 如果没有匹配的无人机，随机选择一个
        return random.choice(range(len(uavs)))  # 如果没有匹配的无人机，随机选择一个
    # 根据任务类型获取匹配能力值
    if task.type == 1:
        key = lambda x: x[1].strike
    elif task.type == 2:
        key = lambda x: x[1].reconnaissance
    else:
        key = lambda x: x[1].assessment
    # 选择能力值最高的无人机索引
    # return max(valid_uavs, key=key)[1]
    return max(valid_uavs, key=key)[0]


def select_uav_by_voyage(task, uavs) -> int:
    """
    基于最短飞行距离的选择策略
    """
    valid_uavs = []
    for idx, uav in enumerate(uavs):
        if check_constraints(uav, task):
            distance = calculate_voyage_distance(uav, task)
            valid_uavs.append((idx, uav, distance))
    if not valid_uavs:
        # return random.choice(uavs) 
        return random.choice(range(len(uavs)))
    # 选择距离最短的无人机索引
    # return min(valid_uavs, key=lambda x: x[2])[1]
    return min(valid_uavs, key=lambda x: x[2])[0]


def select_uav_by_time(task, uavs) -> int:
    """
    基于最短完成时间的选择策略（飞行时间+任务执行时间）
    """
    valid_uavs = []
    for idx, uav in enumerate(uavs):
        if check_constraints(uav, task):
            time = calculate_voyage_time(uav, task)  # 计算飞行时间
            valid_uavs.append((idx, uav, time))
    if not valid_uavs:
        # return random.choice(uavs) 
        return random.choice(range(len(uavs)))
    # 选择总时间最短的无人机索引
    # return min(valid_uavs, key=lambda x: x[2])[1]
    return min(valid_uavs, key=lambda x: x[2])[0]