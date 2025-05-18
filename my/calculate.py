"""
多目标优化函数：
"""

import math


def calculate_voyage_distance(uav, task):
    """
    计算剩余航程
    """
    x1, y1 = uav.location
    x2, y2 = task.location
    distance = math.hypot(x2 - x1, y2 - y1)
    return distance


def calculate_voyage_time(uav, task):
    """
    计算航行时间
    time = distance / speed
    """
    distance = calculate_voyage_distance(uav, task)
    return distance / uav.speed


def calculate_fitness(task, uav):
    """
    任务收益：结合目标(任务)价值、需求与无人机能力
    reward = task.value * capability_match_factor
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
    return task.value * factor


def check_constraints(uav, task):
    """
    检查各类约束
    1. 类型约束
    2. 弹药
    3. 航程
    4. 任务时序 & 时间资源
    5. 弹药资源
    """
    if uav.status != 0:
        return False
    # 类型约束
    if task.type in (2, 3) and uav.type != 2:
        return False
    if task.type == 1 and uav.type != 1:
        return False
    # 弹药需求
    if task.ammunition > uav.ammunition:
        return False
    # 时间资源约束
    if task.time > uav.time:
        return False
    # 航程约束
    x1, y1 = uav.location
    x2, y2 = task.location
    distance = math.hypot(x2 - x1, y2 - y1)
    if distance > uav.voyage:
        return False
    return True


def calculate_reward(uav, task, target, alpha=1.0, beta=1.0, gamma=1.0):
    """
    综合三项指标计算最终 reward:
      reward = alpha * 任务收益
             - beta * 威胁代价
             - gamma * 航行时间惩罚

    参数:
    alpha, beta, gamma: 权重系数，可根据策略重点调节
    """
    # 先检查约束，若不满足则给一个大负奖励
    # if not check_constraints(uav, task):
    #     return 0

    fit = calculate_fitness(task, uav)

    return alpha * fit
