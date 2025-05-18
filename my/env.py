import pandas as pd
import random
from typing import List, Tuple


class Uav:
    """
    异质无人机节点拥有的属性：
    - id: 节点ID
    - type: 节点类型 (1打击、2侦查、3评估)
    - status: 节点状态 (0空闲、1执行任务)
    - location: 节点位置
    - strike: 打击能力
    - reconnaissance: 侦察能力
    - assessment: 评估能力
    - ammunition: 弹药量资源
    - time: 侦查评估任务时间资源
    - voyage: 航程
    - speed: 飞行速度
    - value: 节点价值
    - available_time: 可用时间
    """

    def __init__(
        self,
        id: str,
        type: int,
        status: int,
        location: Tuple[float, float],
        strike: float,
        reconnaissance: float,
        assessment: float,
        ammunition: int,
        time: float,
        voyage: float,
        speed: float,
        value: float,
        available_time: float = 0.0,
    ):
        self.id = id
        self.type = type
        self.status = status
        self.location = (round(location[0], 2), round(location[1], 2))
        self.strike = round(strike, 2)
        self.reconnaissance = round(reconnaissance, 2)
        self.assessment = round(assessment, 2)
        self.ammunition = ammunition
        self.time = time
        self.voyage = round(voyage, 2)
        self.speed = round(speed, 2)
        self.value = round(value, 2)
        self.available_time = round(available_time, 2)


class Task:
    """
    任务拥有的属性：
    - id: 任务ID
    - type: 任务类型 (1打击、2侦查、3评估)
    - location: 任务位置
    - strike: 打击需求
    - reconnaissance: 侦察需求
    - assessment: 评估需求
    - ammunition: 打击任务弹药需求
    - time: 侦查评估任务时间需求
    - value: 任务价值
    """

    def __init__(
        self,
        id: str,
        type: int,
        location: Tuple[float, float],
        strike: float,
        reconnaissance: float,
        assessment: float,
        ammunition: int,
        time: float,
        value: float,
    ):
        self.id = id
        self.type = type
        self.location = (round(location[0], 2), round(location[1], 2))
        self.strike = round(strike, 2)
        self.reconnaissance = round(reconnaissance, 2)
        self.assessment = round(assessment, 2)
        self.ammunition = ammunition
        self.time = time
        self.value = round(value, 2)


class Target:
    """
    目标节点拥有的属性：
    - id: 目标ID
    - tasks: 目标任务集合, 多个不同数量不同类型的任务
    - location: 目标位置
    - threaten: 威胁值
    """

    def __init__(
        self,
        id: str,
        tasks: List[Task],
        location: Tuple[float, float],
    ):
        self.id = id
        self.tasks = tasks
        self.location = (round(location[0], 2), round(location[1], 2))


def parse_location(loc_str: str) -> Tuple[float, float]:
    x, y = loc_str.strip("()").split(",")
    return float(x), float(y)


def load_uavs(csv_path: str) -> List[Uav]:
    df = pd.read_csv(csv_path)
    uavs: List[Uav] = []
    for _, row in df.iterrows():
        uav = Uav(
            id=row["id"],
            type=int(row["type"]),
            status=0,
            location=parse_location(row["location"]),
            strike=float(row["strike"]),
            reconnaissance=float(row["reconnaissance"]),
            assessment=float(row["assessment"]),
            ammunition=int(row["ammunition"]),
            time=int(row["time"]),
            voyage=float(row["voyage"]),
            speed=float(row["speed"]),
            value=float(row["value"]),
        )
        uavs.append(uav)
    return uavs


def load_tasks(csv_path: str) -> List[Task]:
    df = pd.read_csv(csv_path)
    tasks: List[Task] = []
    for _, row in df.iterrows():
        task = Task(
            id=row["id"],
            type=int(row["type"]),
            location=parse_location(row["location"]),
            strike=float(row["strike"]),
            reconnaissance=float(row["reconnaissance"]),
            assessment=float(row["assessment"]),
            ammunition=int(row["ammunition"]),
            time=int(row["time"]),
            value=float(row["value"]),
        )
        tasks.append(task)
    return tasks


def initialize_targets(tasks: List[Task]) -> List[Target]:
    # 按类型分组
    tasks_by_type = {1: [], 2: [], 3: []}
    for task in tasks:
        tasks_by_type[task.type].append(task)
    # 计算可生成的目标数（每个目标一个每类型）
    n_targets = min(len(tasks_by_type[1]), len(tasks_by_type[2]), len(tasks_by_type[3]))
    targets: List[Target] = []
    for i in range(n_targets):
        # 每个目标包含一种类型的任务
        t1 = tasks_by_type[1][i]
        t2 = tasks_by_type[2][i]
        t3 = tasks_by_type[3][i]
        # 目标位置可取三任务平均位置
        avg_x = (t1.location[0] + t2.location[0] + t3.location[0]) / 3
        avg_y = (t1.location[1] + t2.location[1] + t3.location[1]) / 3
        target = Target(
            id=f"TARGET{i+1:02d}",
            tasks=[t1, t2, t3],
            location=(avg_x, avg_y)
        )
        targets.append(target)
    return targets
