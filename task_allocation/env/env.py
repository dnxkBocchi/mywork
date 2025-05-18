class Node:
    """
    异质无人机节点拥有的属性：
    - id: 节点ID
    - type: 节点类型 (1打击、2侦查、3评估)
    - location: 节点位置
    - strike: 打击能力
    - reconnaissance: 侦察能力
    - assessment: 评估能力
    - ammunition: 弹药量资源
    - time: 侦查评估任务时间资源
    - voyage: 航程
    - speed: 飞行速度
    - value: 节点价值
    """

    def __init__(
        self,
        id,
        type,
        location,
        strike,
        reconnaissance,
        assessment,
        ammunition,
        time,
        voyage,
        speed,
        value,
    ):
        self.id = id
        self.type = type
        self.location = location
        self.strike = strike
        self.reconnaissance = reconnaissance
        self.assessment = assessment
        self.ammunition = ammunition
        self.time = time
        self.voyage = voyage
        self.speed = speed
        self.value = value


class Target:
    """
    目标节点拥有的属性：
    - id: 目标ID
    - tasks: 目标任务集合, 多个不同数量不同类型的任务
    - location: 目标位置
    - threaten: 威胁值
    """

    def __init__(self, id, tasks, location, threaten):
        self.id = id
        self.tasks = tasks
        self.location = location
        self.threaten = threaten


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
        id,
        type,
        location,
        strike,
        reconnaissance,
        assessment,
        ammunition,
        time,
        value,
    ):
        self.id = id
        self.type = type
        self.location = location
        self.strike = strike
        self.reconnaissance = reconnaissance
        self.assessment = assessment
        self.ammunition = ammunition
        self.time = time
        self.value = value
