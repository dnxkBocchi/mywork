import numpy as np
import matplotlib.pyplot as plt
from calculate import *
import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"  # 关键：ipynb 中绘图运行时避免报错
debug = False  # 是否打印调试信息
allocation = False


# 简化版环境示例
class UAVEnv:
    def __init__(self, uavs, targets, tasks, max_neighbors, map_size=100.0):
        self.uavs = uavs
        self.targets = targets
        self.tasks = tasks
        self.task = None  # 当前任务
        self.map_size = map_size
        self.comm_radius = 30.0  # 通信半径
        self.max_neighbors = max_neighbors  # 最大邻居数
        self.done = False

        self.max_voyage = 0
        self.max_speed = 0
        # 后面奖励函数要用到
        self.max_total_voyage = 0
        self.max_total_time = 0
        self.reset()

    def reset(self):
        # 恢复所有 UAV 初始状态
        for uav in self.uavs:
            uav.reset()
        for task in self.tasks:
            task.reset()
        # 重置环境状态
        self.current_target_idx = 0
        self.done = False
        # 为每个 Target 初始化任务进度
        self.task_step = {t.id: 0 for t in self.targets}
        # 重置当前任务
        target = self.targets[self.current_target_idx]
        self.task = target.tasks[self.task_step[target.id]]
        self.max_voyage = max(u.voyage for u in self.uavs) or 1
        self.max_speed = max(u.speed for u in self.uavs) or 1
        # 后面奖励函数要用到
        self.max_total_voyage, self.max_total_time = calculate_max_possible_voyage_time(
            self.uavs, self.targets
        )

        return self._get_state()

    def _get_neighbor_uavs(self, uav):
        """
        获取通信半径内的邻居 UAV (按距离排序)
        """
        neighbors = []
        ux, uy = uav.location
        for other in self.uavs:
            if other.id == uav.id:
                continue
            ox, oy = other.location
            dist = np.hypot(ox - ux, oy - uy)
            if dist <= self.comm_radius:
                neighbors.append((dist, other))
        # 按距离排序
        neighbors.sort(key=lambda x: x[0])
        # 取最近 max_neighbors 个
        return [u for _, u in neighbors[: self.max_neighbors]]

    def _get_capacity(self, u_t):
        capacity = 0
        resource = 0
        if u_t.type == 1:
            capacity = u_t.strike
            resource = u_t.ammunition
        elif u_t.type == 2:
            capacity = u_t.reconnaissance
            resource = u_t.time
        elif u_t.type == 3:
            capacity = u_t.assessment
            resource = u_t.time
        return capacity, resource

    def _normalize_uav(self, u):
        # 类型归一化 (1-3 -> 0-1)
        type_norm = (u.type - 1) / 2
        loc_x = u.location[0] / self.map_size
        loc_y = u.location[1] / self.map_size
        capacity, resource = self._get_capacity(u)
        return [
            type_norm,
            # u.status,
            loc_x,
            loc_y,
            capacity,
            resource,
            u.voyage / self.max_voyage,
            u.speed / self.max_speed,
            u.end_time / self.max_total_time,
            u.task_nums / len(self.uavs),
            # u.value,
        ]

    def _normalize_task(self, task):
        type_norm = (task.type - 1) / 2
        loc_x = task.location[0] / self.map_size
        loc_y = task.location[1] / self.map_size
        capacity, resource = self._get_capacity(task)
        return [
            type_norm,
            loc_x,
            loc_y,
            capacity,
            resource,
            # task.value,
        ]

    def get_agent_obs(self, uav):
        obs = []
        # 1. 自身状态
        obs.extend(self._normalize_uav(uav))
        # 2. 当前任务状态
        target = self.targets[self.current_target_idx]
        task = target.tasks[self.task_step[target.id]]
        obs.extend(self._normalize_task(task))
        # 3. 邻居 UAV 状态
        neighbors = self._get_neighbor_uavs(uav)
        for n in neighbors:
            obs.extend(self._normalize_uav(n))
        # padding
        missing = self.max_neighbors - len(neighbors)
        obs.extend([0.0] * missing * len(self._normalize_uav(uav)))
        return np.array(obs, dtype=np.float32)

    def get_obs_all(self):
        obs_all = []
        for uav in self.uavs:
            obs_all.append(self.get_agent_obs(uav))
        return np.array(obs_all)

    def _get_state(self):
        # 将所有 UAV 归一化后状态 + 当前任务归一化后状态
        uav_states = []
        for u in self.uavs:
            uav_states.extend(self._normalize_uav(u))
        target = self.targets[self.current_target_idx]
        task = target.tasks[self.task_step[target.id]]
        task_state = self._normalize_task(task)

        if debug:
            # 将每个数值格式化为两位小数，并拼成字符串
            us = ", ".join(f"{v:.2f}" for v in uav_states)
            ts = ", ".join(f"{v:.2f}" for v in task_state)
            print(f"uav_states: [{us}], uav_states_len: {len(uav_states)}")
            print(f"task_state: [{ts}], task_state_len: {len(task_state)}")

        # 返回 UAV 状态和任务状态 DQN 输入格式
        return np.array(uav_states + task_state, dtype=np.float32)

    def update_uav_status(self, uav, task):
        # 更新 UAV 状态
        uav.voyage -= calculate_voyage_distance(uav, task)
        uav.task_nums += 1
        # 任务分配后的完成时间
        task.waiting_time = uav.end_time
        uav.end_time += calculate_voyage_time(uav, task)
        if task.type == 2:
            task.end_time = uav.end_time
        elif task.type == 1:
            last_task = task.target.tasks[0]
            task.end_time = uav.end_time + last_task.end_time
        else:
            last_task = task.target.tasks[1]
            task.end_time = uav.end_time + last_task.end_time
        task.flag = True  # 标记任务已完成
        if task.type == 3:  # 评估任务
            task.target.total_time = task.end_time

        uav.location = task.location
        uav.ammunition -= task.ammunition
        uav.time -= task.time
        if debug:
            print(f"task waiting time: {task.waiting_time}, end time: {task.end_time}")
            print(
                f"max_voyage: {self.max_voyage}, max_total_time: {self.max_total_time}, max_total_voyage: {self.max_total_voyage}"
            )
            print(
                f"UAV {uav.id} updated: location {uav.location}, ammunition {uav.ammunition}, time {uav.time}, voyage {uav.voyage}"
            )
        if allocation:
            log_allocation(uav, task)

    def step(self, action):
        # action: 选择 UAV 索引
        info = {}
        target = self.targets[self.current_target_idx]
        task = self.task
        choose_uav = self.uavs[action]
        reward = calculate_reward(
            choose_uav, task, target, self.max_total_voyage, self.max_total_time
        )
        # 更新 UAV 状态
        self.update_uav_status(choose_uav, task)

        # 推进该 target 内任务
        self.task_step[target.id] += 1
        if self.task_step[target.id] >= len(target.tasks):
            self.current_target_idx += 1
            if self.current_target_idx >= len(self.targets):
                self.done = True
        if self.current_target_idx < len(self.targets):
            target = self.targets[self.current_target_idx]
            self.task = target.tasks[self.task_step[target.id]]

        next_state = None if self.done else self._get_state()
        return next_state, reward, self.done, info
