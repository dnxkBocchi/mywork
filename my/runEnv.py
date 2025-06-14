import numpy as np
import matplotlib.pyplot as plt
from calculate import *
from model.greedy import *
import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"  # 关键：ipynb 中绘图运行时避免报错
debug = False  # 是否打印调试信息


# 简化版环境示例
class UAVEnv:
    def __init__(self, uavs, targets, tasks, map_size=1000.0, mode="dqn"):
        self.uavs = uavs
        self.targets = targets
        self.tasks = tasks
        self.task = None  # 当前任务
        self.map_size = map_size
        self.done = False
        self.mode = mode

        self.max_ammo = max(u.ammunition for u in uavs) or 1
        self.max_voyage = max(u.voyage for u in uavs) or 1
        self.max_speed = max(u.speed for u in uavs) or 1
        # 后面奖励函数要用到
        self.max_total_voyage, self.max_total_time = calculate_max_possible_voyage_time(
            self.uavs, self.targets
        )
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

        return self._get_state()

    def _normalize_uav(self, u):
        # 类型归一化 (1-3 -> 0-1)
        type_norm = (u.type - 1) / 2
        loc_x = u.location[0] / self.map_size
        loc_y = u.location[1] / self.map_size
        return [
            type_norm,
            # u.status,
            loc_x,
            loc_y,
            u.strike,
            u.reconnaissance,
            u.assessment,
            u.ammunition / self.max_ammo,
            u.time,
            u.voyage / self.max_voyage,
            # u.speed / self.max_speed,
            # u.value,
        ]

    def _normalize_task(self, task):
        type_norm = (task.type - 1) / 2
        loc_x = task.location[0] / self.map_size
        loc_y = task.location[1] / self.map_size
        return [
            type_norm,
            loc_x,
            loc_y,
            task.strike,
            task.reconnaissance,
            task.assessment,
            task.ammunition / self.max_ammo,
            task.time,
            # task.value,
        ]

    def _get_state(self):
        # 将所有 UAV 归一化后状态 + 当前任务归一化后状态
        uav_states = []
        for u in self.uavs:
            if self.mode == "dqn" or self.mode == "my_dqn":
                uav_states.extend(self._normalize_uav(u))
            if self.mode == "attention":
                uav_states.append(self._normalize_uav(u))
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
        if self.mode == "dqn" or self.mode == "my_dqn":
            return np.array(uav_states + task_state, dtype=np.float32)
        # 返回 UAV 状态和任务状态 Attention 输入格式
        elif self.mode == "attention":
            return np.array(uav_states, dtype=np.float32), np.array(
                task_state, dtype=np.float32
            )

    def update_uav_status(self, uav, task):
        # 更新 UAV 状态
        uav.voyage -= calculate_voyage_distance(uav, task)
        # 任务分配后的完成时间
        task.waiting_time = uav.end_time
        uav.end_time += calculate_voyage_time(uav, task)
        task.end_time = uav.end_time
        task.flag = True  # 标记任务已完成
        if task.type == 3:  # 评估任务
            task.target.total_time = task.end_time

        uav.location = task.location
        uav.ammunition -= task.ammunition
        uav.time -= task.time
        if debug:
            print(f"task waiting time: {task.waiting_time}, end time: {task.end_time}")
        if debug:
            print(
                f"UAV {uav.id} updated: location {uav.location}, ammunition {uav.ammunition}, time {uav.time}, voyage {uav.voyage}"
            )

    def step(self, action, ep=100):
        # action: 选择 UAV 索引
        info = {}
        target = self.targets[self.current_target_idx]
        task = self.task
        choose_uav = self.uavs[action]
        if self.mode == "dqn":
            reward = calculate_reward(
                choose_uav, task, target, self.max_total_voyage, self.max_total_time
            )
        elif self.mode == "my_dqn":
            reward = calculate_reward(
                choose_uav, task, target, self.max_total_voyage, self.max_total_time, ep
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

        # dqn 模块
        if self.mode == "dqn" or self.mode == "my_dqn":
            next_state = None if self.done else self._get_state()
            return next_state, reward, self.done, info
        # attention 模块
        elif self.mode == "attention":
            next_state_uav, next_state_task = (
                (None, None) if self.done else self._get_state()
            )
            return next_state_uav, next_state_task, reward, self.done, info
