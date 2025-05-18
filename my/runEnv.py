import numpy as np
from calculate import *


# 简化版环境示例
class UAVEnv:
    def __init__(self, uavs, targets, map_size=1000.0, debug=False):
        self.uavs = uavs
        self.targets = targets
        self.map_size = map_size
        self.debug = debug
        self.max_ammo = max(u.ammunition for u in uavs) or 1
        self.max_time = max(u.time for u in uavs) or 1
        self.max_voyage = max(u.voyage for u in uavs) or 1
        self.max_speed = max(u.speed for u in uavs) or 1
        self.reset()

    def reset(self):
        # 重置环境状态
        self.current_target_idx = 0
        self.done = False
        # 为每个 Target 初始化任务进度
        self.task_step = {t.id: 0 for t in self.targets}
        return self._get_state()

    def _normalize_uav(self, u):
        # 类型归一化 (1-3 -> 0-1)
        type_norm = (u.type - 1) / 2
        loc_x = u.location[0] / self.map_size
        loc_y = u.location[1] / self.map_size
        return [
            # type_norm,
            # u.status,
            loc_x,
            loc_y,
            u.strike,
            u.reconnaissance,
            u.assessment,
            # u.ammunition / self.max_ammo,
            # u.time / self.max_time,
            # u.voyage / self.max_voyage,
            # u.speed / self.max_speed,
            u.value,
        ]

    def _normalize_task(self, task):
        type_norm = (task.type - 1) / 2
        loc_x = task.location[0] / self.map_size
        loc_y = task.location[1] / self.map_size
        return [
            # type_norm,
            loc_x,
            loc_y,
            task.strike,
            task.reconnaissance,
            task.assessment,
            # task.ammunition / 10.0,
            # task.time / 600.0,
            task.value,
        ]

    def _get_state(self):
        # 将所有 UAV 归一化后状态 + 当前任务归一化后状态
        uav_states = []
        for u in self.uavs:
            uav_states.extend(self._normalize_uav(u))
        target = self.targets[self.current_target_idx]
        task = target.tasks[self.task_step[target.id]]
        task_state = self._normalize_task(task)

        if self.debug:
            print(f"uav_states: {uav_states}, uav_states_len: {len(uav_states)}")
            print(f"task_state: {task_state}, task_state_len: {len(task_state)}")

        return np.array(uav_states + task_state, dtype=np.float32)

    def update_uav_status(self, uav, task):
        # 更新 UAV 状态
        uav.status = 0  # 标记为忙碌
        uav.location = task.location
        uav.ammunition -= task.ammunition
        uav.time -= task.time
        uav.voyage -= calculate_voyage_distance(uav, task)

    def step(self, action):
        # action: 选择 UAV 索引
        info = {}
        target = self.targets[self.current_target_idx]
        task = target.tasks[self.task_step[target.id]]
        reward = calculate_reward(self.uavs[action], task, target)

        if self.debug:
            print(
                f"UAV {self.uavs[action].id} assigned to task {task.id} at target {target.id}."
            )

        # 更新 UAV 状态
        self.update_uav_status(self.uavs[action], task)

        # 推进该 target 内任务
        self.task_step[target.id] += 1
        if self.task_step[target.id] >= len(target.tasks):
            self.current_target_idx += 1
            if self.current_target_idx >= len(self.targets):
                self.done = True

        next_state = None if self.done else self._get_state()
        return next_state, reward, self.done, info
