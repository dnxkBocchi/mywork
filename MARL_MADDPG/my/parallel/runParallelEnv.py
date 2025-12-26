import numpy as np
import matplotlib.pyplot as plt
from calculate import *
from env import create_random_target
import os
import copy

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"  # 关键：ipynb 中绘图运行时避免报错
# 打印信息
debug = False
debug_dynamic = False
allocation = False


# 简化版环境示例
class DynamicUAVEnv:
    def __init__(self, uavs, targets, tasks, max_neighbors, map_size=100.0):
        self.uavs = uavs
        # 深拷贝一份“干净”的初始 targets 和 tasks 作为备份
        self.init_targets = copy.deepcopy(targets)
        # 初始化当前使用的 targets
        self.targets = copy.deepcopy(self.init_targets)
        self.tasks = tasks
        self.task = None  # 当前任务
        self.map_size = map_size
        self.comm_radius = 30.0  # 通信半径
        self.max_neighbors = max_neighbors  # 最大邻居数
        self.done = False

        # === 动态环境参数 ===
        self.crash_lambda = 0.005  # 风险系数 lambda，可调
        self.add_threshold = 0.3  # 任务压力阈值 theta_add
        self.prob_add = 0.3  # 生成概率 p_add
        # === 关键参数 ===
        self.K_local = 3  # Actor观测的最近任务数 (Top-K)
        self.M_global = 12  # Critic观测的全局最大任务数 (用于Padding)
        # === 缓存 ===
        # 存储每一步每个UAV观测到的候选任务列表，用于将Action Index映射回真实Task
        # 结构: {uav_idx: [task_obj_1, task_obj_2, ...]}
        self.candidate_cache = {}

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
        # 从备份中深拷贝，覆盖当前的 self.targets
        self.targets = copy.deepcopy(self.init_targets)
        self.tasks = [t for target in self.targets for t in target.tasks]
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
        alive_status = 1.0 if u.alive else 0.0  # 新增特征
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
            alive_status,
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

    def _get_valid_tasks(self, uav):
        """
        为某个 UAV 筛选它现在【能看见】且【能执行】的任务
        条件：1. 距离 < 通信半径
             2. 类型匹配 (侦察机看侦察任务)
             3. 任务未完成
             4. 任务的前置依赖已完成 (环境侧的规则约束)
        """
        valid_tasks = []
        if not uav.alive:
            return []

        for task in self.tasks:
            if task.flag:
                continue  # 已完成
            if task.type != uav.type:
                continue  # 类型不对
            # 距离筛选
            dist = np.hypot(
                uav.location[0] - task.location[0], uav.location[1] - task.location[1]
            )
            if dist > self.comm_radius:
                continue
            # === 依赖链检查 (关键) ===
            target = task.target
            target_tasks = target.tasks  # [侦察, 打击, 评估]
            # 如果是侦察(0)，无前置，可执行
            is_ready = False
            if task.type == 1:  # 侦察
                is_ready = True
            elif task.type == 2:  # 打击
                # 依赖侦察(0)完成
                if target_tasks[0].flag:
                    is_ready = True
            elif task.type == 3:  # 评估
                # 依赖打击(1)完成
                if target_tasks[1].flag:
                    is_ready = True
            if is_ready:
                valid_tasks.append(task)

        return valid_tasks

    def get_agent_obs(self, uav):
        """
        Actor 输入构造:
        [Self_State, Neighbor_1, ..., Neighbor_M, Task_1, ..., Task_K]
        """
        obs_vec = []

        obs_vec.extend(self._normalize_uav(uav))
        # 记录单个 UAV 特征长度，用于后续 Padding
        len_uav_feat = len(self._normalize_uav(uav))
        current_neighbors = self._get_neighbor_uavs(uav)
        for n_uav in current_neighbors:
            obs_vec.extend(self._normalize_uav(n_uav))
        # 填充 (Padding): 如果邻居不够 max_neighbors 个，补 0
        missing_neighbors = self.max_neighbors - len(current_neighbors)
        if missing_neighbors > 0:
            obs_vec.extend([0.0] * len_uav_feat * missing_neighbors)

        candidates = self._get_valid_tasks(uav)
        top_k_tasks = candidates[: self.K_local]

        # *** 关键: 更新缓存，供 step_parallel 使用 ***
        self.candidate_cache[uav.idx] = top_k_tasks
        # 初始化 Mask: 长度 = K_local + 1 (最后一位是 Wait)
        action_mask = np.full(self.K_local + 1, -1e9, dtype=np.float32)
        # 填充任务特征 并 激活Mask
        len_task_feat = 5  # 假设 _normalize_task 返回长度为 5
        for i in range(self.K_local):
            if i < len(top_k_tasks):
                # 有真实任务
                task = top_k_tasks[i]
                obs_vec.extend(self._normalize_task(task))
                action_mask[i] = 1.0  # 激活该位置动作
            else:
                # 没任务 (Padding)
                obs_vec.extend([0.0] * len_task_feat)
                # Mask 保持 -inf，禁止智能体选这个空的占位符
        # Part 4: 待机动作 (Wait Action)
        # 总是允许待机 (Index = K_local)
        action_mask[self.K_local] = 1.0

        return np.array(obs_vec, dtype=np.float32), action_mask

    def get_obs_all(self):
        """
        返回:
        1. agent_obs_list: [N, Obs_Dim]
        2. global_state: [State_Dim]
        3. masks: [N, Action_Dim]
        """
        obs_list = []
        masks_list = []
        for u in self.uavs:
            o, m = self.get_agent_obs(u)
            obs_list.append(o)
            masks_list.append(m)

        global_s = self.get_global_state()
        return np.array(obs_list), global_s, np.array(masks_list)

    def step_parallel(self, actions_idx):
        """
        actions_idx: list or array of shape (N_UAV,), 内容为 0 ~ K
        """
        rewards = np.zeros(len(self.uavs), dtype=np.float32)
        # 1. 解码动作: 哪些 UAV 想去执行哪个 Task
        # 暂存申请: task_id -> [uav_obj_list]
        task_applications = {}
        uav_action_map = {}  # 记录每个UAV实际选了啥，方便算个体惩罚

        for i, uav in enumerate(self.uavs):
            if not uav.alive:
                continue

            act = actions_idx[i]
            candidates = self.candidate_cache.get(uav.idx, [])

            if act < len(candidates):
                # 选择了具体的任务
                selected_task = candidates[act]

                # 双重检查：是否被别人抢了？(在Conflict resolve阶段做)
                # 此时任务肯定未完成且合规 (因为是从 get_agent_obs 生成的缓存里取的)

                if selected_task not in task_applications:
                    task_applications[selected_task] = []
                task_applications[selected_task].append(uav)
                uav_action_map[uav.idx] = "work"
            else:
                # 选择了 No-Op (Index = K) 或 选择了 Padding 的空位
                uav_action_map[uav.idx] = "wait"
                # 待机惩罚 (可选，防止它偷懒)
                rewards[i] -= 0.1

        # 2. 冲突消解 (多个 UAV 选同一个 Task)
        # 规则：距离最近的获胜
        cnt_success = 0
        for task, applicants in task_applications.items():
            applicants.sort(
                key=lambda u: np.hypot(
                    u.location[0] - task.location[0], u.location[1] - task.location[1]
                )
            )
            winner = applicants[0]

            # --- 执行任务逻辑 ---
            # 只有 Winner 获得奖励和状态更新
            # 计算消耗
            dist = np.hypot(
                winner.location[0] - task.location[0],
                winner.location[1] - task.location[1],
            )
            time_cost = dist / winner.speed + (
                task.time if hasattr(task, "time") else 10
            )  # 飞行+执行

            # 更新 Winner 物理状态
            winner.location = task.location
            winner.voyage -= dist
            winner.time -= time_cost
            winner.task_nums += 1

            # 标记任务完成
            task.flag = True

            # 给予 Winner 奖励
            # r_base = 10.0
            # r_cost = - (dist / self.max_voyage)
            # rewards[winner.idx] += (r_base + r_cost)
            rewards[winner.idx] += 5.0  # 简化奖励
            cnt_success += 1

            # 对 Losers 的处理
            for loser in applicants[1:]:
                # 竞争失败惩罚
                rewards[loser.idx] -= 0.5

        # 3. 全局协作奖励 (Global Reward)
        # 所有存活 UAV 共享一个基于本步完成任务数的奖励，促进合作
        global_reward = cnt_success * 2.0
        for u in self.uavs:
            if u.alive:
                rewards[u.idx] += global_reward

        # 4. 动态事件检查
        for u in self.uavs:
            self._check_uav_crash(u)
        self._check_new_task()

        # 5. Done 判定
        # 所有任务完成 或 所有 UAV 坠毁/无资源
        all_tasks_done = all(t.flag for t in self.tasks)
        all_uav_dead = all(not u.alive for u in self.uavs)
        self.done = all_tasks_done or all_uav_dead

        # 6. 获取新观测
        next_obs, next_g_state, next_masks = self.get_obs_all()

        info = {"success_count": cnt_success}
        return (next_obs, next_g_state, next_masks), rewards, self.done, info

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

    def _check_uav_crash_event(self, uav):
        """
        事件2: 无人机坠毁判定
        P_fail = 1 - exp(-lambda * Voyage_used)
        """
        if not uav.alive:
            return
        # 计算已用航程
        used_voyage = uav._init_voyage - uav.voyage
        # 为了防止每步都判死，这里应该判定“即时风险”或者在执行完一个任务后判定
        risk_prob = 1 - np.exp(
            -self.crash_lambda * used_voyage / 100.0
        )  # 除以100归一化一下避免必死
        if np.random.random() < risk_prob:
            uav.alive = False
            if debug_dynamic:
                print(
                    f"!!! EVENT step {self.current_target_idx}: UAV {uav.id} CRASHED after voyage {used_voyage:.2f} !!!"
                )

    def _check_new_task_event(self):
        """
        事件1:新增任务判定
        基于任务压力 Lambda(t)
        """
        # 简化版 Lambda 计算： 剩余任务数 / 存活无人机总能力
        # 或者直接用：当前处理的 target 索引 / 总 target 数 (表示进度)
        progress = self.current_target_idx / len(self.targets)
        # 如果进度过半，且随机触发
        if progress > 0.5 and np.random.random() < self.prob_add:
            # 限制一下最大数量，防止无限循环
            if len(self.targets) < 12:
                new_id = f"NEW_TGT_{len(self.targets)}"
                new_target = create_random_target(new_id, self.map_size)

                self.targets.append(new_target)
                self.tasks.extend(new_target.tasks)
                self.task_step[new_target.id] = 0
                if debug_dynamic:
                    print(
                        f"!!! EVENT step {self.current_target_idx}: New Target {new_id} Detected !!!"
                    )
