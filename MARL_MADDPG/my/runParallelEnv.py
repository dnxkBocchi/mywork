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
    def __init__(
        self, uavs, targets, tasks, K_local, M_global, N_neighbors, map_size=100.0
    ):
        self.uavs = uavs
        # 深拷贝一份“干净”的初始 targets 和 tasks 作为备份
        self.init_targets = copy.deepcopy(targets)
        # 初始化当前使用的 targets
        self.targets = copy.deepcopy(self.init_targets)
        self.tasks = tasks
        self.task = None  # 当前任务
        self.map_size = map_size
        self.comm_radius = 50.0  # 通信半径
        self.done = False
        self.finish_task_count = 0
        self.tasks_num = len(self.tasks)
        self.uavs_num = len(self.uavs)

        # === 动态环境参数 ===
        self.crash_lambda = 0.005  # 风险系数 lambda，可调
        self.add_threshold = 0.3  # 任务压力阈值 theta_add
        self.prob_add = 0.3  # 生成概率 p_add
        self.destroy_flag = False  # 坠毁标志
        # === 关键参数 ===
        self.K_local = K_local  # Actor观测的最近任务数 (Top-K)
        self.M_global = M_global  # Critic观测的全局最大任务数 (用于Padding)
        self.N_neighbors = N_neighbors  # 最大邻居数
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

    def _get_env_dim(self):
        return self.uav_dim, self.task_dim

    def reset(self):
        # 恢复所有 UAV 初始状态
        for uav in self.uavs:
            uav.reset()
        # 从备份中深拷贝，覆盖当前的 self.targets
        self.targets = copy.deepcopy(self.init_targets)
        self.tasks = [t for target in self.targets for t in target.tasks]
        for task in self.tasks:
            task.reset()
        self.done = False
        self.max_voyage = max(u.voyage for u in self.uavs) or 1
        self.max_speed = max(u.speed for u in self.uavs) or 1
        # 后面奖励函数要用到
        self.max_total_voyage, self.max_total_time = calculate_max_possible_voyage_time(
            self.uavs, self.targets
        )
        self.finish_task_count = 0
        self.destroy_flag = False
        self.uav_dim = len(self._normalize_uav(self.uavs[0]))
        self.task_dim = len(self._normalize_task(self.tasks[0]))

        return self.get_obs_all()

    def _normalize_uav(self, u):
        # 类型归一化 (1-3 -> 0-1)
        type_norm = (u.type - 1) / 2
        loc_x = u.location[0] / self.map_size
        loc_y = u.location[1] / self.map_size
        capacity, resource = get_capacity(u)
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
        capacity, resource = get_capacity(task)
        alive_status = 1.0 if task.flag else 0.0  # 新增特征
        return [
            type_norm,
            loc_x,
            loc_y,
            capacity,
            resource,
            # alive_status,
            # task.value,
        ]

    def _get_neighbor_uavs(self, uav):
        """
        获取通信半径内的邻居 UAV (按距离排序)
        """
        neighbors = []
        for other in self.uavs:
            if other.type != uav.type:
                continue
            if other.id == uav.id:
                continue
            dist = calculate_voyage_distance(uav, other)
            if dist <= self.comm_radius:
                neighbors.append((dist, other))
        # 按距离排序
        neighbors.sort(key=lambda x: x[0])
        # 取最近 N_neighbors 个
        return [u for _, u in neighbors[: self.N_neighbors]]

    def _get_valid_tasks(self, uav):
        """
        为某个 UAV 筛选它现在【能看见】且【能执行】的任务
        并按执行代价进行稳定排序
        """
        valid_tasks = []
        if not uav.alive:
            return []
        for task in self.tasks:
            if task.flag:
                continue
            if task.type != uav.type:
                continue
            dist = calculate_voyage_distance(uav, task)
            if dist > self.comm_radius:
                continue
            if check_dependency(task):
                valid_tasks.append(task)
        # 稳定排序
        valid_tasks.sort(key=lambda t: (calculate_voyage_distance(uav, t)))
        return valid_tasks

    def get_agent_obs(self, uav):
        """
        Actor 输入构造:
        [Self_State, Neighbor_1, ..., Neighbor_N, Task_1, ..., Task_K]
        """
        obs_vec = []
        obs_vec.extend(self._normalize_uav(uav))
        current_neighbors = self._get_neighbor_uavs(uav)
        for n_uav in current_neighbors:
            obs_vec.extend(self._normalize_uav(n_uav))
        # 填充 (Padding): 如果邻居不够 N_neighbors 个，补 0
        missing_neighbors = self.N_neighbors - len(current_neighbors)
        if missing_neighbors > 0:
            obs_vec.extend([0.0] * self.uav_dim * missing_neighbors)

        if debug:
            print(
                f"-- {uav.id} neighbors: {[n.id for n in current_neighbors]}, missing: {missing_neighbors}"
            )
            # print(f"neighbors obs_vec : {obs_vec}")

        candidates = self._get_valid_tasks(uav)
        top_k_tasks = candidates[: self.K_local]
        # *** 关键: 更新缓存，供 step_parallel 使用 ***
        self.candidate_cache[uav.idx] = top_k_tasks
        # 初始化 Mask: 长度 = K_local
        action_mask = np.full(self.K_local, -1e9, dtype=np.float32)
        # 填充任务特征 并激活Mask
        for i in range(self.K_local):
            if i < len(top_k_tasks):
                # 有真实任务
                task = top_k_tasks[i]
                obs_vec.extend(self._normalize_task(task))
                action_mask[i] = 1.0  # 激活该位置动作
            else:
                # 没任务 (Padding)
                obs_vec.extend([0.0] * self.task_dim)

        if debug:
            print(
                f"{uav.id} tasks: {[t.id for t in top_k_tasks]}, missing: {self.K_local - len(top_k_tasks)}"
            )
            # print(f"tasks obs_vec : {obs_vec[self.uav_dim * (1 + self.N_neighbors):]}")
            print(f"action_mask: {action_mask}")
        return np.array(obs_vec, dtype=np.float32), action_mask

    def get_global_state(self):
        """
        生成 Critic 的全局状态输入
        [UAV_1...UAV_N, Task_1...Task_M] (固定大小)
        """
        g_state = []
        # 所有 UAV
        for u in self.uavs:
            g_state.extend(self._normalize_uav(u))
        # 所有 Task (Padding or Truncate to M_global)
        current_task_count = 0
        for t in self.tasks:
            if not t.flag:  # 只关注未完成的？或者全部？一般 Critic 需要看全部以了解进度
                if current_task_count < self.M_global:
                    g_state.extend(self._normalize_task(t))
                    current_task_count += 1
        # 补 0
        missing = self.M_global - current_task_count
        if missing > 0:
            g_state.extend([0.0] * self.task_dim * missing)
        return np.array(g_state, dtype=np.float32)

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
        rewards = np.zeros(len(self.uavs), dtype=np.float32)
        cnt_success = 0
        cnt_fitness = 0
        # Phase 1: collect intentions
        intentions = []
        for i, uav in enumerate(self.uavs):
            if not uav.alive:
                intentions.append((uav, None))
                continue
            act = actions_idx[i]
            candidates = self.candidate_cache.get(uav.idx, [])
            task = candidates[act] if act < len(candidates) else None
            intentions.append((uav, task))
        # Phase 2: resolve conflicts
        task_to_uavs = {}
        for uav, task in intentions:
            if task is None:
                continue
            task_to_uavs.setdefault(task, []).append(uav)
        for task, uavs in task_to_uavs.items():
            if task.flag:
                continue
            # 计算所有 UAV 对该 task 的 base reward
            reward_candidates = []
            for uav in uavs:
                r = calculate_reward(
                    uav,
                    task,
                    task.target,
                    self.max_total_voyage,
                    self.max_total_time,
                )
                reward_candidates.append((uav, r))
            best_r = max(r for _, r in reward_candidates)
            # 选 winner（近似最优 + 随机打破平局）
            eps = 1e-3
            # best_uavs = [uav for uav, r in reward_candidates if abs(r - best_r) < eps]
            best_uavs = [uav for uav, r in reward_candidates]
            winner = np.random.choice(best_uavs)

            # -------- Difference Reward --------
            for uav, r in reward_candidates:
                if uav == winner:
                    # winner: 系统获得 best_r
                    rewards[uav.idx] += best_r
                    self.update_uav_status(uav, task)
                    if r > 0:
                        cnt_success += 1
                    cnt_fitness += calculate_fitness_r(task, uav)
                else:
                    # loser: 没有你系统也能拿到 best_r
                    # 你的存在反而引入竞争
                    rewards[uav.idx] -= best_r - r

        # Phase 3: wait penalty
        for uav, task in intentions:
            if task is None and uav.alive:
                rewards[uav.idx] -= 0.05

        # Dynamic events
        # for uav in self.uavs:
        #     self._check_uav_crash_event(uav)
        # self._check_new_task_event()
        self.done = all(t.flag for t in self.tasks)

        next_obs, next_g_state, next_masks = self.get_obs_all()
        info = {"success_count": cnt_success, "fitness_count": cnt_fitness}

        if debug:
            print(f"Step finished: Rewards: {rewards}, finished tasks: {cnt_success}")

        return (next_obs, next_g_state, next_masks), rewards, self.done, info

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
        self.finish_task_count += 1
        if debug:
            # print(f"task waiting time: {task.waiting_time}, end time: {task.end_time}")
            # print(
            #     f"max_voyage: {self.max_voyage}, max_total_time: {self.max_total_time}, max_total_voyage: {self.max_total_voyage}"
            # )
            # print(
            #     f"UAV {uav.id} updated: location {uav.location}, ammunition {uav.ammunition}, time {uav.time}, voyage {uav.voyage}"
            # )
            print(
                f"uav {uav.id} and task {task.id} updated, Total finished tasks: {self.finish_task_count}"
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
        if np.random.random() < risk_prob and not self.destroy_flag:
            uav.alive = False
            self.destroy_flag = True  # 本步只允许一个 UAV 坠毁
            if debug_dynamic:
                print(
                    f"!!! EVENT step {self.finish_task_count}: UAV {uav.id} CRASHED after voyage {used_voyage:.2f} !!!"
                )

    def _check_new_task_event(self):
        """
        事件1:新增任务判定
        基于任务压力 Lambda(t)
        """
        # 简化版 Lambda 计算： 剩余任务数 / 存活无人机总能力
        # 或者直接用：当前处理的 target 索引 / 总 target 数 (表示进度)
        progress = self.finish_task_count / len(self.targets)
        # 如果进度过半，且随机触发
        if progress > 0.5 and np.random.random() < self.prob_add:
            # 限制一下最大数量，防止无限循环
            if len(self.targets) < (self.M_global / 3):
                new_id = f"NEW_TGT_{len(self.targets)}"
                new_target = create_random_target(new_id, self.map_size)
                self.targets.append(new_target)
                self.tasks.extend(new_target.tasks)
                if debug_dynamic:
                    print(
                        f"!!! EVENT step {self.finish_task_count}: New Target {new_id} Detected !!!"
                    )
