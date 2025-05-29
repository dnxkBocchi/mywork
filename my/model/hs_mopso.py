import numpy as np
import random
from typing import List, Dict, Tuple
from env import Uav, Task, Target
from calculate import *

debug = False  # 是否开启调试模式


class Particle:
    def __init__(self, env):
        self.env = env
        self.uavs = env.uavs
        self.tasks = env.tasks
        self.targets = env.targets
        self.n_tasks = len(self.tasks)
        self.n_uavs = len(self.uavs)
        self.target_ids = []  # 目标编号（如"T1"对应侦察/打击/评估任务）
        self.uav_ids = []  # 无人机ID（如"u1O"表示侦察型无人机1）
        self.resources = []  # 资源消耗（打击任务消耗弹药，侦察/评估消耗时间）
        self.range = []  # 航程增加量
        self.reward = []  # 奖励
        self.fitness = None  # 适应度
        self.pbest = None  # 个体最优粒子
        self.archives = []  # 帕累托档案集（用于优势个体选择）
        self.max_total_voyage, self.max_total_time = calculate_max_possible_voyage_time(
            self.uavs, self.targets
        )

    def initialize(self, targets: List[Target]):
        """基于约束的动态优选初始化"""
        task_sequence = []
        for target in targets:
            # 按任务类型排序：侦察→打击→评估
            task_sequence.extend(target.tasks)
        assess_uavs = [u for u in self.uavs if u.type == 3]  # 评估型无人机（类型3）
        recon_uavs = [u for u in self.uavs if u.type == 2]  # 侦察型无人机（类型2）
        attack_uavs = [u for u in self.uavs if u.type == 1]  # 战斗型无人机（类型1）
        available_uavs = {
            3: assess_uavs.copy(),
            2: recon_uavs.copy(),
            1: attack_uavs.copy(),
        }

        for task in task_sequence:
            uav_type = task.type  # 任务类型对应无人机类型（侦察/评估→2，打击→1）
            valid_uavs = available_uavs[uav_type].copy()
            assigned = False

            while valid_uavs and not assigned:
                uav = random.choice(valid_uavs)
                round_trip = calculate_voyage_distance(uav, task)

                # 检查约束
                if not check_constraints(uav, task):
                    valid_uavs.remove(uav)
                    continue
                # 分配任务
                self.target_ids.append(task.target.id)  # 提取目标编号（如"T1"）
                self.uav_ids.append(uav.id)
                self.resources.append(task.ammunition if uav_type == 1 else task.time)
                self.range.append(round_trip)
                self.reward.append(
                    calculate_reward(
                        uav,
                        task,
                        task.target,
                        self.max_total_voyage,
                        self.max_total_time,
                    )
                )
                # 更新无人机状态
                self.env.update_uav_status(uav, task)
                assigned = True

            if not assigned:
                uav = random.choice(self.uavs)  # 如果没有可用无人机，随机分配
                round_trip = calculate_voyage_distance(uav, task)
                # 分配任务
                self.target_ids.append(task.target.id)  # 提取目标编号（如"T1"）
                self.uav_ids.append(uav.id)
                self.resources.append(task.ammunition if uav_type == 1 else task.time)
                self.range.append(round_trip)
                self.reward.append(
                    calculate_reward(
                        uav,
                        task,
                        task.target,
                        self.max_total_voyage,
                        self.max_total_time,
                    )
                )
                # 更新无人机状态
                self.env.update_uav_status(uav, task)

        # 恢复无人机初始状态（避免影响其他粒子）
        self.env.reset()
        self.update_fitness()

    def update_fitness(self):
        """计算三目标函数值（f1:总奖励）"""
        f1 = sum(self.reward) / self.n_tasks  # 总奖励
        self.fitness = f1

    def copy(self):
        """深拷贝粒子状态"""
        new_particle = Particle(self.env)
        new_particle.target_ids = self.target_ids.copy()
        new_particle.uav_ids = self.uav_ids.copy()
        new_particle.resources = self.resources.copy()
        new_particle.range = self.range.copy()
        new_particle.reward = self.reward.copy()
        new_particle.fitness = self.fitness
        return new_particle

    def print(self):
        """打印粒子状态"""
        print(f"Particle Fitness: {self.fitness}")
        print(f"Target IDs: {self.target_ids}")
        print(f"UAV IDs: {self.uav_ids}")
        print(f"Resources: {self.resources}")
        print(f"Range: {self.range}")
        print(f"Reward: {self.reward}")


# 基于约束的粒子动态优选初始化
class HS_MOPSO:
    def __init__(
        self,
        env,
        max_iter: int = 80,
        pop_size: int = 120,
        c: int = 10,  # 初始化优选次数
        C1: float = 0.45,
        C2: float = 0.45,
        m: float = 0.8,  # 变异概率
    ):
        self.env = env
        self.uavs = env.uavs
        self.tasks = env.tasks
        self.targets = env.targets
        self.max_iter = max_iter
        self.pop_size = pop_size
        self.c = c
        self.C1 = C1
        self.C2 = C2
        self.m = m
        self.population = self._initialize_population()
        self.pareto_set = []

    def _initialize_population(self) -> List[Particle]:
        """生成优选初始种群"""
        population = []
        for _ in range(self.c * self.pop_size):
            particle = Particle(self.env)
            particle.initialize(self.targets)
            population.append(particle)

        # 优选：选择前pop_size个优势解（基于欧氏距离到原点）
        population.sort(key=lambda x: np.linalg.norm(np.array(x.fitness)))
        return population[: self.pop_size]

    def _update_pareto_set(self, particle: Particle):
        """更新帕累托解集"""
        dominated = False
        to_remove = []
        for p in self.pareto_set:
            if particle.fitness < p.fitness:
                dominated = True
                break
            if p.fitness < particle.fitness:
                to_remove.append(p)

        for p in to_remove:
            self.pareto_set.remove(p)
        if not dominated:
            self.pareto_set.append(particle.copy())

    # 基于支配关系的优势个体选择策略
    def _select_pbest(self, particle: Particle) -> Particle:
        """个体最优选择（迭代前期用非劣解策略，后期用帕累托策略）"""
        if not particle.pbest:
            particle.pbest = particle.copy()
            return particle.pbest
        current_fitness = particle.fitness
        pbest_fitness = particle.pbest.fitness
        # 非劣解策略（迭代前50%使用）
        if self.iter_count < self.max_iter * 0.5:
            if pbest_fitness < current_fitness:
                particle.pbest = particle.copy()
        else:  # 帕累托策略（迭代后50%使用）
            particle.pbest.archives.append(particle.copy())
            # 维护档案集的非支配前沿
            self._update_pareto_set(particle)
            if particle.fitness < particle.pbest.fitness:  # 此处需根据多目标比较
                particle.pbest = particle.copy()
        return particle.pbest

    def _select_gbest(self) -> Particle:
        """全局最优选择（基于帕累托前沿随机选取）"""
        if not self.pareto_set:
            return random.choice(self.population)
        return random.choice(self.pareto_set)

    def _task_based_crossover(
        self, p: Particle, pbest: Particle, gbest: Particle
    ) -> Particle:
        """基于任务模块的交叉学习"""
        new_p = p.copy()
        n_tasks = len(p.target_ids)
        mask = np.random.choice(
            [0, 1], size=n_tasks, p=[1 - self.C1, self.C1]
        )  # 以C1概率向pbest学习
        for i in range(n_tasks):
            if mask[i]:
                new_p.target_ids[i] = pbest.target_ids[i]
                new_p.uav_ids[i] = pbest.uav_ids[i]
            else:
                new_p.target_ids[i] = gbest.target_ids[i]
                new_p.uav_ids[i] = gbest.uav_ids[i]
        return new_p

    def _task_based_mutation(self, p: Particle) -> Particle:
        """基于任务模块的变异"""
        new_p = p.copy()
        if np.random.rand() < self.m:
            task_idx = np.random.randint(0, len(p.target_ids))
            task = self.tasks[task_idx]
            uav_type = task.type
            valid_uavs = [u.id for u in self.uavs if u.type == uav_type]
            new_uav_id = random.choice(valid_uavs)
            new_p.uav_ids[task_idx] = new_uav_id
        return new_p

    # 基于任务的小模块粒子更新及修正策略
    def _constraint_correction(self, p: Particle) -> Particle:
        """基于约束的粒子修正（时序约束、资源约束）"""
        # 1. 修正任务时序（侦察→打击→评估）
        target_tasks = {}
        task_sequence = []
        for target in self.targets:
            # 按任务类型排序：侦察→打击→评估
            task_sequence.extend(target.tasks)
        for i, t_id in enumerate(p.target_ids):
            task_type = task_sequence[i].type
            if t_id not in target_tasks:
                target_tasks[t_id] = {"MO": None, "MA": None, "ME": None}
            if task_type == 2:
                target_tasks[t_id]["MO"] = i
            elif task_type == 1:
                target_tasks[t_id]["MA"] = i
            elif task_type == 3:
                target_tasks[t_id]["ME"] = i

        if debug:
            print("Target Tasks:", target_tasks)

        for t_id, tasks in target_tasks.items():
            if tasks["MA"] is not None and tasks["MO"] is None:
                # 打击任务在侦察前，交换位置
                ma_idx = tasks["MA"]
                mo_idx = np.random.choice(
                    [
                        i
                        for i, tt in enumerate(p.target_ids)
                        if tt == t_id and self.tasks[i].type == 2
                    ]
                )
                p.target_ids[ma_idx], p.target_ids[mo_idx] = (
                    p.target_ids[mo_idx],
                    p.target_ids[ma_idx],
                )
                p.uav_ids[ma_idx], p.uav_ids[mo_idx] = (
                    p.uav_ids[mo_idx],
                    p.uav_ids[ma_idx],
                )

        # 2. 重新计算资源和时间
        p.range = []
        for i in range(len(p.target_ids)):
            uav = next(u for u in self.uavs if u.id == p.uav_ids[i])
            task = self.tasks[i]
            distance = calculate_voyage_distance(uav, task)
            p.range.append(distance)  # 往返航程
        return p

    # 主优化循环
    def run(self):
        self.iter_count = 0
        self.pareto_set = []

        for self.iter_count in range(self.max_iter):
            # 1. 更新个体最优和全局最优
            for particle in self.population:
                particle.pbest = self._select_pbest(particle)
                gbest = self._select_gbest()
                # 2. 小模块更新：交叉+变异
                new_p = self._task_based_crossover(particle, particle.pbest, gbest)
                new_p = self._task_based_mutation(new_p)
                if debug:
                    new_p.print()
                new_p = self._constraint_correction(new_p)
                new_p.update_fitness()
                if debug:
                    new_p.print()
                # 3. 保留优势解
                if particle.fitness < new_p.fitness:
                    particle = new_p
            # 4. 更新帕累托解集
            for particle in self.population:
                self._update_pareto_set(particle)
            print("pareto set:", self.pareto_set)

            # 输出迭代信息
            particle = None
            for p in self.pareto_set:
                if particle is None or p.fitness > particle.fitness:
                    particle = p
            reward = sum(particle.reward) / len(particle.reward)
            distance = sum(particle.range)
            total_time = calculate_all_voyage_time(self.targets)
            success_rate = 0
            for pr in p.reward:
                if pr > 0:
                    success_rate += 1
            success_rate /= len(p.reward)
            print(
                f"Iter {self.iter_count+1}, Average Reward: {reward:.2f}, Total Distance: {distance:.2f}, Total Time: {total_time:.2f}, Success Rate: {success_rate:.2f}"
            )
