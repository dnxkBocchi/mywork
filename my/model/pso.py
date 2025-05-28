import numpy as np
import math
import random
import sys
import os

# 添加项目根目录到 sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from my.calculate import calculate_fitness, calculate_max_possible_voyage

debug = False  # 是否打印调试信息


class Particle:
    def __init__(self, num_tasks, num_uavs):
        """
        初始化粒子
        :param num_tasks: 任务数量
        :param num_uavs: 无人机数量
        """
        self.num_tasks = num_tasks
        self.num_uavs = num_uavs

        # 位置向量：每个任务分配给哪个无人机
        self.position = np.random.randint(0, num_uavs, size=num_tasks)
        # 速度向量：位置更新的变化量
        self.velocity = np.random.uniform(-1, 1, size=num_tasks)
        # 个体最优位置和适应度
        self.pbest_position = self.position.copy()
        self.pbest_fitness = float("-inf")
        # 当前适应度
        self.fitness = float("-inf")

    def update_velocity(self, gbest_position, w=0.5, c1=1.5, c2=1.5):
        """
        更新粒子速度
        :param gbest_position: 全局最优位置
        :param w: 惯性权重
        :param c1: 个体学习因子
        :param c2: 社会学习因子
        """
        r1, r2 = random.random(), random.random()
        # 认知部分：向个体最优学习
        cognitive = c1 * r1 * (self.pbest_position - self.position)
        # 社会部分：向全局最优学习
        social = c2 * r2 * (gbest_position - self.position)
        # 更新速度
        self.velocity = w * self.velocity + cognitive + social
        # 速度限制（可选）
        self.velocity = np.clip(self.velocity, -2, 2)
        if debug:
            print(
                f"r1: {r1}, r2: {r2}, cognitive: {cognitive}, social: {social}, velocity: {self.velocity}"
            )

    def update_position(self):
        """更新粒子位置"""
        if debug:
            print(f"Before update - Position: {self.position}")
        # 添加速度到位置
        self.position = self.position + self.velocity
        # 对位置进行离散化处理（转为整数）
        self.position = np.round(self.position).astype(int)
        # 确保位置在有效范围内（0到num_uavs-1）
        self.position = np.clip(self.position, 0, self.num_uavs - 1)
        if debug:
            print(f"After update - Position: {self.position}")


class PSO:
    def __init__(self, env, num_particles=30, max_iter=100, w=0.7, c1=1.5, c2=1.5):
        """
        初始化粒子群优化算法
        :param env: 无人机任务分配环境
        :param num_particles: 粒子数量
        :param max_iter: 最大迭代次数
        :param w: 惯性权重
        :param c1: 个体学习因子
        :param c2: 社会学习因子
        """
        self.env = env
        self.num_particles = num_particles
        self.max_iter = max_iter
        self.w = w
        self.c1 = c1
        self.c2 = c2

        # 获取任务和无人机数量
        self.num_tasks = sum(len(target.tasks) for target in env.targets)
        self.num_uavs = len(env.uavs)

        # 初始化粒子群
        self.particles = [
            Particle(self.num_tasks, self.num_uavs) for _ in range(num_particles)
        ]

        # 全局最优位置和适应度
        self.gbest_position = None
        self.gbest_fitness = float("-inf")

        # 计算理论最大航程用于归一化
        self.max_possible_voyage = calculate_max_possible_voyage(env.uavs, env.targets)

        # 初始化全局最优
        self._initialize_gbest()

    def _initialize_gbest(self):
        """初始化全局最优解"""
        for particle in self.particles:
            # 评估粒子适应度
            fitness = self._evaluate_particle(particle)

            # 更新个体最优
            if fitness > particle.pbest_fitness:
                particle.pbest_fitness = fitness
                particle.pbest_position = particle.position.copy()

            # 更新全局最优
            if fitness > self.gbest_fitness:
                self.gbest_fitness = fitness
                self.gbest_position = particle.position.copy()

    def _evaluate_particle(self, particle):
        """
        评估粒子的适应度（任务分配方案的优劣）
        :param particle: 粒子
        :return: 适应度值
        """
        # 重置环境
        state = self.env.reset()
        done = False
        step = 0

        # 按粒子的分配方案执行任务
        while not done and step < self.num_tasks:
            # 获取当前任务应该分配的无人机索引
            uav_index = particle.position[step]
            # 执行动作
            next_state, reward, done, _ = self.env.step(uav_index)
            step += 1

        # 计算任务适配度（所有任务的平均适配度）
        avg_reward = sum(self.env.reward_history) / self.num_tasks

        # 计算归一化的总航程（越小越好）
        normalized_voyage = min(self.env.total_voyage / self.max_possible_voyage, 1.0)

        # 综合评分（可调整权重）
        alpha = 1.0  # 任务适配度权重
        beta = 1  # 任务适配度权重
        gamma = 0.2  # 航程权重

        # 注意：航程是越小越好，所以用1减去归一化航程
        fitness = alpha * avg_reward
        return fitness

    def optimize(self):
        """执行粒子群优化"""
        for iteration in range(self.max_iter):
            # 对每个粒子进行更新和评估
            for particle in self.particles:
                # 更新速度和位置
                particle.update_velocity(self.gbest_position, self.w, self.c1, self.c2)
                particle.update_position()

                # 评估新位置的适应度
                fitness = self._evaluate_particle(particle)

                # 更新个体最优
                if fitness > particle.pbest_fitness:
                    particle.pbest_fitness = fitness
                    particle.pbest_position = particle.position.copy()

                # 更新全局最优
                if fitness > self.gbest_fitness:
                    self.gbest_fitness = fitness
                    self.gbest_position = particle.position.copy()

            # 打印进度
            print(
                f"Iteration {iteration+1}/{self.max_iter} - Best Fitness: {self.gbest_fitness:.4f}, position: {self.gbest_position.tolist()}"
            )

        # 返回最优解
        return self.gbest_position, self.gbest_fitness
