import numpy as np
import random
from calculate import *
from env import Uav, Task, Target
from runEnv import UAVEnv


class GeneticAlgorithmScheduler:
    def __init__(
        self,
        env,
        pop_size=50,
        crossover_rate=0.8,
        mutation_rate=0.1,
        max_generations=100,
    ):
        """
        初始化遗传算法调度器
        :param uavs: 无人机集合
        :param targets: 目标集合
        :param tasks: 任务集合
        :param pop_size: 种群大小
        :param crossover_rate: 交叉概率
        :param mutation_rate: 变异概率
        :param max_generations: 最大迭代次数
        """
        self.env = env
        self.uavs = env.uavs
        self.targets = env.targets
        self.tasks = env.tasks
        self.task_order = [
            task for target in self.targets for task in target.tasks
        ]  # 按目标-任务顺序展开的任务列表
        self.n_tasks = len(self.task_order)
        self.n_uavs = len(self.uavs)
        for idx, uav in enumerate(self.uavs):
            uav.idx = idx

        # 算法参数
        self.pop_size = pop_size
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.max_generations = max_generations

        # 初始化种群
        self.population = self._initialize_population()

    def _initialize_population(self) -> list[np.ndarray]:
        """初始化种群：每个个体是长度为n_tasks的数组，元素为分配的无人机索引"""
        population = []
        for _ in range(self.pop_size):
            # 初始解：随机分配（确保类型约束）
            individual = np.zeros(self.n_tasks, dtype=int)
            for i, task in enumerate(self.task_order):
                # 仅选择类型匹配的无人机
                valid_uavs = [uav.idx for uav in self.uavs if uav.type == task.type]
                individual[i] = random.choice(valid_uavs)
            population.append(individual)
        return population

    def _fitness_function(self, individual: np.ndarray, flag) -> float:
        """
        适应度函数：计算个体的综合适应度（任务完成率 + 总奖励）
        :param individual: 个体（任务-无人机分配方案）
        :return: 适应度值（越高越好）
        """
        # 重置环境状态
        self.env.reset()
        total_reward = 0.0
        total_success = 0.0
        total_fitness = 0.0

        for task_idx, uav_idx in enumerate(individual):
            current_task = self.task_order[task_idx]
            selected_uav = self.uavs[uav_idx]
            # 执行任务分配（模拟环境step）
            _, reward, done, _ = self.env.step(uav_idx)
            if reward > 0:
                total_success += 1
            total_reward += reward
            total_fitness += calculate_fitness_r(current_task, selected_uav)

            if done:
                break  # 所有任务完成
        if flag:
            total_reward /= self.n_tasks  # 平均每个任务的奖励
            total_success /= self.n_tasks  # 平均每个任务的成功率
            total_fitness /= self.n_tasks  # 平均适配度
            total_distance = calculate_all_voyage_distance(self.uavs)
            total_time = calculate_all_voyage_time(self.targets)
            print(
                f"Total Reward: {total_reward:.2f} | Total Fitness: {total_fitness:.2f} \
| Total Distance: {total_distance:.2f} | Total Time: {total_time:.2f} \
| Total Success : {total_success:.2f}"
            )
        return total_reward

    def _selection(self) -> list[np.ndarray]:
        """轮盘赌选择：根据适应度选择下一代个体"""
        fitness_values = [self._fitness_function(ind, False) for ind in self.population]
        total_fitness = sum(fitness_values)
        if total_fitness == 0:  # 避免除零错误
            return random.choices(self.population, k=self.pop_size)
        probabilities = [f / total_fitness for f in fitness_values]
        return random.choices(self.population, weights=probabilities, k=self.pop_size)

    def _crossover(
        self, parent1: np.ndarray, parent2: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """单点交叉操作（保持任务顺序约束）"""
        if random.random() > self.crossover_rate:
            return parent1.copy(), parent2.copy()
        # 随机选择交叉点（在任务序列中间）
        crossover_point = random.randint(1, self.n_tasks - 1)
        child1 = np.concatenate([parent1[:crossover_point], parent2[crossover_point:]])
        child2 = np.concatenate([parent2[:crossover_point], parent1[crossover_point:]])
        return child1, child2

    def _mutation(self, individual: np.ndarray) -> np.ndarray:
        """变异操作（随机调整单个任务的无人机分配）"""
        mutated = individual.copy()
        for i in range(self.n_tasks):
            if random.random() < self.mutation_rate:
                task = self.task_order[i]
                valid_uavs = [uav.idx for uav in self.uavs if uav.type == task.type]
                mutated[i] = random.choice(valid_uavs)
        return mutated

    def run(self) -> tuple[np.ndarray, float]:
        """运行遗传算法主循环"""
        best_fitness = -np.inf
        best_individual = None

        for generation in range(self.max_generations):
            # 选择操作
            selected_pop = self._selection()
            # 交叉操作
            new_pop = []
            for i in range(0, self.pop_size, 2):
                parent1 = selected_pop[i]
                parent2 = (
                    selected_pop[i + 1] if i + 1 < self.pop_size else selected_pop[0]
                )
                child1, child2 = self._crossover(parent1, parent2)
                new_pop.extend([child1, child2])
            # 变异操作
            new_pop = [self._mutation(ind) for ind in new_pop[: self.pop_size]]
            # 评估并更新最优解
            current_best_idx = np.argmax(
                [self._fitness_function(ind, False) for ind in new_pop]
            )
            current_best_fitness = self._fitness_function(
                new_pop[current_best_idx], True
            )
            if current_best_fitness > best_fitness:
                best_fitness = current_best_fitness
                best_individual = new_pop[current_best_idx]

            print(
                f"Generation {generation+1}/{self.max_generations} | Best Fitness: {best_fitness:.4f}"
            )

        return best_individual, best_fitness
