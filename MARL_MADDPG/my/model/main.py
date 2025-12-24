import numpy as np
import torch
import matplotlib.pyplot as plt
from model.maddpg import MADDPG
from runEnv import UAVEnv
from env import load_different_scale_csv
from calculate import *

device = "cuda" if torch.cuda.is_available() else "cpu"
# 设置不同的规模比例


def train_maddpg(scale, episodes=500):
    # 单个 UAV 观测维度（你 normalize 后的）
    n_agents = scale
    max_neighbors = 3
    # 邻居 + 自己 + 任务信息
    obs_dim = 9 * (max_neighbors + 1) + 5
    total_obs_dim = obs_dim * scale
    num_tasks = scale * 3

    # 定义一个探索率 epsilon
    epsilon = 0.9
    epsilon_decay = 0.995
    min_epsilon = 0.05

    # ---------- 加载环境 ----------
    uav_csv = "data/test/uav.csv"
    task_csv = "data/test/task.csv"
    uavs, tasks, targets = load_different_scale_csv(uav_csv, task_csv, size=scale)
    env = UAVEnv(uavs, targets, tasks, max_neighbors)

    maddpg = MADDPG(
        n_agents=n_agents, obs_dim=obs_dim, total_obs_dim=total_obs_dim, device=device
    )

    # 用于存储每集的总 reward
    rewards_per_episode = []
    rewards_per10_episode = []

    # ---------- 训练 ----------
    for ep in range(episodes):
        state = env.reset()
        done = False

        # 记录每次实验的数据，判断优化程度
        total_reward = 0
        total_success = 0
        total_distance = 0
        total_time = 0
        total_fitness = 0

        while not done:
            # 构造 obs_all
            obs_all = env.get_obs_all()

            # 获取原始动作 (保持 (10, 1) 的形状)
            raw_actions = maddpg.select_action(obs_all)
            #  创建一个扁平化的副本用于 Mask 计算
            flat_actions = raw_actions.flatten()

            # Mask 逻辑
            mask = []
            current_task = env.task
            valid_exists = False
            for uav in env.uavs:
                if check_constraints(uav, current_task):
                    mask.append(0.0)
                    valid_exists = True
                else:
                    mask.append(-1e9)

            if not valid_exists:
                mask = [0.0] * len(env.uavs)

            # 将 Mask 加到扁平化的动作上
            masked_flat_actions = flat_actions + np.array(mask)

            # --- 选择动作逻辑 ---
            if np.random.random() > epsilon:
                chosen = np.argmax(masked_flat_actions)
            else:
                valid_indices = [i for i, m in enumerate(mask) if m > -1e8]
                if valid_indices:
                    chosen = np.random.choice(valid_indices)
                else:
                    chosen = np.random.randint(0, n_agents)

            fitness = calculate_fitness_r(env.task, env.uavs[chosen])
            next_state, reward, done, _ = env.step(chosen)

            total_fitness += fitness
            total_reward += reward
            if reward > 0:
                total_success += 1
            if next_state is None:
                break
            next_obs_all = env.get_obs_all()

            # 必须存入 (10, 1) 形状的数据，否则 update 时会报错
            maddpg.buffer.push(obs_all, raw_actions, reward, next_obs_all, done)
            maddpg.update()

        epsilon = max(min_epsilon, epsilon * epsilon_decay)
        # 记录每集的平均 reward
        rewards_per_episode.append(total_reward / num_tasks)
        if ep % 10 == 0:
            mean = np.mean(rewards_per_episode)
            rewards_per10_episode.append(mean)
            rewards_per_episode = []
        total_reward /= num_tasks  # 平均每个任务的奖励
        total_success /= num_tasks  # 平均每个任务的成功率
        total_fitness /= num_tasks
        total_distance = calculate_all_voyage_distance(env.uavs)
        total_time = calculate_all_voyage_time(env.targets)

        print(
            f"Episode {ep} | Total Reward: {total_reward:.2f} | Total Fitness: {total_fitness:.2f} \
| Total Distance: {total_distance:.2f} | Total Time: {total_time:.2f} | Total Success : {total_success:.2f}"
        )

        # 每 100 轮绘制一次 reward 曲线
        if (ep + 1) % 50 == 0:
            epsd = int((ep + 1) / 10)
            plt.figure(figsize=(8, 4))
            plt.plot(range(1, epsd + 1), rewards_per10_episode, marker="o")
            plt.xlabel("Episode")
            plt.ylabel("Total Reward")
            plt.ylim(0)
            plt.title(f"Total Reward up to Episode {epsd}")
            plt.grid(True)
            plt.show()
