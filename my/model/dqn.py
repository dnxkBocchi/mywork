import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import os
from datetime import datetime
from collections import deque

from calculate import *
from env import *
from model.greedy import *
from runEnv import UAVEnv


# DQN 网络结构
class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim),
        )

    def forward(self, x):
        return self.net(x)


# 经验回放
class PrioritizedReplayBuffer:
    def __init__(self, capacity, alpha=0.6):
        self.capacity = capacity
        self.alpha = alpha
        self.buffer = deque(maxlen=capacity)
        self.priorities = deque(maxlen=capacity)
        self.pos = 0
        self.max_priority = 1.0  # 初始优先级

    def push(self, state, action, reward, next_state, done):
        # 新经验优先级设为当前最大，保证能被采样到
        self.buffer.append((state, action, reward, next_state, done))
        self.priorities.append(self.max_priority)

    def sample(self, batch_size, beta=0.4):
        if len(self.buffer) == self.capacity:
            priorities = np.array(self.priorities)
        else:
            priorities = np.array(self.priorities)[: len(self.buffer)]

        probs = priorities**self.alpha
        probs /= probs.sum()

        indices = np.random.choice(len(self.buffer), batch_size, p=probs)
        samples = [self.buffer[idx] for idx in indices]

        # 计算重要性采样权重
        total = len(self.buffer)
        weights = (total * probs[indices]) ** (-beta)
        weights /= weights.max()  # 归一化到 [0,1]

        s, a, r, s2, d = zip(*samples)
        return (
            np.stack(s),
            a,
            r,
            np.stack(s2),
            d,
            indices,
            np.array(weights, dtype=np.float32),
        )

    def update_priorities(self, indices, td_errors, eps=1e-6):
        # 更新优先级
        for idx, td in zip(indices, td_errors):
            priority = abs(td) + eps
            self.priorities[idx] = priority
            self.max_priority = max(self.max_priority, priority)

    def __len__(self):
        return len(self.buffer)


def train_dqn(
    scale,
    episodes=2000,
    batch_size=64,
    gamma=0.99,
    lr=1e-3,
    eps_start=1.0,
    eps_end=0.01,
    eps_decay=0.995,
    save_dir="./results",  # 新增：数据和模型保存目录
):
    # 新增：创建保存目录（如果不存在）
    os.makedirs(save_dir, exist_ok=True)
    # 设置数据目录
    uav_csv = "data/test/uav.csv"
    task_csv = "data/test/task.csv"
    total_uavs, total_tasks, total_targets = load_different_scale_csv(
        uav_csv, task_csv, scale
    )

    # 新增：初始化最优指标追踪变量
    best_total_reward = float("inf")
    state_dim = 5 + 9 * scale  # 状态维度：1个任务特征 + scale个uav特征
    action_dim = scale  # 动作维度应对应总无人机数量
    num_tasks = scale * 3

    policy_net = DQN(state_dim, action_dim)
    target_net = DQN(state_dim, action_dim)
    target_net.load_state_dict(policy_net.state_dict())
    optimizer = optim.Adam(policy_net.parameters(), lr=lr)
    eps = eps_start
    alpha = 0.6  
    beta_start = 0.4  
    beta_frames = 100000  
    buffer = PrioritizedReplayBuffer(10000, alpha=alpha)
    frame_idx = 1  

    # 用于存储每集的总 reward
    rewards_per_episode = []
    rewards_per10_episode = []
    voyage = []
    time = []

    for ep in range(episodes):

        # 关键修改：每个episode随机选择10个无人机和10个target
        # uavs = random.sample(range(total_uavs), scale)
        uavs = total_uavs[:scale]
        # targets = random.sample(total_targets, scale)
        targets = total_targets[:scale]
        tasks = [task for target in targets for task in target.tasks]
        env = UAVEnv(uavs, targets, tasks)

        state = env.reset()
        # 记录每次实验的数据，判断优化程度
        total_reward = 0
        total_success = 0
        total_distance = 0
        total_time = 0
        total_fitness = 0

        done = False
        while not done:
            if random.random() < eps and ep < 1000:
                # my greedy dqn
                actions = []
                # actions.append(select_uav_by_matching(env.task, env.uavs))
                actions.append(select_uav_by_voyage(env.task, env.uavs))
                actions.append(select_uav_by_time(env.task, env.uavs))
                action = random.choice(actions)
                action = random.choice(range(len(uavs)))
            else:
                with torch.no_grad():
                    q_vals = policy_net(torch.tensor(state).unsqueeze(0))
                    action = q_vals.argmax().item()
            fitness = calculate_fitness_r(env.task, env.uavs[action])
            next_state, reward, done, _ = env.step(action)
            total_fitness += fitness
            if reward > 0:
                total_success += 1
            buffer.push(
                state,
                action,
                reward,
                next_state if next_state is not None else np.zeros_like(state),
                done,
            )

            state = next_state
            total_reward += reward

            beta = beta_start + frame_idx * (1.0 - beta_start) / beta_frames
            beta = min(1.0, beta)
            frame_idx += 1

            if len(buffer) >= batch_size:
                s, a, r, s2, d, indices, weights = buffer.sample(batch_size, beta=beta)

                s = torch.tensor(s, dtype=torch.float32)
                a = torch.tensor(a)
                r = torch.tensor(r, dtype=torch.float32)
                s2 = torch.tensor(s2, dtype=torch.float32)
                d = torch.tensor(d, dtype=torch.float32)
                weights = torch.tensor(weights, dtype=torch.float32)

                q_pred = policy_net(s).gather(1, a.unsqueeze(1)).squeeze()
                with torch.no_grad():
                    best_actions = policy_net(s2).argmax(1)
                    q_next = (
                        target_net(s2).gather(1, best_actions.unsqueeze(1)).squeeze()
                    )
                q_target = r + gamma * q_next * (1 - d)
                td_errors = q_target - q_pred
                loss = (weights * td_errors.pow(2)).mean()

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                buffer.update_priorities(indices, td_errors.detach().cpu().numpy())

        # 每集结束，记录 total_r
        rewards_per_episode.append(total_reward / num_tasks)
        eps = max(eps_end, eps * eps_decay)
        if ep % 10 == 0:
            target_net.load_state_dict(policy_net.state_dict())
            mean = np.mean(rewards_per_episode)
            # if np.mean(rewards_per_episode) < 0.4:
            #     mean += 0.3
            rewards_per10_episode.append(mean)
            rewards_per_episode = []
        total_reward /= num_tasks  # 平均每个任务的奖励
        total_success /= num_tasks  # 平均每个任务的成功率
        total_fitness /= num_tasks
        total_distance = calculate_all_voyage_distance(env.uavs)
        total_time = calculate_all_voyage_time(env.targets)

        # 新增：更新最优指标
        if (
            total_distance + total_time < best_total_reward
            and total_success > 0.8
            and ep > 500
        ):
            best_total_reward = total_distance + total_time
            # 新增：保存最优模型
            model_name = f"dqn_model_{scale}x.pth"
            torch.save(policy_net.state_dict(), os.path.join(save_dir, model_name))

            voyage = []
            time = []
            for uav in env.uavs:
                distance = uav._init_voyage - uav.voyage
                distance += calculate_back_voyage(uav)
                voyage.append(distance)
            for target in env.targets:
                time.append(target.total_time)
            log_all_voyage_time(env.uavs, env.targets)
            log_total_method(
                total_reward, total_fitness, total_distance, total_time, total_success
            )

        #         print(
        #             f"Episode {ep} | Total Reward: {total_reward:.2f} | Total Fitness: {total_fitness:.2f} \
        # | Total Distance: {total_distance:.2f} | Total Time: {total_time:.2f} \
        # | Total Success : {total_success:.2f} | Epsilon: {eps:.3f}"
        #         )

        # 每 100 轮绘制一次 reward 曲线
        if (ep + 1) % 100 == 0:
            epsd = int((ep + 1) / 10)
            plt.figure(figsize=(8, 4))
            plt.plot(range(1, epsd + 1), rewards_per10_episode, marker="o")
            plt.xlabel("Episode")
            plt.ylabel("Total Reward")
            plt.ylim(0)
            plt.title(f"Total Reward up to Episode {epsd}")
            plt.grid(True)
            plt.show()

    with open("plt/rewards_per10_episode.txt", "a") as f:
        for reward in rewards_per10_episode:
            f.write(str(reward) + " ")
    return policy_net


def test_dqn(
    env,
    scale,  # 保存的模型路径
    test_episodes=1,  # 测试轮数
):
    # 1. 初始化环境和模型
    state_dim = len(env.reset())  # 获取状态维度（与训练时一致）
    action_dim = len(env.uavs)  # 获取动作维度（与训练时一致）
    # 2. 构建与训练时相同结构的DQN网络
    policy_net = DQN(state_dim, action_dim)
    # 3. 加载保存的模型权重
    model_path = f"./results/dqn_model_{scale}x.pth"
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"模型文件不存在: {model_path}")
    policy_net.load_state_dict(torch.load(model_path))
    policy_net.eval()  # 切换到评估模式（禁用训练相关层）

    # 4. 运行测试
    for ep in range(test_episodes):
        state = env.reset()  # 重置环境
        total_reward = 0
        total_success = 0
        total_fitness = 0
        num_tasks = sum(len(target.tasks) for target in env.targets)  # 任务总数

        done = False
        while not done:
            # 测试时不使用探索，直接用模型选最优动作
            with torch.no_grad():  # 禁用梯度计算，加速推理
                q_vals = policy_net(
                    torch.tensor(state, dtype=torch.float32).unsqueeze(0)
                )
                action = q_vals.argmax().item()  # 选择Q值最大的动作
            # 执行动作并获取反馈
            fitness = calculate_fitness_r(env.task, env.uavs[action])
            next_state, reward, done, _ = env.step(action)
            total_fitness += fitness
            state = next_state
            total_reward += reward
            if reward > 0:  # 假设正奖励表示任务成功
                total_success += 1

        # 计算每轮测试指标
        total_success = total_success / num_tasks
        total_reward = total_reward / num_tasks
        total_fitness = total_fitness / num_tasks
        total_distance = calculate_all_voyage_distance(
            env.uavs
        )  # 复用训练时的距离计算函数
        total_time = calculate_all_voyage_time(env.targets)  # 复用训练时的时间计算函数

        log_all_voyage_time(env.uavs, env.targets)
        log_total_method(
            total_reward, total_fitness, total_distance, total_time, total_success
        )
