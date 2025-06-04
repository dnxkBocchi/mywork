import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import os
from datetime import datetime

from model.dqn import DQN, ReplayBuffer
from calculate import *
from model.greedy import *


def train_dqn(
    env,
    episodes=800,
    batch_size=64,
    gamma=0.99,
    lr=1e-3,
    eps_start=1.0,
    eps_end=0.01,
    eps_decay=0.995,
    save_dir="./results",  # 新增：数据和模型保存目录
    model_name="dqn_model.pth",  # 新增：模型文件名
):
    # 新增：创建保存目录（如果不存在）
    os.makedirs(save_dir, exist_ok=True)
    # 新增：初始化最优指标追踪变量
    best_total_reward = -float("inf")
    # 新增：创建数据日志文件（带时间戳避免覆盖）
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(save_dir, f"training_log_{timestamp}.txt")

    num_tasks = sum(len(target.tasks) for target in env.targets)
    state_dim = len(env.reset())
    action_dim = len(env.uavs)
    policy_net = DQN(state_dim, action_dim)
    target_net = DQN(state_dim, action_dim)
    target_net.load_state_dict(policy_net.state_dict())
    optimizer = optim.Adam(policy_net.parameters(), lr=lr)
    buffer = ReplayBuffer(10000)
    eps = eps_start

    # 用于存储每集的总 reward
    rewards_per_episode = []

    for ep in range(episodes):
        state = env.reset()
        # 记录每次实验的数据，判断优化程度
        total_reward = 0
        total_success = 0
        total_distance = 0
        total_time = 0

        done = False
        while not done:
            if random.random() < eps:
                # action = random.randrange(action_dim)
                # my greedy dqn
                actions = []
                actions.append(select_uav_by_matching(env.task, env.uavs))
                actions.append(select_uav_by_voyage(env.task, env.uavs))
                actions.append(select_uav_by_time(env.task, env.uavs))
                action = random.choice(actions)
            else:
                with torch.no_grad():
                    q_vals = policy_net(torch.tensor(state).unsqueeze(0))
                    action = q_vals.argmax().item()
            # my greedy dqn 加了个 ep 用来动态调整 reward
            next_state, reward, done, _ = env.step(action, ep)
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

            if len(buffer) >= batch_size:
                s, a, r, s2, d = buffer.sample(batch_size)
                s = torch.tensor(s, dtype=torch.float32)
                a = torch.tensor(a)
                r = torch.tensor(r, dtype=torch.float32)
                s2 = torch.tensor(s2, dtype=torch.float32)
                d = torch.tensor(d, dtype=torch.float32)

                q_pred = policy_net(s).gather(1, a.unsqueeze(1)).squeeze()
                with torch.no_grad():
                    q_next = target_net(s2).max(1)[0]
                q_target = r + gamma * q_next * (1 - d)

                loss = nn.functional.mse_loss(q_pred, q_target)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        # 每集结束，记录 total_r
        rewards_per_episode.append(total_reward / num_tasks)
        eps = max(eps_end, eps * eps_decay)
        if ep % 10 == 0:
            target_net.load_state_dict(policy_net.state_dict())
        total_reward /= num_tasks  # 平均每个任务的奖励
        total_success /= num_tasks  # 平均每个任务的成功率
        total_distance = calculate_all_voyage_distance(env.uavs)
        total_time = calculate_all_voyage_time(env.targets)

        # 新增：更新最优指标
        if total_reward > best_total_reward:
            best_total_reward = total_reward
            # 新增：保存最优模型
            torch.save(policy_net.state_dict(), os.path.join(save_dir, model_name))
            # 新增：记录每集数据到文件
            with open(log_file, "a") as f:
                f.write(
                    f"{ep},{total_reward:.4f},{total_success:.4f},{total_distance:.2f},{total_time:.2f},{eps:.4f}\n"
                )

        print(
            f"Episode {ep} | Total Reward: {total_reward:.2f} | Total Distance: {total_distance:.2f} | \
Total Time: {total_time:.2f} | Total Success : {total_success:.2f} | Epsilon: {eps:.3f}"
        )

        # 每 50 轮绘制一次 reward 曲线
        if (ep + 1) % 50 == 0:
            plt.figure(figsize=(8, 4))
            plt.plot(range(1, ep + 2), rewards_per_episode, marker="o")
            plt.xlabel("Episode")
            plt.ylabel("Total Reward")
            plt.title(f"Total Reward up to Episode {ep+1}")
            plt.grid(True)
            plt.show()

    return policy_net
