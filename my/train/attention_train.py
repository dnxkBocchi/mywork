import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt

from model.attention import AttentionDQN, AttentionReplayBuffer
from calculate import *


def train_attention_dqn(
    env,
    episodes=200,
    batch_size=64,
    gamma=0.99,
    lr=1e-3,
    eps_start=1.0,
    eps_end=0.01,
    eps_decay=0.995,
):

    # 初始化参数
    num_tasks = sum(len(target.tasks) for target in env.targets)
    eps = eps_start
    uav_attr_dim, task_attr_dim = env.reset()
    uav_attr_dim = len(uav_attr_dim[0])  # UAV特征维度
    task_attr_dim = len(task_attr_dim)  # 任务特征维度
    action_dim = len(env.uavs)

    # 创建模型
    dqn = AttentionDQN(uav_attr_dim, task_attr_dim, action_dim)
    target_dqn = AttentionDQN(uav_attr_dim, task_attr_dim, action_dim)
    target_dqn.load_state_dict(dqn.state_dict())
    optimizer = optim.Adam(dqn.parameters(), lr=lr)
    buffer = AttentionReplayBuffer(10000)

    # 用于存储每集的总 reward
    rewards_per_episode = []
    loss_per_episode = []

    # 训练循环示例
    for episode in range(episodes):
        uav_features, task_feature = env.reset()

        # 记录每次实验的数据，判断优化程度
        total_reward = 0
        total_success = 0
        total_distance = 0
        total_time = 0

        done = False
        while not done:
            loss_episode = []
            # 转换为Tensor
            uav_feat_tensor = torch.tensor(uav_features, dtype=torch.float32)
            task_feat_tensor = torch.tensor(task_feature, dtype=torch.float32)

            # 选择动作（ε-greedy）
            if random.random() < eps:
                action = random.randint(0, action_dim - 1)
            else:
                with torch.no_grad():
                    q_values = dqn(uav_feat_tensor, task_feat_tensor)
                    action = q_values.argmax().item()

            # 执行动作
            next_uav_features, next_task_feature, reward, done, _ = env.step(action)
            total_reward += reward
            if reward > 0:
                total_success += 1

            # 存储经验
            buffer.push(
                uav_features,
                task_feature,
                action,
                reward,
                (
                    next_uav_features
                    if next_uav_features is not None
                    else np.zeros_like(uav_features)
                ),
                (
                    next_task_feature
                    if next_task_feature is not None
                    else np.zeros_like(task_feature)
                ),
                done,
            )

            # 经验回放更新
            if len(buffer) > batch_size:
                # 采样批量数据
                (
                    uav_feats,
                    task_feats,
                    acts,
                    rews,
                    next_uav_feats,
                    next_task_feats,
                    dones,
                ) = buffer.sample(batch_size)
                # 计算目标Q值
                with torch.no_grad():
                    next_q_values = target_dqn(next_uav_feats, next_task_feats)
                    max_next_q = next_q_values.max(dim=1)[0]
                    target_q = rews + (1 - dones) * gamma * max_next_q
                # 计算当前Q值
                current_q = (
                    dqn(uav_feats, task_feats).gather(1, acts.unsqueeze(1)).squeeze()
                )
                # 计算损失
                loss = F.mse_loss(current_q, target_q)
                loss_episode.append(loss.item())
                # 反向传播
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        # 每集结束，记录 total_reward 和 loss
        rewards_per_episode.append(total_reward / num_tasks)
        loss_per_episode.append(np.mean(loss_episode))
        # 更新目标网络
        eps = max(eps_end, eps * eps_decay)
        if episode % 10 == 0:
            target_dqn.load_state_dict(dqn.state_dict())
        total_reward /= num_tasks  # 平均每个任务的奖励
        total_success /= num_tasks  # 平均每个任务的成功率
        total_distance = calculate_all_voyage_distance(env.uavs)
        total_time = calculate_all_voyage_time(env.targets)
        print(
            f"Episode {episode} | Total Reward: {total_reward:.2f} | Total Distance: {total_distance:.2f} | \
Total Time: {total_time:.2f} | Total Success : {total_success:.2f} | Epsilon: {eps:.3f}"
        )

        # 每 50 轮绘制一次 reward 曲线
        if (episode + 1) % 50 == 0:
            plt.figure(figsize=(8, 4))
            plt.plot(range(1, episode + 2), rewards_per_episode, marker="o")
            plt.xlabel("Episode")
            plt.ylabel("Total Reward")
            plt.title(f"Total Reward up to Episode {episode + 1}")
            plt.grid(True)
            plt.show()

            plt.figure(figsize=(8, 4))
            plt.plot(range(1, episode + 2), loss_per_episode, marker="o")
            plt.xlabel("Episode")
            plt.ylabel("Total Loss")
            plt.title(f"Total Loss up to Episode {episode + 1}")
            plt.grid(True)
            plt.show()

    return dqn
