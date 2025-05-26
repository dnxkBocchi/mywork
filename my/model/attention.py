import torch
import torch.nn as nn
import torch.nn.functional as F
import random
from collections import deque
import numpy as np

class TaskUAVAttention(nn.Module):
    def __init__(self, uav_attr_dim, task_attr_dim, hidden_dim=64):
        """
        任务-无人机注意力模块
        :param uav_attr_dim: 无人机单属性维度（来自_normalize_uav的输出长度）
        :param task_attr_dim: 任务属性维度（来自_normalize_task的输出长度）
        :param hidden_dim: 注意力中间维度
        """
        super().__init__()
        self.uav_attr_dim = uav_attr_dim
        self.task_attr_dim = task_attr_dim
        
        # 定义查询（任务）、键（无人机）、值（无人机）的线性变换
        self.query = nn.Linear(task_attr_dim, hidden_dim)  # 任务→查询向量
        self.key = nn.Linear(uav_attr_dim, hidden_dim)     # 无人机→键向量
        self.value = nn.Linear(uav_attr_dim, hidden_dim)   # 无人机→值向量
        
        # 缩放因子（防止点积过大）
        self.scale = 1.0 / (hidden_dim ** 0.5)

    def forward(self, uav_features, task_feature):
        """
        :param uav_features: (num_uavs, uav_attr_dim) 无人机特征矩阵
        :param task_feature: (task_attr_dim,) 任务特征向量
        :return: 上下文特征 (hidden_dim,)
        """
        # 扩展任务特征维度以便广播计算（1, task_attr_dim）
        task_feature = task_feature.unsqueeze(0)
        
        # 计算查询、键、值向量
        q = self.query(task_feature)  # (1, hidden_dim)
        k = self.key(uav_features)     # (num_uavs, hidden_dim)
        v = self.value(uav_features)   # (num_uavs, hidden_dim)
        
        # 计算注意力分数（1, num_uavs）
        scores = torch.matmul(q, k.T) * self.scale
        
        # 计算注意力权重（softmax归一化）
        attn_weights = F.softmax(scores, dim=-1)  # (1, num_uavs)
        
        # 聚合值向量生成上下文特征（1, hidden_dim）
        context = torch.matmul(attn_weights, v)
        
        return context.squeeze(0)  # (hidden_dim,)


class AttentionDQN(nn.Module):
    def __init__(self, uav_attr_dim, task_attr_dim, action_dim, hidden_dim=64):
        """
        带注意力机制的DQN网络
        :param uav_attr_dim: 无人机单属性维度
        :param task_attr_dim: 任务属性维度
        :param action_dim: 动作空间维度（无人机数量）
        :param hidden_dim: 注意力中间维度
        """
        super().__init__()
        self.attention = TaskUAVAttention(uav_attr_dim, task_attr_dim, hidden_dim)
        
        # 全连接层（上下文特征+任务特征 → 动作值）
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim + task_attr_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim)
        )

    def forward(self, uav_features, task_feature):
        """
        :param uav_features: (num_uavs, uav_attr_dim) 无人机特征矩阵
        :param task_feature: (task_attr_dim,) 任务特征向量
        :return: 各动作的Q值 (action_dim,)
        """
        # 计算注意力上下文
        context = self.attention(uav_features, task_feature)
        
        # 拼接上下文特征和原始任务特征
        combined = torch.cat([context, task_feature], dim=1)
        
        # 计算Q值
        q_values = self.fc(combined)
        
        return q_values


class AttentionReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, uav_features, task_feature, action, reward, next_uav_features, next_task_feature, done):
        # 存储分离的特征
        self.buffer.append((uav_features, task_feature, action, reward, next_uav_features, next_task_feature, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        
        # 高效转换：先转换为NumPy数组，再转换为张量
        uav_features = torch.tensor(np.array([b[0] for b in batch]), dtype=torch.float32)
        task_features = torch.tensor(np.array([b[1] for b in batch]), dtype=torch.float32)
        actions = torch.tensor(np.array([b[2] for b in batch]), dtype=torch.long)
        rewards = torch.tensor(np.array([b[3] for b in batch]), dtype=torch.float32)
        next_uav_features = torch.tensor(np.array([b[4] for b in batch]), dtype=torch.float32)
        next_task_features = torch.tensor(np.array([b[5] for b in batch]), dtype=torch.float32)
        dones = torch.tensor(np.array([b[6] for b in batch]), dtype=torch.float32)

        # uav_features, task_features, actions, rewards, next_uav_features, next_task_features, dones = zip(*batch)
        return uav_features, task_features, actions, rewards, next_uav_features, next_task_features, dones
    
    def __len__(self):
        return len(self.buffer)