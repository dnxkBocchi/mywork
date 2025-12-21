import torch
import torch.nn as nn
import torch.nn.functional as F
import random
from collections import deque
import numpy as np


class TaskUAVAttention(nn.Module):
    def __init__(self, uav_attr_dim, task_attr_dim, hidden_dim=64):
        super().__init__()
        self.uav_attr_dim = uav_attr_dim
        self.task_attr_dim = task_attr_dim
        # 查询（任务）、键（无人机）、值（无人机）的线性变换（支持批量）
        self.query = nn.Linear(task_attr_dim, hidden_dim)  # 任务特征 → 查询向量
        self.key = nn.Linear(uav_attr_dim, hidden_dim)  # 无人机特征 → 键向量
        self.value = nn.Linear(uav_attr_dim, hidden_dim)  # 无人机特征 → 值向量
        # 缩放因子（防止点积过大，与批量无关）
        self.scale = 1.0 / (hidden_dim**0.5)

    def forward(self, uav_features, task_feature):
        """
        支持批量和单次输入的注意力计算
        :param uav_features:
            - 单次输入: (num_uavs, uav_attr_dim)
            - 批量输入: (batch_size, num_uavs, uav_attr_dim)
        :param task_feature:
            - 单次输入: (task_attr_dim,)
            - 批量输入: (batch_size, task_attr_dim)
        :return: 上下文特征
            - 单次输入: (hidden_dim,)
            - 批量输入: (batch_size, hidden_dim)
        """
        # 自动识别输入是否为批量（通过维度判断）
        is_batch = (
            len(uav_features.shape) == 3
        )  # 批量输入时uav_features是3维（batch_size, num_uavs, attr_dim）
        if not is_batch:
            # 单次输入时手动添加批量维度（batch_size=1）
            uav_features = uav_features.unsqueeze(0)  # (1, num_uavs, uav_attr_dim)
            task_feature = task_feature.unsqueeze(0)  # (1, task_attr_dim)

        # 计算查询、键、值向量（自动适配批量）
        q = self.query(
            task_feature
        )  # (batch_size, hidden_dim) → 扩展为 (batch_size, 1, hidden_dim)
        q = q.unsqueeze(1)  # 增加“查询数量”维度（每个任务对应1个查询）
        k = self.key(uav_features)  # (batch_size, num_uavs, hidden_dim)
        v = self.value(uav_features)  # (batch_size, num_uavs, hidden_dim)

        # 计算注意力分数（批量矩阵乘法）
        # k需要转置最后两维，使形状变为 (batch_size, hidden_dim, num_uavs)
        k = k.transpose(-2, -1)  # (batch_size, hidden_dim, num_uavs)
        scores = torch.matmul(q, k) * self.scale  # (batch_size, 1, num_uavs)

        # 计算注意力权重（在无人机维度归一化）
        attn_weights = F.softmax(scores, dim=-1)  # (batch_size, 1, num_uavs)

        # 聚合值向量生成上下文特征（批量矩阵乘法）
        context = torch.matmul(attn_weights, v)  # (batch_size, 1, hidden_dim)
        context = context.squeeze(
            1
        )  # 移除冗余的“查询数量”维度 → (batch_size, hidden_dim)

        # 单次输入时移除批量维度
        if not is_batch:
            context = context.squeeze(0)  # (hidden_dim,)

        return context


class AttentionDQN(nn.Module):
    def __init__(self, uav_attr_dim, task_attr_dim, action_dim, hidden_dim=64):
        super().__init__()
        self.attention = TaskUAVAttention(uav_attr_dim, task_attr_dim, hidden_dim)
        # 全连接层（上下文特征 + 任务特征 → 动作Q值）
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim + task_attr_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim),
        )

    def forward(self, uav_features, task_feature):
        """
        支持批量和单次输入的Q值计算
        :param uav_features: 同TaskUAVAttention的输入
        :param task_feature: 同TaskUAVAttention的输入
        :return:
            - 单次输入: (action_dim,)
            - 批量输入: (batch_size, action_dim)
        """
        # 计算注意力上下文（自动适配批量）
        context = self.attention(
            uav_features, task_feature
        )  # (batch_size, hidden_dim) 或 (hidden_dim,)
        # 拼接上下文特征和原始任务特征（自动适配批量）
        if len(context.shape) == 1:  # 单次输入
            combined = torch.cat(
                [context, task_feature], dim=0
            )  # (hidden_dim + task_attr_dim,)
        else:  # 批量输入
            combined = torch.cat(
                [context, task_feature], dim=1
            )  # (batch_size, hidden_dim + task_attr_dim)
        # 计算Q值
        q_values = self.fc(combined)
        return q_values


class AttentionReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(
        self,
        uav_features,
        task_feature,
        action,
        reward,
        next_uav_features,
        next_task_feature,
        done,
    ):
        # 存储原始数据（单次输入或批量输入均可，但建议存储单次数据）
        self.buffer.append(
            (
                uav_features,
                task_feature,
                action,
                reward,
                next_uav_features,
                next_task_feature,
                done,
            )
        )

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        # 转换为张量时自动保留批量维度
        uav_features = torch.tensor(
            np.array([b[0] for b in batch]), dtype=torch.float32
        )  # (batch_size, num_uavs, uav_attr_dim)
        task_features = torch.tensor(
            np.array([b[1] for b in batch]), dtype=torch.float32
        )  # (batch_size, task_attr_dim)
        actions = torch.tensor(
            np.array([b[2] for b in batch]), dtype=torch.long
        )  # (batch_size,)
        rewards = torch.tensor(
            np.array([b[3] for b in batch]), dtype=torch.float32
        )  # (batch_size,)
        next_uav_features = torch.tensor(
            np.array([b[4] for b in batch]), dtype=torch.float32
        )  # (batch_size, num_uavs, uav_attr_dim)
        next_task_features = torch.tensor(
            np.array([b[5] for b in batch]), dtype=torch.float32
        )  # (batch_size, task_attr_dim)
        dones = torch.tensor(
            np.array([b[6] for b in batch]), dtype=torch.float32
        )  # (batch_size,)

        return (
            uav_features,
            task_features,
            actions,
            rewards,
            next_uav_features,
            next_task_features,
            dones,
        )

    def __len__(self):
        return len(self.buffer)


"""
运行 AttentionDQN 优化
"""
# import torch
# # 全局设置（推荐）
# torch.set_printoptions(
#     threshold=float('inf'),  # 禁用省略，显示所有元素
#     edgeitems=5,             # 触发省略时首尾显示5个元素（仅当threshold未设为inf时有效）
#     precision=6,             # 浮点数保留6位小数
#     linewidth=200            # 每行最大宽度200字符（避免换行过多）
# )

# from train.attention_train import train_attention_dqn
# model = train_attention_dqn(env)
