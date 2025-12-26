import torch
import torch.nn as nn
import torch.nn.functional as F


class Actor(nn.Module):
    def __init__(self, obs_dim, act_dim):
        super().__init__()
        # act_dim 现在等于任务总数 (num_tasks)
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, act_dim),
            # 注意：这里不加 Softmax/Sigmoid，直接输出 logits
            # Gumbel-Softmax 会在 MADDPG 内部调用
        )

    def forward(self, obs):
        return self.net(obs)


class Critic(nn.Module):
    """
    Centralized Critic:
    输入所有 Agent 的状态和动作
    act_dim 也是 One-hot 后的维度
    """

    def __init__(self, total_obs_dim, total_act_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(total_obs_dim + total_act_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
        )

    def forward(self, obs_all, act_all):
        # act_all 是所有 agent 动作的拼接 (Batch, n_agents * num_tasks)
        x = torch.cat([obs_all, act_all], dim=-1)
        return self.net(x)
