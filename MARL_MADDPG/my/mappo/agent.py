import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical


class MAPPOActor(nn.Module):
    """
    Actor网络: 输入局部观测，输出动作概率分布
    """

    def __init__(self, obs_dim, act_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 256),
            nn.Tanh(),  # PPO通常用Tanh或ReLU
            nn.Linear(256, 256),
            nn.Tanh(),
            nn.Linear(256, act_dim),
        )

    def forward(self, obs, masks=None):
        """
        obs: [Batch, obs_dim]
        masks: [Batch, act_dim] (1.0 for valid, -1e9 for invalid)
        """
        logits = self.net(obs)

        if masks is not None:
            # 加上 mask，使无效动作的 logit 变为负无穷
            # 注意：你的环境中 mask 无效时是 -1e9，有效是 0 或 1，请确保和这里逻辑一致
            # 建议环境输出: 0(无效), 1(有效)。如果已经是-1e9格式直接加即可。
            # 这里假设环境传进来的是加法Mask (0有效, -1e9无效)
            logits = logits + masks

        probs = F.softmax(logits, dim=-1)
        return probs

    def evaluate(self, obs, action, masks=None):
        """
        用于训练阶段计算 log_prob 和 entropy
        """
        logits = self.net(obs)
        if masks is not None:
            logits = logits + masks

        probs = F.softmax(logits, dim=-1)
        dist = Categorical(probs)

        action_log_probs = dist.log_prob(action)
        dist_entropy = dist.entropy()

        return action_log_probs, dist_entropy


class MAPPOCritic(nn.Module):
    """
    Critic网络: 输入全局状态，输出状态价值 V(s)
    注意：这里不需要输入 Action，只需要 State
    """

    def __init__(self, state_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.Tanh(),
            nn.Linear(256, 256),
            nn.Tanh(),
            nn.Linear(256, 1),  # 输出标量 Value
        )

    def forward(self, state):
        """
        state: [Batch, state_dim]
        """
        value = self.net(state)
        return value
