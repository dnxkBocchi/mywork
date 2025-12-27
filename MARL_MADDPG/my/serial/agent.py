import torch
import torch.nn as nn
import torch.nn.functional as F


class Actor(nn.Module):
    def __init__(self, obs_dim, act_dim=1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, act_dim),
            nn.Sigmoid()  # action ∈ [0,1]
        )

    def forward(self, obs):
        return self.net(obs)


class Critic(nn.Module):
    """
    Centralized Critic:
    Q(s_1,...,s_N, a_1,...,a_N)
    """
    def __init__(self, total_obs_dim, total_act_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(total_obs_dim + total_act_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )

    def forward(self, obs_all, act_all):
        x = torch.cat([obs_all, act_all], dim=-1)
        return self.net(x)
