import torch
import torch.nn as nn
import torch.nn.functional as F


class Actor(nn.Module):
    def __init__(self, obs_dim, act_dim):
        super().__init__()
        self.fc1 = nn.Linear(obs_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, act_dim)

    def forward(self, obs, mask=None):
        """
        obs: [B, obs_dim]
        mask: [B, act_dim]  (0 or -1e9)
        """
        x = F.relu(self.fc1(obs))
        x = F.relu(self.fc2(x))
        logits = self.fc3(x)

        if mask is not None:
            logits = logits + mask

        probs = F.softmax(logits, dim=-1)
        return probs


class Critic(nn.Module):
    def __init__(self, state_dim, act_dim, n_agents):
        super().__init__()
        self.input_dim = state_dim + n_agents * act_dim

        self.fc1 = nn.Linear(self.input_dim, 512)
        self.fc2 = nn.Linear(512, 512)
        self.fc3 = nn.Linear(512, 1)

    def forward(self, state, actions_probs):
        """
        state: [B, state_dim]
        actions_probs: [B, n_agents * act_dim]
        """
        x = torch.cat([state, actions_probs], dim=-1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        q = self.fc3(x)
        return q
