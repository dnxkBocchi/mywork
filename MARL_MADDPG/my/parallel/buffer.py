import numpy as np
import torch


class ReplayBuffer:
    def __init__(self, max_size, n_agents, obs_dim, state_dim, act_dim, device):
        self.max_size = max_size
        self.device = device
        self.ptr = 0
        self.size = 0

        self.obs = np.zeros((max_size, n_agents, obs_dim), dtype=np.float32)
        self.next_obs = np.zeros((max_size, n_agents, obs_dim), dtype=np.float32)

        self.states = np.zeros((max_size, state_dim), dtype=np.float32)
        self.next_states = np.zeros((max_size, state_dim), dtype=np.float32)

        self.actions = np.zeros((max_size, n_agents, act_dim), dtype=np.float32)
        self.rewards = np.zeros((max_size, n_agents), dtype=np.float32)
        self.dones = np.zeros((max_size, n_agents), dtype=np.float32)

        self.masks = np.zeros((max_size, n_agents, act_dim), dtype=np.float32)
        self.next_masks = np.zeros((max_size, n_agents, act_dim), dtype=np.float32)

    def add(
        self,
        obs,
        state,
        masks,
        actions,
        rewards,
        next_obs,
        next_state,
        next_masks,
        dones,
    ):
        self.obs[self.ptr] = obs
        self.states[self.ptr] = state
        self.masks[self.ptr] = masks
        self.actions[self.ptr] = actions
        self.rewards[self.ptr] = rewards
        self.next_obs[self.ptr] = next_obs
        self.next_states[self.ptr] = next_state
        self.next_masks[self.ptr] = next_masks
        self.dones[self.ptr] = dones

        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(self, batch_size):
        idx = np.random.choice(self.size, batch_size, replace=False)

        return (
            torch.tensor(self.obs[idx]).to(self.device),
            torch.tensor(self.states[idx]).to(self.device),
            torch.tensor(self.masks[idx]).to(self.device),
            torch.tensor(self.actions[idx]).to(self.device),
            torch.tensor(self.rewards[idx]).to(self.device),
            torch.tensor(self.next_obs[idx]).to(self.device),
            torch.tensor(self.next_states[idx]).to(self.device),
            torch.tensor(self.next_masks[idx]).to(self.device),
            torch.tensor(self.dones[idx]).to(self.device),
        )

    def __len__(self):
        return self.size
