import numpy as np
import torch


class ReplayBuffer:
    def __init__(self, capacity, obs_dim, act_dim, n_agents):
        self.capacity = capacity
        self.ptr = 0
        self.size = 0
        self.n_agents = n_agents

        # Buffer 初始化
        self.obs_buf = np.zeros((capacity, n_agents, obs_dim), dtype=np.float32)
        # 动作 Buffer 现在存的是 One-hot 向量 (n_agents, act_dim)
        self.act_buf = np.zeros((capacity, n_agents, act_dim), dtype=np.float32)
        self.rew_buf = np.zeros((capacity, n_agents), dtype=np.float32)
        self.next_obs_buf = np.zeros((capacity, n_agents, obs_dim), dtype=np.float32)
        self.done_buf = np.zeros((capacity, n_agents), dtype=np.float32)

    def add(self, obs, act, rew, next_obs, done):
        """
        act: 应该是 one-hot 格式 (n_agents, act_dim)
        """
        self.obs_buf[self.ptr] = obs
        self.act_buf[self.ptr] = act
        self.rew_buf[self.ptr] = rew
        self.next_obs_buf[self.ptr] = next_obs
        self.done_buf[self.ptr] = done

        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size):
        idxs = np.random.randint(0, self.size, size=batch_size)

        return (
            torch.FloatTensor(self.obs_buf[idxs]),
            torch.FloatTensor(self.act_buf[idxs]),
            torch.FloatTensor(self.rew_buf[idxs]),
            torch.FloatTensor(self.next_obs_buf[idxs]),
            torch.FloatTensor(self.done_buf[idxs]),
        )

    def __len__(self):
        return self.size
