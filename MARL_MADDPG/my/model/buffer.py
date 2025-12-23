import random
import numpy as np
import torch


class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.pos = 0

    def push(self, obs, act, reward, next_obs, done):
        data = (obs, act, reward, next_obs, done)
        if len(self.buffer) < self.capacity:
            self.buffer.append(data)
        else:
            self.buffer[self.pos] = data
        self.pos = (self.pos + 1) % self.capacity

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        obs, act, reward, next_obs, done = map(np.array, zip(*batch))

        return (
            torch.FloatTensor(obs),
            torch.FloatTensor(act),
            torch.FloatTensor(reward).unsqueeze(1),
            torch.FloatTensor(next_obs),
            torch.FloatTensor(done).unsqueeze(1),
        )

    def __len__(self):
        return len(self.buffer)
