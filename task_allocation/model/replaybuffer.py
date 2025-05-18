from collections import deque, namedtuple
import random
import torch
import numpy as np


class ReplayBuffer:
    """用于存储经验元组的固定大小缓冲区。s."""

    def __init__(self, buffer_size, batch_size, device):
        self.device = device
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.experience = namedtuple(
            "Experience",
            field_names=["done", "state", "action", "reward", "next_state"],
        )

    def add(
        self,
        done: bool,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
    ):
        """为memory增添新的ex。"""
        e = self.experience(done, state, action, reward, next_state)
        self.memory.append(e)

    def sample(self):
        """从记memory中随机抽取一批经验"""
        experiences = random.sample(self.memory, k=self.batch_size)
        states = (
            torch.from_numpy(np.stack([e.state for e in experiences if e is not None]))
            .float()
            .to(self.device)
        )
        actions = (
            torch.from_numpy(
                np.vstack([e.action for e in experiences if e is not None])
            )
            .float()
            .to(self.device)
        )
        rewards = (
            torch.from_numpy(
                np.vstack([e.reward for e in experiences if e is not None])
            )
            .float()
            .to(self.device)
        )
        next_states = (
            torch.from_numpy(
                np.stack([e.next_state for e in experiences if e is not None])
            )
            .float()
            .to(self.device)
        )
        dones = (
            torch.from_numpy(
                np.vstack([e.done for e in experiences if e is not None]).astype(
                    np.uint8
                )
            )
            .float()
            .to(self.device)
        )
        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        return len(self.memory)
