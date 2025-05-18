import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import datetime

from .replaybuffer import ReplayBuffer


class Network(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_size):
        """Initialization."""
        super(Network, self).__init__()

        self.layers = nn.Sequential(
            nn.Linear(in_dim, hidden_size),
            nn.Tanh(),  # ReLU
            nn.Linear(hidden_size, hidden_size * 2),
            nn.Tanh(),  # ReLU
            nn.Linear(hidden_size * 2, hidden_size),
            nn.Tanh(),  # ReLU
            nn.Linear(hidden_size, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward method implementation."""
        return self.layers(x)


class DQN:
    def __init__(self, args, model):
        self.action_num = args.action_num
        self.state_dim = args.state_dim
        self.batch_size = args.batch_size
        self.hidden_size = args.hidden_size
        self.discount_factor = args.discount_factor
        self.target_update = args.target_update
        self.device = args.device
        self.debug = args.debug

        self.step_counter = 0

        self.buffer = ReplayBuffer(args.buffer_size, self.batch_size, self.device)

        # dqn model
        self.net = Network(self.state_dim, self.action_num, self.hidden_size).to(
            self.device
        )
        self.dqn_target = Network(self.state_dim, self.action_num, self.hidden_size).to(
            self.device
        )
        self.dqn_target.load_state_dict(self.net.state_dict())
        self.dqn_target.eval()

        # optimizer
        self.optimizer = torch.optim.Adam(
            self.net.parameters(), lr=args.lr, weight_decay=args.l2_weight_decay
        )

    def get_action(self, state, epsilon):
        # epsilon greedy policy
        if self.net.training and epsilon > random.uniform(0, 1):  # [0 , 1)
            return random.randint(0, self.action_num - 1)
        else:
            state = state.to(self.device)
            # 将状态移动到同一设备，因为model在GPU上，而buffer里面开始存的都是在cpu
            action = self.net(state).argmax().detach().cpu().item()
            return action

    # TD误差
    def computeLoss(self, samples):
        states, actions, rewards, next_states, dones = samples

        # gather 操作，基于智能体在每个状态下实际选择的动作（由 action_index 指定），
        # 计算当前状态下所采取动作的Q值
        curr_q_value = self.net(states).gather(1, actions.long())

        # DQN
        # 用于估计从当前状态转移到下一个状态后的预期回报，并根据当前奖励和未来回报更新 Q 值。
        if self.discount_factor:
            next_q_value = (
                self.dqn_target(next_states).max(dim=1, keepdim=True)[0].detach()
            )

            # Double DQN 改进: 使用主网络选择动作
            next_action = self.net(next_states).argmax(dim=1, keepdim=True)
            next_q_value = self.dqn_target(next_states).gather(1, next_action).detach()
            target = (rewards + self.discount_factor * next_q_value * (1 - dones)).to(
                self.device
            )
        else:
            target = (rewards).to(self.device)
        loss = F.mse_loss(curr_q_value, target)
        # loss = F.smooth_l1_loss(curr_q_value, target)
        return loss

    def learn(self, example):
        self.step_counter += 1
        if self.step_counter == self.target_update:
            self.dqn_target.load_state_dict(self.net.state_dict())
            self.step_counter = 0
        self.optimizer.zero_grad()
        loss = self.computeLoss(example)
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def save_model(self, path):
        time_str = str(datetime.datetime.now().strftime("%Y-%m-%d_%H-%M"))
        model_path = "logs/" + time_str + "/dqn_model" + path
        torch.save(self.net.state_dict(), model_path)

    def load_model(self, path):
        # self.net.load_state_dict(torch.load(path))
        self.net.load_state_dict(torch.load(path, map_location=torch.device('cpu')))
