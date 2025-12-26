import torch
import torch.optim as optim
import torch.nn.functional as F
from parallel.agent import Actor, Critic
from parallel.buffer import ReplayBuffer
import numpy as np


class MADDPG:
    def __init__(
        self,
        n_agents,
        obs_dim,
        total_obs_dim,
        act_dim,  # 每个 Agent 的动作维度 (任务数)
        lr_actor=1e-3,
        lr_critic=1e-3,
        gamma=0.95,
        tau=0.01,
        buffer_size=100000,
        batch_size=256,
        device="cpu",
    ):
        self.n_agents = n_agents
        self.act_dim = act_dim
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size
        self.device = device

        # total_act_dim 用于 Critic 输入，等于所有 Agent 的 One-hot 动作拼接
        total_act_dim = n_agents * act_dim

        # actors
        self.actors = [Actor(obs_dim, act_dim).to(device) for _ in range(n_agents)]
        self.target_actors = [
            Actor(obs_dim, act_dim).to(device) for _ in range(n_agents)
        ]

        # centralized critic
        self.critic = Critic(total_obs_dim, total_act_dim).to(device)
        self.target_critic = Critic(total_obs_dim, total_act_dim).to(device)

        # optimizers
        self.actor_opts = [
            optim.Adam(actor.parameters(), lr=lr_actor) for actor in self.actors
        ]
        self.critic_opt = optim.Adam(self.critic.parameters(), lr=lr_critic)

        # replay buffer
        # 这里的 act_dim 存的是 One-hot 向量，所以是 act_dim
        self.buffer = ReplayBuffer(buffer_size, obs_dim, act_dim, n_agents)

        self._hard_update()

    def _hard_update(self):
        for i in range(self.n_agents):
            self.target_actors[i].load_state_dict(self.actors[i].state_dict())
        self.target_critic.load_state_dict(self.critic.state_dict())

    def select_action(self, obs_all, noise=0.0):
        """
        返回两个动作：
        1. actions_onehot: 用于存 Buffer (n_agents, act_dim)
        2. actions_index: 用于环境 step (n_agents, )
        """
        actions_onehot = []
        actions_index = []

        for i in range(self.n_agents):
            obs = (
                torch.FloatTensor(obs_all[i]).unsqueeze(0).to(self.device)
            )  # (1, obs_dim)
            logits = self.actors[i](obs)  # (1, act_dim)

            # 探索策略：Gumbel-Softmax 或 Epsilon-Greedy
            # 这里使用 Gumbel-Softmax 进行采样探索 (hard=True 返回 one-hot)
            # 也可以手动实现 epsilon-greedy
            if np.random.rand() < noise:
                # 随机探索
                idx = np.random.randint(0, self.act_dim)
                # 转 One-hot
                act_onehot = np.zeros(self.act_dim)
                act_onehot[idx] = 1.0
            else:
                # 选择概率最大的
                action_probs = F.softmax(logits, dim=-1)
                idx = torch.argmax(action_probs, dim=-1).item()
                act_onehot = np.zeros(self.act_dim)
                act_onehot[idx] = 1.0

            actions_onehot.append(act_onehot)
            actions_index.append(idx)

        return np.array(actions_onehot), np.array(actions_index)

    def update(self):
        if len(self.buffer) < self.batch_size:
            return

        # 采样: act 已经是 One-hot 形式
        obs, act, reward, next_obs, done = self.buffer.sample(self.batch_size)

        obs = obs.to(self.device)  # (B, n_agents, obs_dim)
        act = act.to(self.device)  # (B, n_agents, act_dim)
        reward = reward.to(self.device)  # (B, n_agents)
        next_obs = next_obs.to(self.device)  # (B, n_agents, obs_dim)
        done = done.to(self.device)  # (B, n_agents)

        # 扁平化 obs 用于 Critic
        obs_flat = obs.view(self.batch_size, -1)
        next_obs_flat = next_obs.view(self.batch_size, -1)
        act_flat = act.view(self.batch_size, -1)

        # -------- Critic update --------
        with torch.no_grad():
            next_actions_onehot = []
            for i in range(self.n_agents):
                # 目标网络输出 Logits
                target_logits = self.target_actors[i](next_obs[:, i, :])
                # 使用 Gumbel-Softmax 得到离散动作的 One-hot 近似 (可微)
                # hard=False 时返回概率分布，hard=True 返回 One-hot 但带有梯度
                # 训练时通常用 hard=True 或 False 均可，这里用 True 模拟确定性执行
                target_act = F.gumbel_softmax(target_logits, tau=1.0, hard=True)
                next_actions_onehot.append(target_act)

            next_actions_flat = torch.cat(next_actions_onehot, dim=-1)

            target_q = reward + self.gamma * (1 - done) * self.target_critic(
                next_obs_flat, next_actions_flat
            )

        current_q = self.critic(obs_flat, act_flat)
        critic_loss = F.mse_loss(current_q, target_q)

        self.critic_opt.zero_grad()
        critic_loss.backward()
        self.critic_opt.step()

        # -------- Actor update --------
        for i in range(self.n_agents):
            # 获取当前 Agent 的预测动作 (Gumbel Softmax)
            cur_actions_onehot = []
            for j in range(self.n_agents):
                logits = self.actors[j](obs[:, j, :])
                # 注意：Actor 更新时必须要有梯度流回 Actor
                # Gumbel-Softmax 是关键
                gumbel_act = F.gumbel_softmax(logits, tau=1.0, hard=True)
                cur_actions_onehot.append(gumbel_act)

            cur_actions_flat = torch.cat(cur_actions_onehot, dim=-1)

            # 最大化 Q 值 -> 最小化 -Q
            actor_loss = -self.critic(obs_flat, cur_actions_flat).mean()

            # 为了防止 logits 爆炸，可以加一个正则项 (可选)
            # actor_loss += (logits**2).mean() * 1e-3

            self.actor_opts[i].zero_grad()
            actor_loss.backward()
            self.actor_opts[i].step()

        self._soft_update()

    def _soft_update(self):
        for i in range(self.n_agents):
            for p, tp in zip(
                self.actors[i].parameters(), self.target_actors[i].parameters()
            ):
                tp.data.copy_(self.tau * p.data + (1 - self.tau) * tp.data)

        for p, tp in zip(self.critic.parameters(), self.target_critic.parameters()):
            tp.data.copy_(self.tau * p.data + (1 - self.tau) * tp.data)
