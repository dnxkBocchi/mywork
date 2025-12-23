import torch
import torch.optim as optim
from model.agent import Actor, Critic
from model.buffer import ReplayBuffer
import numpy as np


class MADDPG:
    def __init__(
        self,
        n_agents,
        obs_dim,
        total_obs_dim,
        lr_actor=1e-3,
        lr_critic=1e-3,
        gamma=0.95,
        tau=0.01,
        buffer_size=100000,
        batch_size=256,
        device="cpu",
    ):
        self.n_agents = n_agents
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size
        self.device = device

        # actors
        self.actors = [Actor(obs_dim).to(device) for _ in range(n_agents)]
        self.target_actors = [Actor(obs_dim).to(device) for _ in range(n_agents)]

        # centralized critic
        self.critic = Critic(total_obs_dim, n_agents).to(device)
        self.target_critic = Critic(total_obs_dim, n_agents).to(device)

        # optimizers
        self.actor_opts = [
            optim.Adam(actor.parameters(), lr=lr_actor)
            for actor in self.actors
        ]
        self.critic_opt = optim.Adam(self.critic.parameters(), lr=lr_critic)

        # replay buffer
        self.buffer = ReplayBuffer(buffer_size)

        self._hard_update()

    def _hard_update(self):
        for i in range(self.n_agents):
            self.target_actors[i].load_state_dict(self.actors[i].state_dict())
        self.target_critic.load_state_dict(self.critic.state_dict())

    def select_action(self, obs_all, noise=0.1):
        actions = []
        for i in range(self.n_agents):
            obs = torch.FloatTensor(obs_all[i]).unsqueeze(0).to(self.device)
            a = self.actors[i](obs).detach().cpu().numpy()[0]
            a += noise * np.random.randn(*a.shape)
            actions.append(np.clip(a, 0, 1))
        return np.array(actions)

    def update(self):
        if len(self.buffer) < self.batch_size:
            return

        obs, act, reward, next_obs, done = self.buffer.sample(self.batch_size)

        obs = obs.to(self.device)
        act = act.to(self.device)
        reward = reward.to(self.device)
        next_obs = next_obs.to(self.device)
        done = done.to(self.device)

        # -------- Critic update --------
        with torch.no_grad():
            next_actions = []
            for i in range(self.n_agents):
                next_actions.append(
                    self.target_actors[i](next_obs[:, i, :])
                )
            next_actions = torch.cat(next_actions, dim=-1)

            target_q = reward + self.gamma * (1 - done) * \
                       self.target_critic(
                           next_obs.view(next_obs.size(0), -1),
                           next_actions
                       )

        current_q = self.critic(
            obs.view(obs.size(0), -1),
            act.view(act.size(0), -1)
        )

        critic_loss = ((current_q - target_q) ** 2).mean()

        self.critic_opt.zero_grad()
        critic_loss.backward()
        self.critic_opt.step()

        # -------- Actor update --------
        for i in range(self.n_agents):
            cur_actions = []
            for j in range(self.n_agents):
                if i == j:
                    cur_actions.append(self.actors[j](obs[:, j, :]))
                else:
                    cur_actions.append(act[:, j, :])

            cur_actions = torch.cat(cur_actions, dim=-1)

            actor_loss = -self.critic(
                obs.view(obs.size(0), -1),
                cur_actions
            ).mean()

            self.actor_opts[i].zero_grad()
            actor_loss.backward()
            self.actor_opts[i].step()

        self._soft_update()

    def _soft_update(self):
        for i in range(self.n_agents):
            for p, tp in zip(self.actors[i].parameters(), self.target_actors[i].parameters()):
                tp.data.copy_(self.tau * p.data + (1 - self.tau) * tp.data)

        for p, tp in zip(self.critic.parameters(), self.target_critic.parameters()):
            tp.data.copy_(self.tau * p.data + (1 - self.tau) * tp.data)
