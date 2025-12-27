import torch
import torch.nn.functional as F
from torch.distributions import Categorical

from parallel.agent import Actor, Critic
from parallel.buffer import ReplayBuffer


class MADDPG:
    def __init__(
        self,
        n_agents,
        obs_dim,
        state_dim,
        act_dim,
        device,
        gamma=0.95,
        tau=0.01,
        lr_actor=1e-4,
        lr_critic=1e-3,
        batch_size=64,
        buffer_size=100000,
    ):
        self.n_agents = n_agents
        self.act_dim = act_dim
        self.device = device
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size

        # === Networks ===
        self.actor = Actor(obs_dim, act_dim).to(device)
        self.actor_target = Actor(obs_dim, act_dim).to(device)

        self.critic = Critic(state_dim, act_dim, n_agents).to(device)
        self.critic_target = Critic(state_dim, act_dim, n_agents).to(device)

        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic_target.load_state_dict(self.critic.state_dict())

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=lr_actor)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=lr_critic)

        # === Buffer ===
        self.memory = ReplayBuffer(
            buffer_size, n_agents, obs_dim, state_dim, act_dim, device
        )

    # ============================================================
    # Action Selection (NO epsilon)
    # ============================================================
    def select_action(self, obs, masks, noise=0.0):
        """
        obs: [N, obs_dim]
        masks: [N, act_dim]
        """
        obs_t = torch.tensor(obs, dtype=torch.float32).to(self.device)
        mask_t = torch.tensor(masks, dtype=torch.float32).to(self.device)

        actions_idx = []
        actions_probs = []

        for i in range(self.n_agents):
            probs = self.actor(obs_t[i].unsqueeze(0), mask_t[i].unsqueeze(0))
            dist = Categorical(probs)
            act = dist.sample()

            actions_idx.append(act.item())
            actions_probs.append(probs.squeeze(0).detach().cpu().numpy())

        return actions_idx, actions_probs

    # ============================================================
    # Training
    # ============================================================
    def update(self):
        (
            obs,
            states,
            actions,
            rewards,
            next_obs,
            next_states,
            next_masks,
            dones,
        ) = self.memory.sample(self.batch_size)

        # -------- Critic Update --------
        with torch.no_grad():
            next_actions = []
            for i in range(self.n_agents):
                probs = self.actor_target(next_obs[:, i, :], next_masks[:, i, :])
                next_actions.append(probs)

            next_actions = torch.cat(next_actions, dim=-1)
            q_next = self.critic_target(next_states, next_actions)

            y = (
                rewards.sum(dim=1, keepdim=True)
                + self.gamma * (1 - dones[:, 0:1]) * q_next
            )

        cur_actions = actions.view(self.batch_size, -1)
        q = self.critic(states, cur_actions)

        critic_loss = F.mse_loss(q, y)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # -------- Actor Update --------
        cur_policy_actions = []
        for i in range(self.n_agents):
            probs = self.actor(obs[:, i, :], None)
            cur_policy_actions.append(probs)

        cur_policy_actions = torch.cat(cur_policy_actions, dim=-1)
        actor_loss = -self.critic(states, cur_policy_actions).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # -------- Soft Update --------
        self._soft_update(self.actor, self.actor_target)
        self._soft_update(self.critic, self.critic_target)

    def _soft_update(self, net, target):
        for p, tp in zip(net.parameters(), target.parameters()):
            tp.data.copy_(self.tau * p.data + (1 - self.tau) * tp.data)
