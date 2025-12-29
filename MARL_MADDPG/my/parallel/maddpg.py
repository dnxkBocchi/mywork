import torch
import torch.nn.functional as F
from torch.distributions import Categorical
import numpy as np
import matplotlib.pyplot as plt

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

        # 监控指标
        self.actor_losses = []
        self.critic_losses = []
        self.action_entropies = []
        self.q_values = []
        self.episode_rewards = []

        # 梯度监控
        self.actor_grad_norms = []
        self.critic_grad_norms = []

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

        # 计算平均动作熵
        avg_entropy = np.mean(
            [-(prob * np.log(prob + 1e-8)).sum() for prob in actions_probs]
        )
        self.action_entropies.append(avg_entropy)

        return actions_idx, actions_probs

    # ============================================================
    # Training
    # ============================================================
    def update(self):
        (
            obs,
            states,
            masks,
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
        # 监控梯度范数
        critic_grad_norm = torch.nn.utils.clip_grad_norm_(
            self.critic.parameters(), max_norm=1.0
        )
        self.critic_grad_norms.append(critic_grad_norm.item())
        self.critic_optimizer.step()

        # -------- Actor Update --------
        cur_policy_actions = []
        entropies = []  # 用于收集每个智能体的熵
        for i in range(self.n_agents):
            probs = self.actor(obs[:, i, :], masks[:, i, :])
            cur_policy_actions.append(probs)
            # 计算当前智能体的熵 (H = -Σ p log p)
            entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=1)
            entropies.append(entropy)

        # 将所有智能体的策略合并
        cur_policy_actions = torch.cat(cur_policy_actions, dim=-1)
        q = self.critic(states, cur_policy_actions)
        avg_entropy = torch.mean(torch.stack(entropies, dim=1), dim=1).mean()
        alpha = 0.05  # 可根据任务调整
        # Actor损失 = -Q + α * 平均熵 (最大化Q值和熵)
        actor_loss = -q.mean() + alpha * avg_entropy

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        # 监控梯度范数
        actor_grad_norm = torch.nn.utils.clip_grad_norm_(
            self.actor.parameters(), max_norm=1.0
        )
        self.actor_grad_norms.append(actor_grad_norm.item())
        # print(
        #     "Actor grad norm:",
        #     sum(
        #         p.grad.abs().mean().item()
        #         for p in self.actor.parameters()
        #         if p.grad is not None
        #     ),
        # )
        self.actor_optimizer.step()

        # -------- Soft Update --------
        self._soft_update(self.actor, self.actor_target)
        self._soft_update(self.critic, self.critic_target)

        # 记录损失
        self.actor_losses.append(actor_loss.item())
        self.critic_losses.append(critic_loss.item())
        self.q_values.append(q.mean().item())

    def _soft_update(self, net, target):
        for p, tp in zip(net.parameters(), target.parameters()):
            tp.data.copy_(self.tau * p.data + (1 - self.tau) * tp.data)

    def log_episode_reward(self, reward):
        """记录episode奖励"""
        self.episode_rewards.append(reward)

    def plot_training_curves(self, save_path=None):
        """绘制训练曲线"""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))

        # 1. Episode Rewards
        if len(self.episode_rewards) > 0:
            axes[0, 0].plot(self.episode_rewards)
            axes[0, 0].set_title("Episode Rewards Over Time")
            axes[0, 0].set_xlabel("Episode")
            axes[0, 0].set_ylabel("Total Reward")
            axes[0, 0].grid(True)

            # 添加移动平均线
            if len(self.episode_rewards) > 10:
                moving_avg = np.convolve(
                    self.episode_rewards, np.ones(10) / 10, mode="valid"
                )
                axes[0, 0].plot(
                    range(9, len(self.episode_rewards)),
                    moving_avg,
                    color="red",
                    linewidth=2,
                    label="Moving Average (10)",
                )
                axes[0, 0].legend()

        # 2. Actor Loss
        if len(self.actor_losses) > 0:
            axes[0, 1].plot(self.actor_losses, alpha=0.7)
            axes[0, 1].set_title("Actor Loss Over Time")
            axes[0, 1].set_xlabel("Update Step")
            axes[0, 1].set_ylabel("Actor Loss")
            axes[0, 1].grid(True)

            # 添加移动平均
            if len(self.actor_losses) > 50:
                avg_actor_loss = np.convolve(
                    self.actor_losses, np.ones(50) / 50, mode="valid"
                )
                axes[0, 1].plot(
                    range(49, len(self.actor_losses)),
                    avg_actor_loss,
                    color="red",
                    linewidth=2,
                    label="Moving Average (50)",
                )
                axes[0, 1].legend()

        # 3. Critic Loss
        if len(self.critic_losses) > 0:
            axes[0, 2].plot(self.critic_losses, alpha=0.7)
            axes[0, 2].set_title("Critic Loss Over Time")
            axes[0, 2].set_xlabel("Update Step")
            axes[0, 2].set_ylabel("Critic Loss")
            axes[0, 2].grid(True)

            # 添加移动平均
            if len(self.critic_losses) > 50:
                avg_critic_loss = np.convolve(
                    self.critic_losses, np.ones(50) / 50, mode="valid"
                )
                axes[0, 2].plot(
                    range(49, len(self.critic_losses)),
                    avg_critic_loss,
                    color="red",
                    linewidth=2,
                    label="Moving Average (50)",
                )
                axes[0, 2].legend()

        # 4. Action Entropy (Exploration)
        if len(self.action_entropies) > 0:
            axes[1, 0].plot(self.action_entropies, alpha=0.7)
            axes[1, 0].set_title("Action Entropy (Exploration)")
            axes[1, 0].set_xlabel("Step")
            axes[1, 0].set_ylabel("Entropy")
            axes[1, 0].grid(True)

            # 添加移动平均
            if len(self.action_entropies) > 50:
                avg_entropy = np.convolve(
                    self.action_entropies, np.ones(50) / 50, mode="valid"
                )
                axes[1, 0].plot(
                    range(49, len(self.action_entropies)),
                    avg_entropy,
                    color="red",
                    linewidth=2,
                    label="Moving Average (50)",
                )
                axes[1, 0].legend()

        # 5. Q-Values
        if len(self.q_values) > 0:
            axes[1, 1].plot(self.q_values, alpha=0.7)
            axes[1, 1].set_title("Average Q-Values")
            axes[1, 1].set_xlabel("Update Step")
            axes[1, 1].set_ylabel("Q-Value")
            axes[1, 1].grid(True)

            # 添加移动平均
            if len(self.q_values) > 50:
                avg_q = np.convolve(self.q_values, np.ones(50) / 50, mode="valid")
                axes[1, 1].plot(
                    range(49, len(self.q_values)),
                    avg_q,
                    color="red",
                    linewidth=2,
                    label="Moving Average (50)",
                )
                axes[1, 1].legend()

        # 6. Gradient Norms
        if len(self.actor_grad_norms) > 0 and len(self.critic_grad_norms) > 0:
            axes[1, 2].plot(
                self.actor_grad_norms, alpha=0.7, label="Actor Grads", color="blue"
            )
            axes[1, 2].plot(
                self.critic_grad_norms, alpha=0.7, label="Critic Grads", color="orange"
            )
            axes[1, 2].set_title("Gradient Norms")
            axes[1, 2].set_xlabel("Update Step")
            axes[1, 2].set_ylabel("Gradient Norm")
            axes[1, 2].legend()
            axes[1, 2].grid(True)

        plt.tight_layout()
        plt.show()
