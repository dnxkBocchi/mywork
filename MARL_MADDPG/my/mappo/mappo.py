import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from torch.distributions import Categorical
from mappo.agent import MAPPOActor, MAPPOCritic


class MAPPO:
    def __init__(
        self,
        n_agents,
        obs_dim,
        state_dim,
        act_dim,
        device,
        lr_actor=3e-4,
        lr_critic=1e-3,
        gamma=0.99,
        gae_lambda=0.95,
        clip_param=0.2,
        ppo_epoch=10,
        mini_batch_size=64,
        entropy_coef=0.01,
        value_loss_coef=0.5,
    ):
        self.n_agents = n_agents
        self.device = device
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_param = clip_param
        self.ppo_epoch = ppo_epoch
        self.mini_batch_size = mini_batch_size
        self.entropy_coef = entropy_coef
        self.value_loss_coef = value_loss_coef

        # === 关键：参数共享 (Parameter Sharing) ===
        # 所有 UAV 使用同一个 Policy 和 Value Network
        self.actor = MAPPOActor(obs_dim, act_dim).to(device)
        self.critic = MAPPOCritic(state_dim).to(device)

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr_actor)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr_critic)

    def select_action(self, obs, masks):
        """
        Rollout 阶段选择动作
        """
        # obs: [n_agents, obs_dim] -> Tensor
        obs_t = torch.FloatTensor(obs).to(self.device)
        masks_t = torch.FloatTensor(masks).to(self.device)

        with torch.no_grad():
            probs = self.actor(obs_t, masks_t)
            dist = Categorical(probs)
            actions = dist.sample()
            action_log_probs = dist.log_prob(actions)

            # Critic 评估当前状态价值，用于后续计算 Advantage
            # 注意：Critic 需要 Global State，这里仅演示 Action 选择，
            # 实际 Rollout 循环里通常会显式调用 get_value

        return (
            actions.cpu().numpy(),  # [n_agents] 整数索引
            action_log_probs.cpu().numpy(),  # [n_agents] 对应的 log概率
        )

    def get_value(self, state):
        """获取状态价值"""
        state_t = torch.FloatTensor(state).to(self.device)
        with torch.no_grad():
            value = self.critic(state_t)
        return value.cpu().numpy().flatten()  # [1]

    def update(self, rollout_buffer):
        """
        PPO 更新主函数
        rollout_buffer: 存储了一个 Episode (或固定步数) 的数据
        """
        # 1. 计算 GAE (优势函数)
        advantages = []
        gae = 0

        # 提取 buffer 数据
        # 假设 buffer.rewards 是 [Time_Steps, n_agents]
        # values 是 [Time_Steps + 1, 1] (Critic对全局状态的打分)

        # 将数据转为 Tensor
        obs_batch = torch.FloatTensor(rollout_buffer["obs"]).to(
            self.device
        )  # [T*N, obs_dim]
        state_batch = torch.FloatTensor(rollout_buffer["state"]).to(
            self.device
        )  # [T*N, state_dim]
        actions_batch = torch.FloatTensor(rollout_buffer["actions"]).to(
            self.device
        )  # [T*N]
        old_log_probs_batch = torch.FloatTensor(rollout_buffer["log_probs"]).to(
            self.device
        )  # [T*N]
        masks_batch = torch.FloatTensor(rollout_buffer["masks"]).to(
            self.device
        )  # [T*N, act_dim]

        # 计算 Returns 和 Advantages
        # 这里为了简化，假设 buffer 已经处理好了 returns 和 advantages
        # 如果没有，需要在这里编写 GAE 循环 (参考下方 buffer 实现)
        returns_batch = torch.FloatTensor(rollout_buffer["returns"]).to(self.device)
        advantages_batch = torch.FloatTensor(rollout_buffer["advantages"]).to(
            self.device
        )

        # 归一化 Advantage (重要技巧)
        advantages_batch = (advantages_batch - advantages_batch.mean()) / (
            advantages_batch.std() + 1e-8
        )

        # 2. PPO Epoch 迭代
        dataset_length = obs_batch.shape[0]  # T * N

        for _ in range(self.ppo_epoch):
            # 随机 Batch 采样
            indices = np.random.permutation(dataset_length)

            for start in range(0, dataset_length, self.mini_batch_size):
                end = start + self.mini_batch_size
                batch_idx = indices[start:end]

                # 取出 Mini-batch
                b_obs = obs_batch[batch_idx]
                b_state = state_batch[batch_idx]
                b_actions = actions_batch[batch_idx]
                b_old_log_probs = old_log_probs_batch[batch_idx]
                b_masks = masks_batch[batch_idx]
                b_returns = returns_batch[batch_idx]
                b_advantages = advantages_batch[batch_idx]

                # --- Actor Loss ---
                # 计算新的 log_probs 和 entropy
                new_log_probs, dist_entropy = self.actor.evaluate(
                    b_obs, b_actions, b_masks
                )

                # Ratio
                ratio = torch.exp(new_log_probs - b_old_log_probs)

                # Clip
                surr1 = ratio * b_advantages
                surr2 = (
                    torch.clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param)
                    * b_advantages
                )
                actor_loss = -torch.min(surr1, surr2).mean()

                # --- Critic Loss ---
                new_values = self.critic(b_state).squeeze(-1)
                critic_loss = F.mse_loss(new_values, b_returns)

                # --- Total Loss ---
                loss = (
                    actor_loss
                    + self.value_loss_coef * critic_loss
                    - self.entropy_coef * dist_entropy.mean()
                )

                # Update
                self.actor_optimizer.zero_grad()
                self.critic_optimizer.zero_grad()
                loss.backward()

                # 梯度裁剪
                nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5)
                nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5)

                self.actor_optimizer.step()
                self.critic_optimizer.step()
