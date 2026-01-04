import numpy as np


class PPOBuffer:
    def __init__(
        self, steps_per_epoch, n_agents, obs_dim, state_dim, act_dim, gamma, gae_lambda
    ):
        self.max_size = steps_per_epoch
        self.n_agents = n_agents
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.ptr = 0

        # 初始化存储空间
        self.obs = np.zeros((self.max_size, n_agents, obs_dim), dtype=np.float32)
        self.state = np.zeros(
            (self.max_size, state_dim), dtype=np.float32
        )  # 全局状态每步只有1个(或N个看实现，这里用1个广播)
        self.actions = np.zeros((self.max_size, n_agents), dtype=np.float32)
        self.log_probs = np.zeros((self.max_size, n_agents), dtype=np.float32)
        self.rewards = np.zeros((self.max_size, n_agents), dtype=np.float32)
        self.masks = np.zeros((self.max_size, n_agents, act_dim), dtype=np.float32)
        self.values = np.zeros((self.max_size, 1), dtype=np.float32)  # Critic 预测值
        self.dones = np.zeros((self.max_size, 1), dtype=np.float32)

    def store(self, obs, state, action, log_prob, reward, mask, value, done):
        """存储一步数据"""
        self.obs[self.ptr] = obs
        self.state[self.ptr] = state
        self.actions[self.ptr] = action
        self.log_probs[self.ptr] = log_prob
        self.rewards[self.ptr] = reward
        self.masks[self.ptr] = mask
        self.values[self.ptr] = value
        self.dones[self.ptr] = done
        self.ptr += 1

    def finish_path(self, last_val=0):
        """
        在一个 Episode 结束或 Buffer 满时，计算 GAE 和 Returns
        last_val: 下一个状态的 Value (如果未结束)，结束则为 0
        """
        # 将 state 扩展为 (max_size, n_agents, state_dim) 以便扁平化处理
        # 或者在 update 时处理。这里我们采用 update 时 repeat state 的方式。

        # 展平所有数据，准备给 PPO Update
        # 注意：这里我们把 Time 和 Agent 维度合并 -> [T * N, feature_dim]
        # 这是 Parameter Sharing 的关键，把所有 Agent 的经验视为同一批数据

        # 计算 GAE
        rewards = np.append(
            self.rewards[: self.ptr], np.zeros((1, self.n_agents)), axis=0
        )  # 方便计算
        # 将 last_val 强制转换为 (1, 1) 的二维数组，无论它是标量还是数组
        val_formatted = np.array(last_val).reshape(1, -1)
        values = np.append(self.values[: self.ptr], val_formatted, axis=0)

        # 简单处理：全局 State Value 对所有 Agent 是一样的
        # 但 Reward 是各自的。

        returns = np.zeros_like(self.rewards[: self.ptr])
        advantages = np.zeros_like(self.rewards[: self.ptr])

        gae = 0
        for t in reversed(range(self.ptr)):
            # 如果 done，next_non_terminal = 0
            # 这里的 dones 是全局 done，如果你的环境是全员同时结束，这样没问题
            non_terminal = 1.0 - self.dones[t]

            delta = (
                self.rewards[t] + self.gamma * values[t + 1] * non_terminal - values[t]
            )
            # 注意：values 是 (T, 1)，rewards 是 (T, N)，这里广播一下
            delta = delta + (values[t + 1] * non_terminal * self.gamma)  # 修正维度广播
            # 上面写法有点乱，规范写法如下：

            v_t = values[t]  # [1]
            v_tp1 = values[t + 1]  # [1]
            r_t = self.rewards[t]  # [N]

            delta = r_t + self.gamma * v_tp1 * non_terminal - v_t  # [N]
            gae = delta + self.gamma * self.gae_lambda * non_terminal * gae

            advantages[t] = gae
            returns[t] = gae + v_t

        # 展平数据
        obs_flat = self.obs[: self.ptr].reshape(-1, self.obs.shape[-1])
        # State 需要重复 N 次以匹配 obs 数量
        state_repeated = np.repeat(self.state[: self.ptr], self.n_agents, axis=0)

        actions_flat = self.actions[: self.ptr].flatten()
        log_probs_flat = self.log_probs[: self.ptr].flatten()
        masks_flat = self.masks[: self.ptr].reshape(-1, self.masks.shape[-1])
        returns_flat = returns.flatten()
        advantages_flat = advantages.flatten()

        return {
            "obs": obs_flat,
            "state": state_repeated,
            "actions": actions_flat,
            "log_probs": log_probs_flat,
            "masks": masks_flat,
            "returns": returns_flat,
            "advantages": advantages_flat,
        }

    def clear(self):
        self.ptr = 0
