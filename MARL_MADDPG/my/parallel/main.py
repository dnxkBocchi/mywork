import numpy as np
import torch
from parallel.maddpg import MADDPG  # 假设你之后会改这个模型
from runEnv import DynamicUAVEnv
from env import load_different_scale_csv

device = "cuda" if torch.cuda.is_available() else "cpu"


def train_maddpg_v2(scale, episodes=1000):
    # === 参数配置 ===
    K_local = 5  # 局部观测任务数
    M_global = 30  # 全局 Critic 观测任务数
    MAX_STEPS = 50

    # 1. 初始化环境
    uav_csv = "data/test/uav.csv"
    task_csv = "data/test/task.csv"
    uavs, tasks, targets = load_different_scale_csv(uav_csv, task_csv, size=scale)
    for i, u in enumerate(uavs):
        u.idx = i

    env = DynamicUAVEnv(uavs, targets, tasks, K_local=K_local, M_global=M_global)
    n_agents = len(uavs)

    # 2. 获取维度信息
    # obs: [N, Obs_Dim], global_s: [State_Dim], mask: [N, Act_Dim]
    obs, global_s, masks = env.reset()

    obs_dim = obs.shape[1]  # Actor 输入维度
    state_dim = global_s.shape[0]  # Critic 输入维度
    act_dim = K_local + 1  # 动作维度 (0~K-1: Task, K: Wait)

    print(f"Dims -> Agent Obs: {obs_dim}, Global State: {state_dim}, Action: {act_dim}")

    # 3. 初始化 MADDPG
    # 注意：你需要修改你的 MADDPG 类以接受 state_dim (全局状态) 和 mask
    maddpg = MADDPG(
        n_agents=n_agents,
        obs_dim=obs_dim,
        state_dim=state_dim,  # 新增：Critic 专用输入
        act_dim=act_dim,
        device=device,
    )

    # === 训练循环 ===
    epsilon = 1.0
    min_epsilon = 0.05
    decay = 0.995

    for ep in range(episodes):
        obs, global_s, masks = env.reset()
        episode_reward = 0

        for step in range(MAX_STEPS):
            # 1. 动作选择 (传入 mask)
            # actions_idx: [N] (整数索引)
            # actions_probs: [N, Act_Dim] (用于存 Buffer 训练)
            actions_idx, actions_probs = maddpg.select_action(obs, masks, noise=epsilon)

            # 2. 环境执行
            next_data, rewards, done, info = env.step_parallel(actions_idx)
            next_obs, next_global_s, next_masks = next_data

            # 3. 存储经验
            # Buffer 需要存: (obs, global_s, actions, rewards, next_obs, next_global_s, masks, done)
            # 注意 mask 也要存，因为训练下一时刻动作时也需要 mask
            maddpg.memory.add(
                obs,
                global_s,
                actions_probs,
                rewards,
                next_obs,
                next_global_s,
                next_masks,
                done,
            )

            # 状态流转
            obs = next_obs
            global_s = next_global_s
            masks = next_masks
            episode_reward += np.sum(rewards)

            # 4. 训练更新
            if len(maddpg.memory) > 1000:
                maddpg.update()  # 内部从 buffer 采样并更新

            if done:
                break

        epsilon = max(min_epsilon, epsilon * decay)

        if ep % 10 == 0:
            print(
                f"Episode {ep} | Reward: {episode_reward:.2f} | Epsilon: {epsilon:.2f}"
            )
