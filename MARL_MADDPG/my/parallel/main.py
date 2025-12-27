import numpy as np
import torch
from parallel.maddpg import MADDPG  # 假设你之后会改这个模型
from parallel.runParallelEnv import DynamicUAVEnv
from env import load_different_scale_csv
from calculate import *

device = "cuda" if torch.cuda.is_available() else "cpu"


def train_maddpg(scale, episodes=1000):
    # === 参数配置 ===
    K_local = 4  # 局部观测任务数
    M_global = (scale + 2) * 3  # 全局 Critic 观测任务数 (安全起见稍微调大)
    N_neighbors = 2  # 最大观测邻居数
    MAX_STEPS = 30  # 单回合最大步数

    uav_csv = "data/test/uav.csv"
    task_csv = "data/test/task.csv"
    uavs, tasks, targets = load_different_scale_csv(uav_csv, task_csv, size=scale)

    # 确保每个 UAV 都有 idx 属性
    for i, u in enumerate(uavs):
        u.idx = i

    n_agents = scale
    env = DynamicUAVEnv(uavs, targets, tasks, K_local, M_global, N_neighbors)
    # 无人机维度，任务维度
    uav_dim, task_dim = env._get_env_dim()
    # Actor 维度 = Self_State + Neighbor_1 + ... + Neighbor_N + Task_1 + ... + Task_K
    obs_dim = uav_dim + N_neighbors * uav_dim + K_local * task_dim
    # Critic 全局状态维度 = All_UAVs + M_global Tasks
    state_dim = env.uav_dim * n_agents + task_dim * M_global
    # 动作维度 (K个任务 + 1个待机 = 4)
    act_dim = K_local + 1

    print(f"--- Environment Ready ---")
    print(f"Agents: {n_agents}")
    print(f"Dims -> Agent Obs: {obs_dim}, Global State: {state_dim}, Action: {act_dim}")
    print(f"Params -> K_local: {K_local}, M_global: {M_global}")

    # 3. 初始化 MADDPG
    maddpg = MADDPG(
        n_agents=n_agents,
        obs_dim=obs_dim,
        state_dim=state_dim,  # Critic 需要全局状态维度
        act_dim=act_dim,
        device=device,
        batch_size=64,  # 批次大小
        lr_actor=1e-4,  # 学习率
        lr_critic=1e-3,
    )

    # 记录曲线
    reward_history = []

    for ep in range(episodes):
        obs, global_s, masks = env.reset()
        episode_reward = 0
        success_count = 0
        fitness_count = 0

        for step in range(MAX_STEPS):
            # print(f"step: {step}")
            # actions_idx: [N] (整数索引, 给环境)
            # actions_probs: [N, Act_Dim] (概率分布, 给 Buffer)
            actions_idx, actions_probs = maddpg.select_action(obs, masks)
            next_data, rewards, done, info = env.step_parallel(actions_idx)
            next_obs, next_global_s, next_masks = next_data
            # 环境返回的 done 是标量 (True/False)，Buffer 需要 [N] 数组
            dones_array = np.array([done] * n_agents, dtype=np.float32)
            maddpg.memory.add(
                obs,
                global_s,
                actions_probs,  # 存 Soft 动作
                rewards,
                next_obs,
                next_global_s,
                next_masks,  # 存下一时刻的 mask (用于 Target Actor)
                dones_array,  # 存广播后的 done
            )

            # 状态流转
            obs = next_obs
            global_s = next_global_s
            masks = next_masks
            episode_reward += np.sum(rewards)
            success_count += info.get("success_count", 0)
            fitness_count += info.get("fitness_count", 0)
            # 只有当 Buffer 里有足够数据时才训练
            if len(maddpg.memory) > 200:  # 预热一下
                maddpg.update()
            if done:
                print(f"Episode finished after {step+1} steps")
                break

        tasks_num = len(env.tasks)
        success_rate = success_count / tasks_num
        fitness_rate = fitness_count / tasks_num
        total_reward = episode_reward / tasks_num
        total_distance = calculate_all_voyage_distance(env.uavs)
        total_time = calculate_all_voyage_time(env.targets)

        reward_history.append(total_reward)
        print(
            f"Ep {ep} | Avg Reward: {total_reward:.1f} | Success: {success_rate:.2f} | Fitness: {fitness_rate:.2f} \
| distance: {total_distance:.2f}, time: {total_time:.2f}"
        )
