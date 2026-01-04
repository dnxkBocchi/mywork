import numpy as np
import torch
from mappo.mappo import MAPPO  # 假设你之后会改这个模型
from mappo.ppo_buffer import PPOBuffer
from runParallelEnv import DynamicUAVEnv
from env import load_different_scale_csv
from calculate import *

device = "cuda" if torch.cuda.is_available() else "cpu"


def train_mappo(scale, episodes=1000):
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
    # 动作维度 (K个任务)
    act_dim = K_local

    print(f"--- Environment Ready ---")
    print(f"Agents: {n_agents}")
    print(f"Dims -> Agent Obs: {obs_dim}, Global State: {state_dim}, Action: {act_dim}")
    print(f"Params -> K_local: {K_local}, M_global: {M_global}")

    # 初始化 MAPPO 和 Buffer
    mappo = MAPPO(n_agents, obs_dim, state_dim, act_dim, device)
    buffer = PPOBuffer(MAX_STEPS, n_agents, obs_dim, state_dim, act_dim, 0.99, 0.95)

    for ep in range(episodes):
        obs, global_s, masks = env.reset()
        episode_reward = 0
        success_count = 0
        fitness_count = 0

        for step in range(MAX_STEPS):
            # 1. 获取动作 & 价值
            # 注意：MAPPO Critic 需要 Global State
            value = mappo.get_value(global_s.reshape(1, -1))  # [1]
            actions_idx, action_log_probs = mappo.select_action(obs, masks)

            # 2. 环境步进
            next_data, rewards, done, info = env.step_parallel(actions_idx)
            next_obs, next_global_s, next_masks = next_data

            # 3. 存入 Buffer
            buffer.store(
                obs,
                global_s,
                actions_idx,
                action_log_probs,
                rewards,
                masks,
                value,
                done,
            )

            obs = next_obs
            global_s = next_global_s
            masks = next_masks
            episode_reward += np.sum(rewards)
            success_count += info.get("success_count", 0)
            fitness_count += info.get("fitness_count", 0)

            if done:
                print(f"Episode finished after {step+1} steps")
                break

        # 4. Episode 结束，计算 GAE 并更新
        # 获取最后一步的 value 用于 GAE 计算
        last_val = mappo.get_value(global_s.reshape(1, -1))
        data_to_update = buffer.finish_path(last_val)

        # 5. 执行 PPO 更新
        mappo.update(data_to_update)

        # 6. 清空 Buffer
        buffer.clear()

        tasks_num = len(env.tasks)
        success_rate = success_count / tasks_num
        fitness_rate = fitness_count / tasks_num
        total_reward = episode_reward / tasks_num
        total_distance = calculate_all_voyage_distance(env.uavs)
        total_time = calculate_all_voyage_time(env.targets)
        print(
            f"Ep {ep} | Avg Reward: {total_reward:.1f} | Success: {success_rate:.2f} | Fitness: {fitness_rate:.2f} \
| distance: {total_distance:.2f}, time: {total_time:.2f}"
        )
