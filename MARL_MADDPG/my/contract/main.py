import numpy as np
import time
from MARL_MADDPG.my.runParallelEnv import DynamicUAVEnv
from env import load_different_scale_csv
from contract.contract_net import ContractNetSolver


def run_cnp(scale, episodes=10):
    # === 参数配置 ===
    K_local = 4
    M_global = (scale + 2) * 3
    N_neighbors = 2
    MAX_STEPS = 30

    uav_csv = "data/test/uav.csv"
    task_csv = "data/test/task.csv"

    # 加载数据
    uavs, tasks, targets = load_different_scale_csv(uav_csv, task_csv, size=scale)
    for i, u in enumerate(uavs):
        u.idx = i

    # 初始化环境
    env = DynamicUAVEnv(uavs, targets, tasks, K_local, M_global, N_neighbors)

    # 开启动态性 (CNP 处理动态性能力很强，可以直接开)
    env.crash_lambda = 0.005
    env.prob_add = 0.3

    # === 初始化 CNP 求解器 ===
    solver = ContractNetSolver(env)

    print(f"--- CNP Solver Ready ---")

    total_rewards = []

    for ep in range(episodes):
        # Reset 环境
        obs, global_s, masks = env.reset()
        episode_reward = 0
        success_count = 0

        start_time = time.time()

        for step in range(MAX_STEPS):
            # 1. 观察：环境在 get_agent_obs 时已经更新了 candidate_cache
            # 这里的 obs 对 CNP 没用，CNP 直接读 env 对象里的对象属性

            # 2. 决策：CNP 计算所有投标并选择最佳 Action Index
            actions_idx = solver.select_action()

            # 3. 执行：环境处理冲突并推演
            # 环境的 step_parallel 会处理多个 UAV 选同一个任务的情况 (Difference Reward 逻辑)
            next_data, rewards, done, info = env.step_parallel(actions_idx)

            next_obs, next_global_s, next_masks = next_data

            # 记录
            episode_reward += np.sum(rewards)
            success_count += info.get("success_count", 0)

            if done:
                print(f"Ep {ep} finished after {step+1} steps.")
                break

        # 统计本局数据
        tasks_num = len(env.tasks)
        avg_reward = episode_reward / tasks_num
        success_rate = success_count / tasks_num
        total_rewards.append(avg_reward)

        print(
            f"Ep {ep} | Avg Reward: {avg_reward:.2f} | Success Rate: {success_rate:.2f} | Time: {time.time()-start_time:.2f}s"
        )

    print(f"Average Reward over {episodes} episodes: {np.mean(total_rewards):.2f}")
