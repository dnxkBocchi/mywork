import numpy as np
import torch
import matplotlib.pyplot as plt
from serial.maddpg import MADDPG
from serial.runSerialEnv import DynamicUAVEnv
from env import load_different_scale_csv
from calculate import *

device = "cuda" if torch.cuda.is_available() else "cpu"


def init_env_and_maddpg(scale, device):
    n_agents = scale
    max_neighbors = 3
    obs_dim = 10 * (max_neighbors + 1) + 5
    total_obs_dim = obs_dim * scale

    uav_csv = "data/test/uav.csv"
    task_csv = "data/test/task.csv"
    uavs, tasks, targets = load_different_scale_csv(uav_csv, task_csv, size=scale)
    env = DynamicUAVEnv(uavs, targets, tasks, max_neighbors)
    maddpg = MADDPG(
        n_agents=n_agents, obs_dim=obs_dim, total_obs_dim=total_obs_dim, device=device
    )

    return env, maddpg, n_agents


def select_action_with_mask(env, maddpg, obs_all, epsilon, n_agents):
    # 原始动作 (10,1)
    raw_actions = maddpg.select_action(obs_all)
    flat_actions = raw_actions.flatten()

    mask = []
    current_task = env.task
    valid_exists = False
    for uav in env.uavs:
        if check_constraints(uav, current_task):
            mask.append(0.0)
            valid_exists = True
        else:
            mask.append(-1e9)
    if not valid_exists:
        mask = [0.0] * len(env.uavs)
    masked_actions = flat_actions + np.array(mask)
    # epsilon-greedy
    if np.random.random() > epsilon:
        chosen = np.argmax(masked_actions)
    else:
        valid_indices = [i for i, m in enumerate(mask) if m > -1e8]
        if valid_indices:
            chosen = np.random.choice(valid_indices)
        else:
            chosen = np.random.randint(0, n_agents)

    return chosen, raw_actions


def train_maddpg(scale, episodes=500):
    epsilon = 0.9
    epsilon_decay = 0.995
    min_epsilon = 0.05
    env, maddpg, n_agents = init_env_and_maddpg(scale, device)
    rewards_per_episode = []
    rewards_per10_episode = []

    for ep in range(episodes):
        env.reset()
        done = False
        total_reward = 0
        total_success = 0
        total_fitness = 0

        while not done:
            obs_all = env.get_obs_all()
            chosen, raw_actions = select_action_with_mask(
                env, maddpg, obs_all, epsilon, n_agents
            )
            fitness = calculate_fitness_r(env.task, env.uavs[chosen])
            next_state, reward, done, _ = env.step_serial(chosen)
            total_reward += reward
            total_fitness += fitness
            if reward > 0:
                total_success += 1
            if next_state is None:
                break
            next_obs_all = env.get_obs_all()
            maddpg.buffer.push(obs_all, raw_actions, reward, next_obs_all, done)
            maddpg.update()

        epsilon = max(min_epsilon, epsilon * epsilon_decay)
        num_tasks = len(env.targets) * 3
        rewards_per_episode.append(total_reward / num_tasks)
        if ep % 10 == 0:
            rewards_per10_episode.append(np.mean(rewards_per_episode))
            rewards_per_episode = []

        print_and_plot(
            ep,
            total_reward,
            total_success,
            total_fitness,
            env,
            num_tasks,
            rewards_per10_episode,
        )


def print_and_plot(
    ep,
    total_reward,
    total_success,
    total_fitness,
    env,
    num_tasks,
    rewards_per10_episode,
):
    total_reward /= num_tasks
    total_success /= num_tasks
    total_fitness /= num_tasks

    total_distance = calculate_all_voyage_distance(env.uavs)
    total_time = calculate_all_voyage_time(env.targets)

    print(
        f"Episode {ep} | Total Reward: {total_reward:.2f} | "
        f"Total Fitness: {total_fitness:.2f} | "
        f"Total Distance: {total_distance:.2f} | "
        f"Total Time: {total_time:.2f} | "
        f"Total Success: {total_success:.2f}"
    )

    if (ep + 1) % 50 == 0:
        epsd = len(rewards_per10_episode)
        plt.figure(figsize=(8, 4))
        plt.plot(range(1, epsd + 1), rewards_per10_episode, marker="o")
        plt.xlabel("Episode")
        plt.ylabel("Total Reward")
        plt.ylim(0)
        plt.title(f"Total Reward up to Episode {epsd * 10}")
        plt.grid(True)
        plt.show()
