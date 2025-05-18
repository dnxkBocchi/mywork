import math
import datetime
import matplotlib.pyplot as plt
import torch
import numpy as np
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'  # 关键：必须在所有库导入前设置


class Scheduler:
    def __init__(self, agent, model, args):
        self.action_num = args.action_num
        self.state_dim = args.state_dim
        self.batch_size = args.batch_size
        self.debug = args.debug
        self.beta = args.beta
        self.agent = agent
        self.buffer = agent.buffer
        self.target_update = args.target_update
        self.model = model

        self.losses = []
        self.rewards = []
        self.mean_losses = []
        self.mean_rewards = []
        self.step_counter = 0
        self.update_counter = 0
        self.time_step = 0

        self.transition = list()
        self.epsilons = []
        self.epsilon_decay = 5e-4
        self.epsilon_start = 0.5
        self.epsilon_end = 0.01
        self.epsilon = 1

        self.l1s = []
        self.l2s = []
        self.l3s = []

    def createState_time(self, task, ncp_list):
        # state: action * 2 + 1 deadline
        x = torch.zeros(self.state_dim, dtype=torch.float)
        if len(task.succ) == 0:
            return x
        index = 0
        unfinished_tasks_nums = 0
        max_time = max(
            task.deadline, max([task.vm_time_cost[ncp][0] for ncp in ncp_list])
        )

        # task
        x[index] = task.deadline / max_time
        index += 1
        # ncp
        for ncp in ncp_list:
            x[index] = task.vm_time_cost[ncp][0] / max_time
            index += 1
        # for ncp in ncp_list:
        #     unfinished_tasks_nums += ncp.vm.unfinished_tasks_number
        # if unfinished_tasks_nums != 0:
        #     for ncp in ncp_list:
        #         x[index] = ncp.vm.unfinished_tasks_number / unfinished_tasks_nums
        #         index += 1
        for ncp in ncp_list:
            x[index] = ncp.cpu
            index += 1
        return x

    def timeReward(self, state, action):
        deadline = state[0]
        time = state[1 : 1 + self.action_num]

        if time[action] <= deadline:
            if time[action] != min(time):
                time_r = (deadline - time[action]) / (deadline - min(time))
                time_r = time_r.item()
            else:
                time_r = 1
        else:
            if max(time) != time[action]:
                time_r = (deadline - time[action]) / (max(time) - deadline)
                time_r = time_r.item()
            else:
                time_r = -1
        return time_r
    
    def timeReward2(self, state, action):
        deadline = state[0]
        time = state[1 : 1 + self.action_num]
        min_time = min(time)
        max_time = max(time)
        if time[action] == min_time:
            r = 1
        elif time[action] == max_time:
            r = -1
        else:
            # 线性映射到 -1 到 1 的区间
            r = -1 + 2 * (max_time - time[action]) / (max_time - min_time)
        return r
    
    def cpuReward(self, state, action):
        cpu_utilizations = state[1+self.action_num :].numpy()
        # 计算 CPU 利用率的方差
        variance = np.var(cpu_utilizations)
        # 最大可能的方差（假设一个节点利用率为 1，其他为 0）
        max_nodes = len(cpu_utilizations)
        max_variance = ((1 - 1 / max_nodes) ** 2 + (max_nodes - 1) * (0 - 1 / max_nodes) ** 2) / max_nodes
        # 归一化方差得到奖励，方差越小奖励越高
        reward = 1 - (variance / max_variance)
        # 确保奖励在 0 到 1 之间
        if cpu_utilizations[action] > 0.7 :
            reward = -1
        else:
            reward = np.clip(reward, 0, 1)
        return reward

    def reward2(self, state, action, task, vms):
        time_r = self.timeReward2(state, action)
        balance_r = self.cpuReward(state, action)
        if self.debug:
            print(
                "[reward] task.id: {}, time_r: {:.2f}, load_balance_l: {:.2f}".format(
                    task.id,
                    time_r,
                    balance_r,
                )
            )
        return (time_r + balance_r) / 2

    def update_epsilon(self):
        self.epsilon = self.epsilon_end + (
            self.epsilon_start - self.epsilon_end
        ) * math.exp(
            -1.0 * self.update_counter * self.step_counter * self.epsilon_decay
        )

    def trainPlot(self):
        mean = sum(self.losses) / len(self.losses)
        self.mean_losses.append(mean)
        self.losses = []

        mean = sum(self.rewards) / len(self.rewards)
        self.mean_rewards.append(mean)
        self.rewards = []

        if len(self.mean_rewards) % 200 == 0:
            print("pictures")
            plt.plot(self.mean_losses, "-o", linewidth=1, markersize=2)
            plt.xlabel(str(self.target_update * len(self.mean_losses)) + "iterations")
            plt.ylabel("Mean Losses")
            plt.show()

            # plt.plot(self.epsilons, linewidth=1)
            # # plt.title("epsilons");
            # plt.ylabel("Epsilon")
            # plt.show()

            plt.plot(self.mean_rewards, "-o", linewidth=1, markersize=2)
            plt.xlabel(str(self.target_update * len(self.mean_rewards)) + "iterations")
            plt.ylabel("Mean Rewards")
            plt.show()

    def runDQN(self, task, ncp_list, vms, done):
        state = self.createState_time(task, ncp_list)

        action = self.agent.get_action(state, self.epsilon)
        self.time_step += 1
        r = self.reward2(state, action, task, vms)
        # print(task.id, action, r)

        if self.agent.net.training:
            self.rewards.append(r)
            if self.transition:
                self.transition += [state]
                self.buffer.add(*self.transition)
            self.transition = [done, state, action, r]

            if len(self.buffer) >= self.batch_size:
                loss = self.agent.learn(self.buffer.sample())
                self.losses.append(loss)
                self.step_counter += 1
                self.update_epsilon()

                if self.step_counter == self.target_update:
                    self.step_counter = 0
                    self.update_counter += 1
                    self.epsilons.append(self.epsilon)
                    self.trainPlot()

            if done:
                self.transition += [state]
                self.transition[0] = done
                self.buffer.add(*self.transition)
                self.transition = []
                self.time_step = 0

        return action

    def runPPO(self, task, ncp_list, vms, done):
        state = self.createState_time(task, ncp_list)
        action = self.agent.get_action(state)
        if self.agent.net.training:
            r = self.reward2(state, action, task, vms)
            # 若虚拟机坠毁，则奖励为 -1
            if ncp_list[action].fail:
                r = -1
            self.rewards.append(r)
            self.time_step += 1
            self.agent.buffer.rewards.append(r)
            self.agent.buffer.is_terminals.append(done)

            if self.time_step == self.target_update:
                loss, l1, l2, l3 = self.agent.update()
                # self.ppo_loss_plot(l1, l2, l3)
                self.losses.append(loss)
                self.time_step = 0
                self.trainPlot()

            # 如果任务完成（done），则执行策略更新
            # if done:
            #     self.agent.update()

        return action

    def save_hyperparameters(self, args):
        time_str = str(datetime.datetime.now().strftime("%Y-%m-%d_%H-%M"))
        file_path = "logs/" + time_str + "/hyperparameters.txt"
        with open(file_path, "w") as file:
            file.write("Hyperparameters:\n")
            for key, value in vars(args).items():
                file.write(f"{key}: {value}\n")

    def trainPlotFinal(
        self,
        mean_makespan=[],
        load_balance=[],
        succes_deadline_rate=[]
    ):
        time_str = str(datetime.datetime.now().strftime("%Y-%m-%d_%H-%M"))
        log_dir = "logs/" + time_str + "/"
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        print("final pictures")

        if mean_makespan:
            # self.makespan += mean_makespan
            plt.plot(mean_makespan, "-o", linewidth=1, markersize=2)
            plt.xlabel("Episode")
            plt.ylabel("Makespan")
            plt.savefig(log_dir + "_makespan.png", facecolor="w")
            # transparent=False
            plt.show()
            plt.clf()

        if load_balance:
            # self.makespan += mean_makespan
            plt.plot(load_balance, "-o", linewidth=1, markersize=2)
            plt.xlabel("Episode")
            plt.ylabel("load_balance")
            plt.savefig(log_dir + "_load_balance.png", facecolor="w")
            # transparent=False
            plt.show()
            plt.clf()

        if succes_deadline_rate:
            # self.time_rate += succes_deadline_rate
            plt.plot(succes_deadline_rate, "-o", linewidth=1, markersize=2)
            plt.xlabel("Episode")
            plt.ylabel("Time Rate")
            plt.savefig(log_dir + "_time_rate.png", facecolor="w")
            # transparent=False
            plt.show()
            plt.clf()
