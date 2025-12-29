import numpy as np
import math


class ContractNetSolver:
    def __init__(self, env):
        self.env = env
        # 权重参数，可以根据需求调整
        self.w_dist = 1.0  # 距离的权重（成本）
        self.w_reward = 2.0  # 任务价值的权重（收益）
        self.w_type = 5.0  # 类型匹配的奖励（虽然环境已筛选，但可加权）

    def calculate_bid(self, uav, task):
        """
        计算 UAV 对某个 Task 的投标值 (Bid)
        Bid = 预期收益 - 预期成本
        """
        # 1. 基础物理距离计算 (使用你代码中的逻辑)
        dist = np.sqrt(
            (uav.location[0] - task.location[0]) ** 2
            + (uav.location[1] - task.location[1]) ** 2
        )

        # 2. 航程约束检查 (硬约束)
        # 如果去执行任务会导致回不来或者油不够，Bid 为负无穷
        if uav.voyage < dist:
            return -1e9

        # 3. 计算成本 (Cost)
        # 归一化距离成本，防止数值过大
        norm_dist = dist / self.env.map_size
        cost = self.w_dist * norm_dist

        # 4. 计算收益 (Utility)
        # 你的环境里不同类型任务可能有不同价值，这里简化处理
        # 也可以直接调用 calculate_reward(uav, task, ...) 如果能访问的话
        utility = self.w_reward * 1.0

        # 优先处理紧急任务或先序任务 (根据 ID 或 Type)
        if task.type == 1:  # 假设侦察任务优先级更高
            utility += 1.0

        # 5. 最终标书值
        bid_value = utility - cost
        return bid_value

    def select_action(self):
        """
        为所有 UAV 选择最佳任务的索引
        对应 MADDPG 的 select_action
        """
        actions_idx = []

        # 遍历所有 UAV
        for uav in self.env.uavs:
            if not uav.alive:
                actions_idx.append(0)  # 坠毁的 UAV 随便填
                continue

            # 获取该 UAV 当前视野内的候选任务
            # 注意：必须使用 environment 这一步生成的 candidate_cache
            # 因为 step_parallel 是根据 idx 去这个 cache 里取任务的
            candidates = self.env.candidate_cache.get(uav.idx, [])

            best_bid = -1e9
            best_act_idx = 0  # 默认为 0 (可能是空操作或第一个任务)

            # 如果没有候选任务 (Padding 情况)
            if not candidates:
                actions_idx.append(0)
                continue

            # 遍历 K_local 个候选任务进行“投标”
            # 环境中 K_local 是动作维度
            for act_idx, task in enumerate(candidates):
                # 计算标书
                bid = self.calculate_bid(uav, task)

                # 记录最高出价
                if bid > best_bid:
                    best_bid = bid
                    best_act_idx = act_idx

            # 选择 Bid 最高的那个任务索引
            actions_idx.append(best_act_idx)

        return actions_idx
