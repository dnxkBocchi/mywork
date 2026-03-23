from typing import Dict, List, Optional

from cbba_basic import EPS, NEG_INF


class AuctionSolver:
    """
    简单顺序拍卖基线：
    - 每次在“所有 UAV-任务 对”里选择当前全局最高边际收益的一对
    - 立即将该任务授予胜者，并把任务插入其当前路径
    - 重复直到没有可分配任务

    特点：
    1. 中心化、全局最高价优先；
    2. 不做 CBBA 那种 bundle + consensus；
    3. 适合作为一个容易解释的拍卖对比基线。
    """

    def __init__(self, max_rounds: Optional[int] = None, debug: bool = False):
        self.max_rounds = max_rounds
        self.debug = debug

    def solve_current_wave(self, env, available_tasks) -> dict:
        if not available_tasks:
            return {
                "assignments": {},
                "iterations": 0,
                "winners": {},
                "winning_bids": {},
            }

        remaining = {task.id for task in available_tasks}
        path_by_uav: Dict[int, List[str]] = {uav.idx: [] for uav in env.uavs}
        winners = {task.id: None for task in available_tasks}
        winning_bids = {task.id: NEG_INF for task in available_tasks}

        max_rounds = (
            self.max_rounds if self.max_rounds is not None else len(available_tasks)
        )
        iterations = 0

        while remaining and iterations < max_rounds:
            iterations += 1
            best_task_id = None
            best_uav_idx = None
            best_gain = NEG_INF
            best_pos = -1

            for task in available_tasks:
                task_id = task.id
                if task_id not in remaining:
                    continue

                for uav in env.uavs:
                    current_path = path_by_uav[uav.idx]
                    gain, pos = env._best_insertion(uav, current_path, task_id)
                    if gain <= EPS or pos < 0:
                        continue

                    if env._better_winner(gain, uav.idx, best_gain, best_uav_idx):
                        best_gain = gain
                        best_task_id = task_id
                        best_uav_idx = uav.idx
                        best_pos = pos

            if best_task_id is None:
                break

            path_by_uav[best_uav_idx].insert(best_pos, best_task_id)
            winners[best_task_id] = best_uav_idx
            winning_bids[best_task_id] = best_gain
            remaining.remove(best_task_id)

        assignments: Dict[int, List[object]] = {}
        for uav_idx, path in path_by_uav.items():
            if path:
                assignments[uav_idx] = [env.task_dict[task_id] for task_id in path]

        return {
            "assignments": assignments,
            "iterations": iterations,
            "winners": winners,
            "winning_bids": winning_bids,
        }
