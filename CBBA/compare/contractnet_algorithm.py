from typing import Dict, List, Optional

from cbba_basic import EPS, NEG_INF


class ContractNetSolver:
    """
    合同网协议（CNP）基线：
    - 以“任务”为中心逐个发布公告
    - 所有可行 UAV 对该任务投标
    - 任务管理者选择最高投标者授标
    - 胜者更新自己的当前执行路径

    与拍卖法的区别：
    1. 拍卖法是“全局最高价优先”；
    2. 合同网是“按任务逐个公告并授标”。
    """

    def __init__(self, max_passes: int = 1, debug: bool = False):
        self.max_passes = max_passes
        self.debug = debug

    @staticmethod
    def _task_sort_key(task):
        # 尽量按目标内时序 + 任务编号稳定排序
        stage = 0
        for idx, t in enumerate(task.target.tasks):
            if t.id == task.id:
                stage = idx
                break
        return (task.target.id, stage, task.id)

    def solve_current_wave(self, env, available_tasks) -> dict:
        if not available_tasks:
            return {
                "assignments": {},
                "iterations": 0,
                "winners": {},
                "winning_bids": {},
            }

        ordered_tasks = sorted(available_tasks, key=self._task_sort_key)
        remaining = {task.id for task in ordered_tasks}
        path_by_uav: Dict[int, List[str]] = {uav.idx: [] for uav in env.uavs}
        winners = {task.id: None for task in ordered_tasks}
        winning_bids = {task.id: NEG_INF for task in ordered_tasks}

        iterations = 0
        for _ in range(max(1, self.max_passes)):
            any_award = False
            for task in ordered_tasks:
                task_id = task.id
                if task_id not in remaining:
                    continue

                iterations += 1
                best_uav_idx: Optional[int] = None
                best_gain = NEG_INF
                best_pos = -1

                # 任务公告：所有 UAV 提交投标
                for uav in env.uavs:
                    current_path = path_by_uav[uav.idx]
                    gain, pos = env._best_insertion(uav, current_path, task_id)
                    if gain <= EPS or pos < 0:
                        continue

                    if env._better_winner(gain, uav.idx, best_gain, best_uav_idx):
                        best_gain = gain
                        best_uav_idx = uav.idx
                        best_pos = pos

                # 管理者授标
                if best_uav_idx is None:
                    continue

                path_by_uav[best_uav_idx].insert(best_pos, task_id)
                winners[task_id] = best_uav_idx
                winning_bids[task_id] = best_gain
                remaining.remove(task_id)
                any_award = True

            if not any_award:
                break

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
