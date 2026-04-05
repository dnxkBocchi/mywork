from typing import Dict, List, Optional

from cbba_basic import EPS, NEG_INF


class ContractNetSolver:
    """
    带通信半径约束的合同网协议（CNP）基线：
    - 每个任务选择一个“任务管理者”（离任务最近的存活 UAV）
    - 只有在通信半径内的 UAV 才能收到公告、提交投标、接收授标
    - 通信量定义：
        1) 管理者 -> 可达 UAV 的公告
        2) 可达且可行 UAV -> 管理者 的投标
        3) 管理者 -> 中标 UAV 的授标
    """

    def __init__(
        self,
        max_passes: int = 1,
        comm_radius: float = 40.0,
        debug: bool = False,
    ):
        self.max_passes = max_passes
        self.comm_radius = comm_radius
        self.debug = debug

    @staticmethod
    def _task_sort_key(task):
        stage = 0
        for idx, t in enumerate(task.target.tasks):
            if t.id == task.id:
                stage = idx
                break
        return (task.target.id, stage, task.id)

    def _dist(self, env, uav_a, uav_b) -> float:
        return env.euclidean_pos(uav_a.location, uav_b.location)

    def _can_communicate(self, env, uav_a, uav_b) -> bool:
        return self._dist(env, uav_a, uav_b) <= self.comm_radius + EPS

    def _choose_manager(self, env, task):
        alive_uavs = [u for u in env.uavs if getattr(u, "alive", True)]
        if not alive_uavs:
            return None
        return min(
            alive_uavs,
            key=lambda u: (env.euclidean_pos(u.location, task.location), u.idx),
        )

    def solve_current_wave(self, env, available_tasks) -> dict:
        if not available_tasks:
            return {
                "assignments": {},
                "iterations": 0,
                "winners": {},
                "winning_bids": {},
                "comm_messages": 0,
            }

        ordered_tasks = sorted(available_tasks, key=self._task_sort_key)
        remaining = {task.id for task in ordered_tasks}
        path_by_uav: Dict[int, List[str]] = {uav.idx: [] for uav in env.uavs}
        winners = {task.id: None for task in ordered_tasks}
        winning_bids = {task.id: NEG_INF for task in ordered_tasks}

        iterations = 0
        comm_messages = 0

        for _ in range(max(1, self.max_passes)):
            any_award = False

            for task in ordered_tasks:
                task_id = task.id
                if task_id not in remaining:
                    continue

                manager = self._choose_manager(env, task)
                if manager is None:
                    continue

                iterations += 1
                best_uav_idx: Optional[int] = None
                best_gain = NEG_INF
                best_pos = -1

                # 能收到公告的 UAV
                reachable_uavs = [
                    uav
                    for uav in env.uavs
                    if getattr(uav, "alive", True)
                    and self._can_communicate(env, manager, uav)
                ]

                # 1) 公告消息：manager -> 其他可达 UAV
                # 自己不给自己发
                comm_messages += sum(
                    1 for uav in reachable_uavs if uav.idx != manager.idx
                )

                # 2) 投标消息：可达且可行 UAV -> manager
                for uav in reachable_uavs:
                    current_path = path_by_uav[uav.idx]
                    gain, pos = env._best_insertion(uav, current_path, task_id)
                    if gain <= EPS or pos < 0:
                        continue

                    # 自己是 manager 时，不记“发给自己”的投标消息
                    if uav.idx != manager.idx:
                        comm_messages += 1

                    if env._better_winner(gain, uav.idx, best_gain, best_uav_idx):
                        best_gain = gain
                        best_uav_idx = uav.idx
                        best_pos = pos

                # 3) 授标消息：manager -> winner
                if best_uav_idx is None:
                    continue

                if best_uav_idx != manager.idx:
                    comm_messages += 1

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
            "comm_messages": comm_messages,
        }
