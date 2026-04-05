from typing import Dict, List, Optional

from cbba_basic import EPS, NEG_INF


class AuctionSolver:
    """
    带通信半径约束的顺序拍卖基线：
    - 每个任务由一个 auctioneer 发起拍卖（离任务最近的存活 UAV）
    - 只有在 auctioneer 通信半径内的 UAV 才能参与该任务报价
    - 每轮在所有“可通信任务拍卖”中选择全局最高报价授标

    通信量定义：
        1) auctioneer -> 可达 UAV 的拍卖公告
        2) 可达且可行 UAV -> auctioneer 的报价
        3) auctioneer -> winner 的授标
    """

    def __init__(
        self,
        max_rounds: Optional[int] = None,
        comm_radius: float = 40.0,
        debug: bool = False,
    ):
        self.max_rounds = max_rounds
        self.comm_radius = comm_radius
        self.debug = debug

    def _dist(self, env, uav_a, uav_b) -> float:
        return env.euclidean_pos(uav_a.location, uav_b.location)

    def _can_communicate(self, env, uav_a, uav_b) -> bool:
        return self._dist(env, uav_a, uav_b) <= self.comm_radius + EPS

    def _choose_auctioneer(self, env, task):
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

        remaining = {task.id for task in available_tasks}
        path_by_uav: Dict[int, List[str]] = {uav.idx: [] for uav in env.uavs}
        winners = {task.id: None for task in available_tasks}
        winning_bids = {task.id: NEG_INF for task in available_tasks}

        max_rounds = (
            self.max_rounds if self.max_rounds is not None else len(available_tasks)
        )

        iterations = 0
        comm_messages = 0

        while remaining and iterations < max_rounds:
            iterations += 1

            best_task_id = None
            best_uav_idx = None
            best_gain = NEG_INF
            best_pos = -1
            best_auctioneer_idx = None

            # 本轮所有待拍任务都各自发起一次拍卖
            for task in available_tasks:
                task_id = task.id
                if task_id not in remaining:
                    continue

                auctioneer = self._choose_auctioneer(env, task)
                if auctioneer is None:
                    continue

                reachable_uavs = [
                    uav
                    for uav in env.uavs
                    if getattr(uav, "alive", True)
                    and self._can_communicate(env, auctioneer, uav)
                ]

                # 1) 公告
                comm_messages += sum(
                    1 for uav in reachable_uavs if uav.idx != auctioneer.idx
                )

                local_best_uav_idx = None
                local_best_gain = NEG_INF
                local_best_pos = -1

                # 2) 报价
                for uav in reachable_uavs:
                    current_path = path_by_uav[uav.idx]
                    gain, pos = env._best_insertion(uav, current_path, task_id)
                    if gain <= EPS or pos < 0:
                        continue

                    if uav.idx != auctioneer.idx:
                        comm_messages += 1

                    if env._better_winner(
                        gain, uav.idx, local_best_gain, local_best_uav_idx
                    ):
                        local_best_gain = gain
                        local_best_uav_idx = uav.idx
                        local_best_pos = pos

                if local_best_uav_idx is None:
                    continue

                # 在所有任务的局部赢家里，选全局最高价
                if env._better_winner(
                    local_best_gain, local_best_uav_idx, best_gain, best_uav_idx
                ):
                    best_task_id = task_id
                    best_uav_idx = local_best_uav_idx
                    best_gain = local_best_gain
                    best_pos = local_best_pos
                    best_auctioneer_idx = auctioneer.idx

            if best_task_id is None:
                break

            # 3) 授标
            if (
                best_auctioneer_idx is not None
                and best_uav_idx is not None
                and best_auctioneer_idx != best_uav_idx
            ):
                comm_messages += 1

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
            "comm_messages": comm_messages,
        }
