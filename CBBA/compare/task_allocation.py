import copy
import math
from typing import Dict, List, Optional, Tuple

from calculate import (
    calculate_all_voyage_distance,
    calculate_all_voyage_time,
    calculate_fitness_r,
    calculate_max_possible_voyage_time,
    calculate_reward,
    calculate_voyage_distance,
    calculate_voyage_time,
    check_constraints,
)
from env import Target, Task, Uav
from compare.cbba_basic import CBBASolver, EPS, NEG_INF

debug = True


class CBBAEnv:
    """
    通用任务分配环境 + 默认 CBBA 求解器入口。

    这里负责：
    1. 环境初始化与 reset
    2. 任务释放逻辑（侦察 -> 打击 -> 评估）
    3. 路径评分与最优插入评估
    4. 真正执行任务并更新状态
    5. 统计整个 episode 的指标
    """

    def __init__(
        self,
        uavs: List[Uav],
        targets: List[Target],
        tasks: Optional[List[Task]] = None,
        map_size: float = 100.0,
        max_bundle_size: Optional[int] = None,
        max_consensus_rounds: int = 50,
        debug: bool = False,
    ):
        self.init_uavs = copy.deepcopy(uavs)
        self.init_targets = copy.deepcopy(targets)
        self.map_size = map_size
        self.debug = debug
        self.solver = CBBASolver(
            max_bundle_size=max_bundle_size,
            max_consensus_rounds=max_consensus_rounds,
            debug=debug,
        )
        self.reset()

    @staticmethod
    def euclidean_pos(p1: Tuple[float, float], p2: Tuple[float, float]) -> float:
        return math.hypot(p1[0] - p2[0], p1[1] - p2[1])

    def reset(self):
        self.uavs: List[Uav] = copy.deepcopy(self.init_uavs)
        self.targets: List[Target] = copy.deepcopy(self.init_targets)
        self.tasks: List[Task] = [
            task for target in self.targets for task in target.tasks
        ]

        for idx, uav in enumerate(self.uavs):
            uav.idx = idx
        self.task_dict: Dict[str, Task] = {task.id: task for task in self.tasks}

        self.max_total_voyage, self.max_total_time = calculate_max_possible_voyage_time(
            self.uavs, self.targets
        )
        self.max_total_voyage = max(self.max_total_voyage, 1.0)
        self.max_total_time = max(self.max_total_time, 1.0)

        self.episode_reward = 0.0
        self.success_count = 0
        self.fitness_count = 0.0
        self.total_cbba_iters = 0
        self.total_replan_rounds = 0
        self.round_history: List[dict] = []
        return self

    # =========================
    # 基础工具函数
    # =========================
    def _task_is_released(self, task: Task) -> bool:
        """判断任务是否已解锁（严格按照目标顺序）。"""
        for t in task.target.tasks:
            if t.id == task.id:
                return True
            if not t.flag:
                return False
        return False

    def _available_tasks(self) -> List[Task]:
        return [
            task
            for task in self.tasks
            if (not task.flag) and self._task_is_released(task)
        ]

    def _copy_uav_state(self, uav: Uav) -> Uav:
        return copy.deepcopy(uav)

    def _apply_transition_sim(self, sim_uav: Uav, task: Task):
        """在临时状态上模拟执行某个任务。"""
        dist = calculate_voyage_distance(sim_uav, task)
        travel_time = calculate_voyage_time(sim_uav, task)

        sim_uav.voyage -= dist
        sim_uav.location = task.location
        sim_uav.ammunition -= task.ammunition
        sim_uav.time -= task.time
        sim_uav.task_nums += 1
        sim_uav.end_time += travel_time

    def _evaluate_path_score(self, uav: Uav, path_ids: List[str]) -> float:
        """
        计算一条路径的总收益，无人机按这个顺序执行这些任务时的总收益。
        这里的 path 内任务都来自当前已解锁任务集合，因此不会出现“同轮内再解锁”的情况。
        """
        sim_uav = self._copy_uav_state(uav)
        total_score = 0.0
        for task_id in path_ids:
            task = self.task_dict[task_id]
            if not check_constraints(sim_uav, task):
                return NEG_INF
            reward = calculate_reward(
                sim_uav, task, task.target, self.max_total_voyage, self.max_total_time
            )
            if reward < 0:
                return NEG_INF
            total_score += reward
            self._apply_transition_sim(sim_uav, task)
        return total_score

    def _best_insertion(
        self, uav: Uav, current_path: List[str], candidate_task_id: str
    ) -> Tuple[float, int]:
        """返回最优插入位置和对应的边际收益。"""
        base_score = self._evaluate_path_score(uav, current_path)
        if base_score <= NEG_INF / 10:
            return NEG_INF, -1

        best_gain = NEG_INF
        best_pos = -1
        for pos in range(len(current_path) + 1):
            new_path = current_path[:pos] + [candidate_task_id] + current_path[pos:]
            new_score = self._evaluate_path_score(uav, new_path)
            if new_score <= NEG_INF / 10:
                continue
            gain = new_score - base_score
            if gain > best_gain + EPS or (
                abs(gain - best_gain) <= EPS and (best_pos == -1 or pos < best_pos)
            ):
                best_gain = gain
                best_pos = pos
        return best_gain, best_pos

    @staticmethod
    def _better_winner(
        cand_bid: float,
        cand_uav_idx: int,
        best_bid: float,
        best_uav_idx: Optional[int],
    ) -> bool:
        """统一 tie-break：先比较 bid，再比较 UAV 索引（索引小者优先）。
        当前候选无人机，是否比当前记录的赢家更好"""
        if cand_bid > best_bid + EPS:
            return True
        if abs(cand_bid - best_bid) <= EPS:
            if best_uav_idx is None:
                return True
            return cand_uav_idx < best_uav_idx
        return False

    # =========================
    # 执行与统计
    # =========================
    def _apply_transition_real(self, uav: Uav, task: Task):
        """
        在真实环境中执行任务，并更新时间轴。
        """
        dist = calculate_voyage_distance(uav, task)
        travel_time = calculate_voyage_time(uav, task)
        arrival_time = uav.end_time + travel_time

        predecessor_finish = 0.0
        for prev_task in task.target.tasks:
            if prev_task.id == task.id:
                break
            predecessor_finish = max(predecessor_finish, prev_task.end_time)

        start_time = max(arrival_time, predecessor_finish)
        finish_time = start_time

        task.waiting_time = max(0.0, start_time - arrival_time)
        task.end_time = finish_time
        task.flag = True

        uav.voyage -= dist
        uav.location = task.location
        uav.ammunition -= task.ammunition
        uav.time -= task.time
        uav.end_time = finish_time
        uav.task_nums += 1

        if task.type == 3:
            task.target.total_time = task.end_time

    def execute_assignments(self, assignments: Dict[int, List[Task]]) -> dict:
        executed = []
        round_reward = 0.0
        round_success = 0
        round_fitness = 0.0

        for uav_idx in sorted(assignments.keys()):
            uav = self.uavs[uav_idx]
            for task in assignments[uav_idx]:
                if task.flag:
                    continue
                if not self._task_is_released(task):
                    continue
                if not check_constraints(uav, task):
                    continue

                reward = calculate_reward(
                    uav, task, task.target, self.max_total_voyage, self.max_total_time
                )
                if reward < 0:
                    continue

                self._apply_transition_real(uav, task)
                fit = 0

                round_reward += reward
                round_success += 1
                round_fitness += fit
                executed.append((uav.id, task.id, reward, fit))
                uav.tasks.append(task.id)

        self.episode_reward += round_reward
        self.success_count += round_success
        self.fitness_count += round_fitness

        return {
            "executed": executed,
            "round_reward": round_reward,
            "round_success": round_success,
            "round_fitness": round_fitness,
        }

    def run_episode(self) -> dict:
        self.reset()
        toal_comm_messages = 0
        while True:
            available_tasks = self._available_tasks()
            if not available_tasks:
                break

            wave_result = self.solver.solve_current_wave(self, available_tasks)
            self.total_cbba_iters += wave_result["iterations"]
            self.total_replan_rounds += 1
            toal_comm_messages += wave_result["comm_messages"]

            execute_result = self.execute_assignments(wave_result["assignments"])
            self.round_history.append(
                {
                    "round": self.total_replan_rounds,
                    "available_tasks": [task.id for task in available_tasks],
                    "iterations": wave_result["iterations"],
                    "winners": {
                        task_id: (None if uav_idx is None else self.uavs[uav_idx].id)
                        for task_id, uav_idx in wave_result["winners"].items()
                    },
                    "executed": execute_result["executed"],
                }
            )

            if execute_result["round_success"] == 0:
                break

            if all(task.flag for task in self.tasks):
                break

        tasks_num = len(self.tasks)
        total_distance = calculate_all_voyage_distance(self.uavs)
        total_time = calculate_all_voyage_time(self.targets)
        unassigned_tasks = [task.id for task in self.tasks if not task.flag]

        return {
            "tasks_num": tasks_num,
            "success_count": self.success_count,
            "fitness_count": 0,
            "episode_reward": self.episode_reward,
            "success_rate": self.success_count / tasks_num if tasks_num else 0.0,
            "fitness_rate": 0 / tasks_num if tasks_num else 0.0,
            "total_reward": self.episode_reward / tasks_num if tasks_num else 0.0,
            "total_distance": total_distance,
            "total_time": total_time,
            "unassigned_tasks": unassigned_tasks,
            "done": len(unassigned_tasks) == 0,
            "replan_rounds": self.total_replan_rounds,
            "total_cbba_iters": self.total_cbba_iters,
            "round_history": self.round_history,
            "comm_messages": toal_comm_messages,
        }


def format_episode_metrics(ep: int, result: dict) -> str:
    return (
        f"Ep {ep} | Avg Reward: {result['total_reward']:.3f} | "
        f"Success: {result['success_rate']:.2f} | "
        f"distance: {result['total_distance']:.2f}, "
        f"time: {result['total_time']:.2f} | "
        f"common messages: {result.get('comm_messages', 'N/A')} | "
        f"rounds: {result['replan_rounds']}, cbba_iters: {result['total_cbba_iters']}"
    )
