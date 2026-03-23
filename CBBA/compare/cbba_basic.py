from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from calculate import check_constraints

NEG_INF = -(10**18)
EPS = 1e-9

debug = False


@dataclass
class AgentCBBAState:
    """CBBA 中每个智能体维护的本地信息。"""

    uav_idx: int
    uav_id: str
    # bundle: 表示这架无人机当前声明想要拿下的任务集合
    bundle: List[str] = field(default_factory=list)
    # path: 表示这些任务真正的执行顺序
    # 因为一个任务加入时，要考虑插入到当前路径的哪个位置收益最高
    # 所以 path 是一个 带顺序优化的执行路径。
    path: List[str] = field(default_factory=list)
    # 当前视角下每个任务的 winning bid； 当前已知最高出价
    y: Dict[str, float] = field(default_factory=dict)
    # 当前视角下每个任务的 winner uav_idx；当前已知赢家是谁
    z: Dict[str, Optional[int]] = field(default_factory=dict)


class CBBASolver:
    """
    只负责 CBBA 算法本身：
    - 初始化 agent 本地状态
    - bundle construction
    - consensus update
    - 根据最终 winner 重建 assignments

    环境相关逻辑（任务解锁、执行、统计）全部放到 task_allocation.py。
    """

    def __init__(
        self,
        max_bundle_size: Optional[int] = None,
        max_consensus_rounds: int = 50,
        debug: bool = False,
    ):
        self.max_bundle_size = max_bundle_size
        self.max_consensus_rounds = max_consensus_rounds
        self.debug = debug

    def _init_agent_states(self, env, available_tasks) -> List[AgentCBBAState]:
        task_ids = [task.id for task in available_tasks]
        states = []
        for uav in env.uavs:
            agent_state = AgentCBBAState(
                uav_idx=uav.idx,
                uav_id=uav.id,
                y={task_id: NEG_INF for task_id in task_ids},
                z={task_id: None for task_id in task_ids},
            )
            states.append(agent_state)
        return states

    def _bundle_construction(
        self,
        env,
        agent_state: AgentCBBAState,
        global_y: Dict[str, float],
        global_z: Dict[str, Optional[int]],
        available_tasks,
    ) -> bool:
        """标准 CBBA 的 bundle construction。"""
        uav = env.uavs[agent_state.uav_idx]
        if not uav.alive:
            return False

        changed = False
        limit = (
            self.max_bundle_size
            if self.max_bundle_size is not None
            else len(available_tasks)
        )

        while len(agent_state.bundle) < limit:
            best_task_id = None
            best_insert_pos = -1
            best_gain = NEG_INF

            for task in available_tasks:
                task_id = task.id
                if task_id in agent_state.bundle:
                    continue
                if not check_constraints(uav, task):
                    continue

                # 这个任务插入当前路径的最优位置
                # 插进去后的边际收益
                gain, pos = env._best_insertion(uav, agent_state.path, task_id)
                if debug:
                    print(
                        f"UAV {uav.id} evaluating task {task_id}: gain={gain:.2f}, "
                        f"pos={pos}, best_gain={best_gain:.2f}, "
                        f"agent_state.path={agent_state.path}"
                    )
                if gain <= EPS:
                    continue

                # 判断“我能不能宣称这个任务归我”
                known_bid = global_y[task_id]
                known_winner = global_z[task_id]
                can_claim = (
                    known_winner is None
                    or known_winner == agent_state.uav_idx
                    or gain > known_bid + EPS
                    or (
                        abs(gain - known_bid) <= EPS
                        and known_winner is not None
                        and agent_state.uav_idx < known_winner
                    )
                )
                if not can_claim:
                    continue

                if env._better_winner(
                    gain,
                    agent_state.uav_idx,
                    best_gain,
                    agent_state.uav_idx if best_task_id is not None else None,
                ):
                    best_gain = gain
                    best_task_id = task_id
                    best_insert_pos = pos

            if best_task_id is None:
                break

            agent_state.bundle.append(best_task_id)
            agent_state.path.insert(best_insert_pos, best_task_id)
            agent_state.y[best_task_id] = best_gain
            agent_state.z[best_task_id] = agent_state.uav_idx
            changed = True

        return changed

    def _consensus_update(
        self,
        env,
        agent_states: List[AgentCBBAState],
        available_tasks,
    ) -> Tuple[Dict[str, float], Dict[str, Optional[int]], bool]:
        """聚合所有 agent 的本地赢家信息，形成新一轮全局一致结果。"""
        changed = False
        task_ids = [task.id for task in available_tasks]
        global_y = {task_id: NEG_INF for task_id in task_ids}
        global_z = {task_id: None for task_id in task_ids}

        for task_id in task_ids:
            best_bid = NEG_INF
            best_uav_idx = None
            for agent_state in agent_states:
                if agent_state.z.get(task_id) != agent_state.uav_idx:
                    continue
                cand_bid = agent_state.y.get(task_id, NEG_INF)
                if env._better_winner(
                    cand_bid, agent_state.uav_idx, best_bid, best_uav_idx
                ):
                    best_bid = cand_bid
                    best_uav_idx = agent_state.uav_idx

            global_y[task_id] = best_bid
            global_z[task_id] = best_uav_idx

        for agent_state in agent_states:
            loss_idx = None
            for idx, task_id in enumerate(agent_state.bundle):
                if global_z.get(task_id) != agent_state.uav_idx:
                    loss_idx = idx
                    break

            if loss_idx is None:
                for task_id in task_ids:
                    agent_state.y[task_id] = global_y[task_id]
                    agent_state.z[task_id] = global_z[task_id]
                continue

            changed = True
            released = set(agent_state.bundle[loss_idx:])
            agent_state.bundle = agent_state.bundle[:loss_idx]
            agent_state.path = [
                task_id for task_id in agent_state.path if task_id not in released
            ]

            for task_id in released:
                agent_state.y[task_id] = NEG_INF
                agent_state.z[task_id] = None

            for task_id in task_ids:
                if task_id not in released:
                    agent_state.y[task_id] = global_y[task_id]
                    agent_state.z[task_id] = global_z[task_id]

        return global_y, global_z, changed

    def _rebuild_assignments_from_winners(
        self,
        env,
        global_z: Dict[str, Optional[int]],
        global_y: Dict[str, float],
    ) -> Dict[int, list]:
        """
        依据最终 winner 向量重建每个 UAV 的可执行路径。
        这样即使 CBBA 在最后一轮刚好停在 prune 之后，也不会出现 winner 已有、assignments 为空的情况。
        """
        tasks_by_uav: Dict[int, List[str]] = {}
        for task_id, winner_idx in global_z.items():
            if winner_idx is None:
                continue
            if global_y.get(task_id, NEG_INF) <= NEG_INF / 10:
                continue
            tasks_by_uav.setdefault(winner_idx, []).append(task_id)

        assignments: Dict[int, list] = {}
        for uav_idx, task_ids in tasks_by_uav.items():
            uav = env.uavs[uav_idx]
            path: List[str] = []
            for task_id in sorted(task_ids):
                gain, pos = env._best_insertion(uav, path, task_id)
                if gain <= EPS or pos < 0:
                    continue
                path.insert(pos, task_id)
            if path:
                assignments[uav_idx] = [env.task_dict[task_id] for task_id in path]
        return assignments

    def solve_current_wave(self, env, available_tasks) -> dict:
        """
        对“当前已解锁任务集合”执行一次完整 CBBA。
        返回当前轮分配结果。
        """
        if not available_tasks:
            return {
                "assignments": {},
                "iterations": 0,
                "winners": {},
                "winning_bids": {},
            }

        agent_states = self._init_agent_states(env, available_tasks)
        task_ids = [task.id for task in available_tasks]
        global_y = {task_id: NEG_INF for task_id in task_ids}
        global_z = {task_id: None for task_id in task_ids}

        iterations = 0
        while iterations < self.max_consensus_rounds:
            iterations += 1
            any_change = False

            for agent_state in agent_states:
                changed = self._bundle_construction(
                    env, agent_state, global_y, global_z, available_tasks
                )
                any_change = any_change or changed

            global_y, global_z, pruned = self._consensus_update(
                env, agent_states, available_tasks
            )
            any_change = any_change or pruned

            if not any_change:
                break

        assignments = self._rebuild_assignments_from_winners(env, global_z, global_y)

        return {
            "assignments": assignments,
            "iterations": iterations,
            "winners": global_z,
            "winning_bids": global_y,
        }
