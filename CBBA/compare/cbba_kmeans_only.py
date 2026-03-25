import math
import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

from calculate import calculate_all_voyage_distance, calculate_all_voyage_time
from env import Task
from task_allocation import CBBAEnv as BaseCBBAEnv
from task_allocation import EPS, NEG_INF, format_episode_metrics


@dataclass
class AgentCBBAState:
    """CBBA 中每个智能体维护的局部状态。"""

    uav_idx: int
    uav_id: str
    bundle: List[str] = field(default_factory=list)
    path: List[str] = field(default_factory=list)
    y: Dict[str, float] = field(default_factory=dict)
    z: Dict[str, Optional[int]] = field(default_factory=dict)


class ImprovedCBBASolver:
    """
    仅保留一个创新点的 CBBA：
    基于 k-means++ 的聚类分层一致性。

    其余部分均复用 task_allocation.CBBAEnv 中的常规环境逻辑：
    1. 路径收益评估
    2. 最优插入位置计算
    3. 任务释放与真实执行
    """

    def __init__(
        self,
        max_bundle_size: Optional[int] = None,
        max_consensus_rounds: int = 50,
        cluster_count: Optional[int] = None,
        kmeans_iters: int = 8,
        debug: bool = False,
    ):
        self.max_bundle_size = max_bundle_size
        self.max_consensus_rounds = max_consensus_rounds
        self.cluster_count = cluster_count
        self.kmeans_iters = kmeans_iters
        self.debug = debug

    def _log(self, *args):
        if self.debug:
            print(*args)

    def _init_agent_states(
        self, env, available_tasks: List[Task]
    ) -> List[AgentCBBAState]:
        task_ids = [task.id for task in available_tasks]
        states = []
        for uav in env.uavs:
            states.append(
                AgentCBBAState(
                    uav_idx=uav.idx,
                    uav_id=uav.id,
                    y={task_id: NEG_INF for task_id in task_ids},
                    z={task_id: None for task_id in task_ids},
                )
            )
        return states

    def _choose_cluster_count(self, n: int) -> int:
        if n <= 1:
            return 1
        if self.cluster_count is not None:
            return max(1, min(self.cluster_count, n))
        return max(1, min(n, int(round(math.sqrt(n)))))

    def _kmeans_pp_clusters(
        self, env
    ) -> Tuple[List[List[int]], Dict[int, int], List[int]]:
        """基于 UAV 当前坐标做 k-means++ 聚类。"""
        alive_uavs = [uav for uav in env.uavs if getattr(uav, "alive", True)]
        if not alive_uavs:
            return [], {}, []

        points = [
            (uav.idx, float(uav.location[0]), float(uav.location[1]))
            for uav in alive_uavs
        ]
        k = self._choose_cluster_count(len(points))
        if k == 1:
            cluster = [idx for idx, _, _ in points]
            centroid_x = sum(x for _, x, _ in points) / len(points)
            centroid_y = sum(y for _, _, y in points) / len(points)
            leader = min(
                cluster,
                key=lambda idx: (
                    env.euclidean_pos(env.uavs[idx].location, (centroid_x, centroid_y)),
                    idx,
                ),
            )
            return [cluster], {idx: 0 for idx in cluster}, [leader]

        rng = random.Random(42)
        centers: List[Tuple[float, float]] = []
        first = rng.choice(points)
        centers.append((first[1], first[2]))

        while len(centers) < k:
            dist2 = []
            for _, x, y in points:
                best = min((x - cx) ** 2 + (y - cy) ** 2 for cx, cy in centers)
                dist2.append(best)
            total = sum(dist2)
            if total <= EPS:
                for _, x, y in points:
                    if (x, y) not in centers:
                        centers.append((x, y))
                        break
                if len(centers) == len(set(centers)) and len(centers) < k:
                    centers.append(centers[-1])
                continue
            r = rng.random() * total
            acc = 0.0
            selected = None
            for (_, x, y), d2 in zip(points, dist2):
                acc += d2
                if acc >= r:
                    selected = (x, y)
                    break
            centers.append(
                selected if selected is not None else (points[-1][1], points[-1][2])
            )

        assignments = [0] * len(points)
        for _ in range(self.kmeans_iters):
            changed = False
            for i, (_, x, y) in enumerate(points):
                best_cid = min(
                    range(k),
                    key=lambda cid: (
                        (x - centers[cid][0]) ** 2 + (y - centers[cid][1]) ** 2,
                        cid,
                    ),
                )
                if assignments[i] != best_cid:
                    assignments[i] = best_cid
                    changed = True

            new_centers = []
            for cid in range(k):
                members = [
                    (x, y) for (_, x, y), a in zip(points, assignments) if a == cid
                ]
                if members:
                    mx = sum(x for x, _ in members) / len(members)
                    my = sum(y for _, y in members) / len(members)
                    new_centers.append((mx, my))
                else:
                    new_centers.append(centers[cid])
            centers = new_centers
            if not changed:
                break

        clusters: List[List[int]] = [[] for _ in range(k)]
        membership: Dict[int, int] = {}
        for (uav_idx, _, _), cid in zip(points, assignments):
            clusters[cid].append(uav_idx)
            membership[uav_idx] = cid

        compact_clusters = []
        leaders = []
        compact_membership: Dict[int, int] = {}
        for cid, members in enumerate(clusters):
            if not members:
                continue
            centroid = centers[cid]
            leader = min(
                members,
                key=lambda idx: (
                    env.euclidean_pos(env.uavs[idx].location, centroid),
                    idx,
                ),
            )
            new_cid = len(compact_clusters)
            compact_clusters.append(sorted(members))
            leaders.append(leader)
            for idx in members:
                compact_membership[idx] = new_cid

        return compact_clusters, compact_membership, leaders

    def _bundle_construction(
        self,
        env,
        agent_state: AgentCBBAState,
        global_y: Dict[str, float],
        global_z: Dict[str, Optional[int]],
        available_tasks: List[Task],
    ) -> bool:
        """标准 CBBA 的 bundle 构建：选择当前最大边际收益任务进行插入。"""
        uav = env.uavs[agent_state.uav_idx]
        if not getattr(uav, "alive", True):
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

                gain, pos = env.best_insertion(uav, agent_state.path, task_id)
                if gain <= EPS or pos < 0:
                    continue

                known_bid = global_y.get(task_id, NEG_INF)
                known_winner = global_z.get(task_id)
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

                if gain > best_gain + EPS:
                    best_task_id = task_id
                    best_insert_pos = pos
                    best_gain = gain
                elif abs(gain - best_gain) <= EPS:
                    if best_task_id is None or task_id < best_task_id:
                        best_task_id = task_id
                        best_insert_pos = pos
                        best_gain = gain

            if best_task_id is None:
                break

            agent_state.bundle.append(best_task_id)
            agent_state.path.insert(best_insert_pos, best_task_id)
            agent_state.y[best_task_id] = best_gain
            agent_state.z[best_task_id] = agent_state.uav_idx
            changed = True

        return changed

    def _hierarchical_consensus_update(
        self,
        env,
        agent_states: List[AgentCBBAState],
        available_tasks: List[Task],
        clusters: List[List[int]],
    ) -> Tuple[Dict[str, float], Dict[str, Optional[int]], Set[int]]:
        """
        分两层做一致性：
        1. 簇内先由成员竞争产生局部赢家
        2. 各簇局部赢家再参与簇间全局一致性
        """
        task_ids = [task.id for task in available_tasks]
        cluster_claims: List[Tuple[Dict[str, float], Dict[str, Optional[int]]]] = []

        for members in clusters:
            local_y = {task_id: NEG_INF for task_id in task_ids}
            local_z = {task_id: None for task_id in task_ids}
            for task_id in task_ids:
                best_bid = NEG_INF
                best_uav_idx = None
                for uav_idx in members:
                    state = agent_states[uav_idx]
                    if state.z.get(task_id) != uav_idx:
                        continue
                    cand_bid = state.y.get(task_id, NEG_INF)
                    if env.better_winner(cand_bid, uav_idx, best_bid, best_uav_idx):
                        best_bid = cand_bid
                        best_uav_idx = uav_idx
                local_y[task_id] = best_bid
                local_z[task_id] = best_uav_idx
            cluster_claims.append((local_y, local_z))

        global_y = {task_id: NEG_INF for task_id in task_ids}
        global_z = {task_id: None for task_id in task_ids}
        for task_id in task_ids:
            best_bid = NEG_INF
            best_uav_idx = None
            for local_y, local_z in cluster_claims:
                cand_uav_idx = local_z.get(task_id)
                if cand_uav_idx is None:
                    continue
                cand_bid = local_y.get(task_id, NEG_INF)
                if env.better_winner(cand_bid, cand_uav_idx, best_bid, best_uav_idx):
                    best_bid = cand_bid
                    best_uav_idx = cand_uav_idx
            global_y[task_id] = best_bid
            global_z[task_id] = best_uav_idx

        pruned_agents: Set[int] = set()
        for state in agent_states:
            loss_idx = None
            for idx, task_id in enumerate(state.bundle):
                if global_z.get(task_id) != state.uav_idx:
                    loss_idx = idx
                    break

            if loss_idx is None:
                for task_id in task_ids:
                    state.y[task_id] = global_y[task_id]
                    state.z[task_id] = global_z[task_id]
                continue

            pruned_agents.add(state.uav_idx)
            released = set(state.bundle[loss_idx:])
            state.bundle = state.bundle[:loss_idx]
            state.path = [task_id for task_id in state.path if task_id not in released]

            for task_id in released:
                state.y[task_id] = NEG_INF
                state.z[task_id] = None

            for task_id in task_ids:
                if task_id not in released:
                    state.y[task_id] = global_y[task_id]
                    state.z[task_id] = global_z[task_id]

        return global_y, global_z, pruned_agents

    def _rebuild_assignments_from_winners(
        self,
        env,
        global_z: Dict[str, Optional[int]],
        global_y: Dict[str, float],
    ) -> Dict[int, List[Task]]:
        tasks_by_uav: Dict[int, List[str]] = {}
        for task_id, winner_idx in global_z.items():
            if winner_idx is None:
                continue
            if global_y.get(task_id, NEG_INF) <= NEG_INF / 10:
                continue
            tasks_by_uav.setdefault(winner_idx, []).append(task_id)

        assignments: Dict[int, List[Task]] = {}
        for uav_idx, task_ids in tasks_by_uav.items():
            uav = env.uavs[uav_idx]
            path: List[str] = []
            for task_id in sorted(task_ids):
                gain, pos = env.best_insertion(uav, path, task_id)
                if gain <= EPS or pos < 0:
                    continue
                path.insert(pos, task_id)
            if path:
                assignments[uav_idx] = [env.task_dict[task_id] for task_id in path]
        return assignments

    def solve_current_wave(self, env, available_tasks: List[Task]) -> dict:
        if not available_tasks:
            return {
                "assignments": {},
                "iterations": 0,
                "winners": {},
                "winning_bids": {},
                "clusters": [],
                "comm_messages": 0,
            }

        clusters, _, leaders = self._kmeans_pp_clusters(env)
        if not clusters:
            return {
                "assignments": {},
                "iterations": 0,
                "winners": {},
                "winning_bids": {},
                "clusters": [],
                "comm_messages": 0,
            }

        self._log(
            "[CBBA] clusters:", [[env.uavs[idx].id for idx in c] for c in clusters]
        )
        self._log("[CBBA] leaders:", [env.uavs[idx].id for idx in leaders])

        agent_states = self._init_agent_states(env, available_tasks)
        task_ids = [task.id for task in available_tasks]
        global_y = {task_id: NEG_INF for task_id in task_ids}
        global_z = {task_id: None for task_id in task_ids}

        alive_agent_count = sum(1 for uav in env.uavs if getattr(uav, "alive", True))
        iterations = 0
        comm_messages = 0

        while iterations < self.max_consensus_rounds:
            iterations += 1
            any_change = False

            for state in agent_states:
                changed = self._bundle_construction(
                    env, state, global_y, global_z, available_tasks
                )
                any_change = any_change or changed

            prev_global_y = dict(global_y)
            prev_global_z = dict(global_z)

            global_y, global_z, pruned_agents = self._hierarchical_consensus_update(
                env=env,
                agent_states=agent_states,
                available_tasks=available_tasks,
                clusters=clusters,
            )

            changed_tasks = {
                task_id
                for task_id in task_ids
                if prev_global_z.get(task_id) != global_z.get(task_id)
                or abs(
                    prev_global_y.get(task_id, NEG_INF) - global_y.get(task_id, NEG_INF)
                )
                > EPS
            }
            any_change = any_change or bool(pruned_agents) or bool(changed_tasks)

            # 聚类分层一致性的通信估计：
            # 1) 簇内：每个簇按有向全连接交换局部 winner/bid，记为 n_c * (n_c - 1)
            # 2) 簇间：各簇 leader 按有向全连接交换，记为 K * (K - 1)
            intra_cluster_msgs = sum(
                len(cluster) * (len(cluster) - 1) for cluster in clusters
            )
            inter_cluster_msgs = len(clusters) * (len(clusters) - 1)
            comm_messages += intra_cluster_msgs + inter_cluster_msgs

            if not any_change:
                break

        assignments = self._rebuild_assignments_from_winners(env, global_z, global_y)
        return {
            "assignments": assignments,
            "iterations": iterations,
            "cbba_iters": iterations,
            "winners": global_z,
            "winning_bids": global_y,
            "clusters": clusters,
            "comm_messages": comm_messages,
        }


class CBBAEnv(BaseCBBAEnv):
    """
    复用 task_allocation.CBBAEnv 的环境逻辑，仅将求解器替换为 KMeans 分层一致性版本。

    兼容用法：
        from cbba_kmeans_only import CBBAEnv, format_episode_metrics
    """

    def __init__(
        self,
        uavs,
        targets,
        tasks: Optional[List[Task]] = None,
        map_size: float = 100.0,
        max_bundle_size: Optional[int] = None,
        max_consensus_rounds: int = 50,
        cluster_count: Optional[int] = None,
        kmeans_iters: int = 8,
        debug: bool = False,
    ):
        super().__init__(
            uavs=uavs,
            targets=targets,
            tasks=tasks,
            map_size=map_size,
            max_bundle_size=max_bundle_size,
            max_consensus_rounds=max_consensus_rounds,
            debug=debug,
        )
        self.solver = ImprovedCBBASolver(
            max_bundle_size=max_bundle_size,
            max_consensus_rounds=max_consensus_rounds,
            cluster_count=cluster_count,
            kmeans_iters=kmeans_iters,
            debug=debug,
        )

    def reset(self):
        super().reset()
        self.total_comm_messages = 0
        return self

    @staticmethod
    def euclidean_pos(p1: Tuple[float, float], p2: Tuple[float, float]) -> float:
        return math.hypot(p1[0] - p2[0], p1[1] - p2[1])

    @staticmethod
    def better_winner(
        cand_bid: float,
        cand_uav_idx: int,
        best_bid: float,
        best_uav_idx: Optional[int],
    ) -> bool:
        return BaseCBBAEnv._better_winner(
            cand_bid, cand_uav_idx, best_bid, best_uav_idx
        )

    def best_insertion(self, uav, current_path: List[str], candidate_task_id: str):
        return self._best_insertion(uav, current_path, candidate_task_id)

    def run_episode(self) -> dict:
        self.reset()
        while True:
            available_tasks = self._available_tasks()
            if not available_tasks:
                break

            wave_result = self.solver.solve_current_wave(self, available_tasks)
            self.total_cbba_iters += wave_result["iterations"]
            self.total_replan_rounds += 1
            self.total_comm_messages += wave_result.get("comm_messages", 0)

            execute_result = self.execute_assignments(wave_result["assignments"])
            self.round_history.append(
                {
                    "round": self.total_replan_rounds,
                    "available_tasks": [task.id for task in available_tasks],
                    "iterations": wave_result["iterations"],
                    "comm_messages": wave_result.get("comm_messages", 0),
                    "clusters": [
                        [self.uavs[idx].id for idx in cluster]
                        for cluster in wave_result.get("clusters", [])
                    ],
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
            "fitness_count": self.fitness_count,
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
            "total_comm_messages": self.total_comm_messages,
        }


# 兼容旧名称
CBBAKMEnv = CBBAEnv


def format_episode_metrics2(ep: int, result: dict) -> str:
    return (
        f"Ep {ep} | Avg Reward: {result['total_reward']:.3f} | "
        f"Success: {result['success_rate']:.2f} | "
        f"distance: {result['total_distance']:.2f}, "
        f"time: {result['total_time']:.2f} | "
        f"cbba_iters: {result['total_cbba_iters']} | "
        f"comm_msgs: {result.get('total_comm_messages', 0)}"
    )
