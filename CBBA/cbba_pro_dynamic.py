import copy
import math
import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

from calculate import (
    calculate_all_voyage_distance,
    calculate_all_voyage_time,
    calculate_max_possible_voyage_time,
    calculate_voyage_distance,
    calculate_voyage_time,
)
from env import Target, Task, Uav


NEG_INF = -(10**18)
EPS = 1e-9


@dataclass
class AgentCBBAState:
    """改进 CBBA 中每个智能体维护的局部状态。"""

    uav_idx: int
    uav_id: str
    bundle: List[str] = field(default_factory=list)
    path: List[str] = field(default_factory=list)
    y: Dict[str, float] = field(default_factory=dict)
    z: Dict[str, Optional[int]] = field(default_factory=dict)
    feasibility_mask: Dict[str, bool] = field(default_factory=dict)


class ImprovedCBBASolver:
    """
    改进版 CBBA：
    1. 基于可行域掩码的 bundle 构建
    2. 新任务插入采用“最小延迟优先、最大收益择优”
    3. 基于 k-means++ 的聚类分层一致性 + 事件触发式部分重选
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
    def _emit_event_log(self, message: str):
        self.event_logs.append(message)
        print(message)

    def _new_blank_instance(self, cls):
        try:
            return cls.__new__(cls)
        except Exception:
            class _Blank:
                pass

            return _Blank()

    def _recompute_global_bounds(self):
        self.max_total_voyage, self.max_total_time = calculate_max_possible_voyage_time(
            self.uavs, self.targets
        )
        self.max_total_voyage = max(self.max_total_voyage, 1.0)
        self.max_total_time = max(self.max_total_time, 1.0)

    def _build_dynamic_target(
        self,
        target_prefix: str,
        location: Tuple[float, float],
        strike_ammunition: int = 3,
        recon_time: float = 5.0,
        assessment_time: float = 4.0,
    ) -> Target:
        base_prefix = target_prefix
        suffix = 1
        while (
            f"{target_prefix}S" in self.task_dict
            or f"{target_prefix}R" in self.task_dict
            or f"{target_prefix}A" in self.task_dict
        ):
            target_prefix = f"{base_prefix}_{suffix}"
            suffix += 1

        target = self._new_blank_instance(Target)
        target.id = target_prefix
        target.location = location
        target.total_time = 0.0
        target.tasks = []

        def make_task(task_id: str, task_type: int, ammunition: int, time_cost: float):
            task = self._new_blank_instance(Task)
            task.id = task_id
            task.type = task_type
            task.location = location
            task.ammunition = ammunition
            task.time = time_cost
            task.flag = False
            task.waiting_time = 0.0
            task.end_time = 0.0
            task.target = target
            task.strike = 1.0 if task_type == 1 else 0.0
            task.reconnaissance = 1.0 if task_type == 2 else 0.0
            task.assessment = 1.0 if task_type == 3 else 0.0
            return task

        strike_task = make_task(f"{target_prefix}S", 1, strike_ammunition, 0.0)
        recon_task = make_task(f"{target_prefix}R", 2, 0, recon_time)
        assess_task = make_task(f"{target_prefix}A", 3, 0, assessment_time)
        target.tasks = [strike_task, recon_task, assess_task]
        return target

    def add_dynamic_target(
        self,
        target_prefix: str = "TASKDYN01",
        location: Tuple[float, float] = (38.4, 81.2),
        strike_ammunition: int = 3,
        recon_time: float = 5.0,
        assessment_time: float = 4.0,
    ) -> Target:
        target = self._build_dynamic_target(
            target_prefix=target_prefix,
            location=location,
            strike_ammunition=strike_ammunition,
            recon_time=recon_time,
            assessment_time=assessment_time,
        )
        self.targets.append(target)
        self.tasks.extend(target.tasks)
        for task in target.tasks:
            self.task_dict[task.id] = task
        self._recompute_global_bounds()
        self._emit_event_log(
            f"[动态事件-新增目标] 新增目标 {target.id}，位置={target.location}，新增任务={[task.id for task in target.tasks]}"
        )
        return target

    def _select_crash_uav(self, crash_type: int = 2) -> Optional[Uav]:
        candidates = [uav for uav in self.uavs if uav.alive and getattr(uav, "type", None) == crash_type]
        if not candidates:
            return None
        candidates.sort(key=lambda u: (u.idx, u.id))
        return candidates[0]

    def _collect_competition_snapshot(self, tasks: List[Task]) -> Dict[str, List[dict]]:
        snapshot: Dict[str, List[dict]] = {}
        for task in tasks:
            candidates = []
            for uav in self.uavs:
                if not uav.alive:
                    continue
                if not self._static_feasible(uav, task):
                    continue
                gain, pos, delay = self.best_insertion_delay_first(uav, [], task.id)
                if pos < 0 or gain <= EPS:
                    continue
                candidates.append(
                    {
                        "uav_idx": uav.idx,
                        "uav_id": uav.id,
                        "uav_type": uav.type,
                        "gain": gain,
                        "delay": delay,
                        "insert_pos": pos,
                    }
                )
            candidates.sort(key=lambda x: (-x["gain"], x["delay"], x["uav_idx"]))
            snapshot[task.id] = candidates
        return snapshot

    def _winner_reason(self, winner_idx: Optional[int], winner_bid: float, candidates: List[dict]) -> str:
        if winner_idx is None:
            return "没有可行赢家，当前无 UAV 能满足类型/资源/航程约束。"
        winner = next((c for c in candidates if c["uav_idx"] == winner_idx), None)
        if winner is None:
            return f"该 UAV 在一致性阶段保持全局最优 winning bid={winner_bid:.4f}。"
        if len(candidates) == 1:
            return "它是唯一满足任务类型、资源与航程约束的可行竞标者。"
        runner_up = next((c for c in candidates if c["uav_idx"] != winner_idx), None)
        if runner_up is None:
            return "它是唯一有效竞标者。"
        if winner["gain"] > runner_up["gain"] + EPS:
            return (
                f"它的边际收益最高（{winner['gain']:.4f}），高于次优 {runner_up['uav_id']} "
                f"的 {runner_up['gain']:.4f}。"
            )
        if winner["delay"] < runner_up["delay"] - EPS:
            return (
                f"它与其他 UAV 收益接近，但插入延迟更小（{winner['delay']:.4f} < "
                f"{runner_up['delay']:.4f}）。"
            )
        return (
            f"它与其他 UAV 收益接近，按 CBBA 的 tie-break 规则由编号更小的 {winner['uav_id']} 获胜。"
        )

    def _print_competition_report(self, header: str, tasks: List[Task], snapshot: Dict[str, List[dict]], wave_result: dict):
        self._emit_event_log(header)
        for task in tasks:
            candidates = snapshot.get(task.id, [])
            self._emit_event_log(f"  任务 {task.id} (type={task.type}, location={task.location})")
            if not candidates:
                self._emit_event_log("    可行竞争 UAV: 无")
            else:
                for cand in candidates:
                    self._emit_event_log(
                        "    候选 -> "
                        f"{cand['uav_id']} (type={cand['uav_type']}), "
                        f"gain={cand['gain']:.4f}, delay={cand['delay']:.4f}, pos={cand['insert_pos']}"
                    )
            winner_idx = wave_result["winners"].get(task.id)
            winner_bid = wave_result["winning_bids"].get(task.id, NEG_INF)
            if winner_idx is None:
                self._emit_event_log("    最终赢家: 无")
                self._emit_event_log("    原因: 没有 UAV 成功获得该任务。")
            else:
                winner_uav = self.uavs[winner_idx]
                reason = self._winner_reason(winner_idx, winner_bid, candidates)
                self._emit_event_log(
                    f"    最终赢家: {winner_uav.id} (type={winner_uav.type}), winning_bid={winner_bid:.4f}"
                )
                self._emit_event_log(f"    原因: {reason}")

    def _merge_assignments(self, base_assignments: Dict[int, List[Task]], extra_assignments: Dict[int, List[Task]]) -> Dict[int, List[Task]]:
        merged: Dict[int, List[Task]] = {idx: list(tasks) for idx, tasks in base_assignments.items()}
        for uav_idx, task_list in extra_assignments.items():
            merged.setdefault(uav_idx, [])
            existing_ids = {task.id for task in merged[uav_idx]}
            for task in task_list:
                if task.id not in existing_ids:
                    merged[uav_idx].append(task)
                    existing_ids.add(task.id)
        return merged

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
                    feasibility_mask={task_id: False for task_id in task_ids},
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
        alive_uavs = [uav for uav in env.uavs if uav.alive]
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
            feasibility_mask = env.build_feasibility_mask(
                uav=uav,
                current_path=agent_state.path,
                available_tasks=available_tasks,
                excluded_task_ids=set(agent_state.bundle),
            )
            agent_state.feasibility_mask = dict(feasibility_mask)

            best_task_id = None
            best_insert_pos = -1
            best_gain = NEG_INF
            best_delay = float("inf")

            for task in available_tasks:
                task_id = task.id
                if not feasibility_mask.get(task_id, False):
                    continue

                gain, pos, delay = env.best_insertion_delay_first(
                    uav, agent_state.path, task_id
                )
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

                better = False
                if delay < best_delay - EPS:
                    better = True
                elif abs(delay - best_delay) <= EPS:
                    if gain > best_gain + EPS:
                        better = True
                    elif (
                        abs(gain - best_gain) <= EPS
                        and best_task_id is not None
                        and task_id < best_task_id
                    ):
                        better = True

                if better or best_task_id is None:
                    best_task_id = task_id
                    best_insert_pos = pos
                    best_gain = gain
                    best_delay = delay

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

    def _select_active_agents(
        self,
        changed_tasks: Set[str],
        prev_global_z: Dict[str, Optional[int]],
        global_z: Dict[str, Optional[int]],
        pruned_agents: Set[int],
        membership: Dict[int, int],
        clusters: List[List[int]],
    ) -> Set[int]:
        """事件触发式局部一致性：仅让受影响 UAV 及其簇内邻居参与下一轮重选。"""
        active_agents = set(pruned_agents)
        for task_id in changed_tasks:
            old_winner = prev_global_z.get(task_id)
            new_winner = global_z.get(task_id)
            for winner in (old_winner, new_winner):
                if winner is None:
                    continue
                active_agents.add(winner)
                cid = membership.get(winner)
                if cid is not None:
                    active_agents.update(clusters[cid])
        return active_agents

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
                gain, pos, _ = env.best_insertion_delay_first(uav, path, task_id)
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
            }

        clusters, membership, leaders = self._kmeans_pp_clusters(env)
        if not clusters:
            return {
                "assignments": {},
                "iterations": 0,
                "winners": {},
                "winning_bids": {},
                "clusters": [],
            }

        self._log(
            "[CBBA] clusters:", [[env.uavs[idx].id for idx in c] for c in clusters]
        )
        self._log("[CBBA] leaders:", [env.uavs[idx].id for idx in leaders])

        agent_states = self._init_agent_states(env, available_tasks)
        task_ids = [task.id for task in available_tasks]
        global_y = {task_id: NEG_INF for task_id in task_ids}
        global_z = {task_id: None for task_id in task_ids}

        active_agents: Set[int] = {uav.idx for uav in env.uavs if uav.alive}
        iterations = 0

        while iterations < self.max_consensus_rounds:
            iterations += 1
            any_change = False

            for state in agent_states:
                if state.uav_idx not in active_agents:
                    continue
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

            if not any_change:
                break

            active_agents = self._select_active_agents(
                changed_tasks=changed_tasks,
                prev_global_z=prev_global_z,
                global_z=global_z,
                pruned_agents=pruned_agents,
                membership=membership,
                clusters=clusters,
            )
            if not active_agents and changed_tasks:
                active_agents = {uav.idx for uav in env.uavs if uav.alive}

        assignments = self._rebuild_assignments_from_winners(env, global_z, global_y)
        return {
            "assignments": assignments,
            "iterations": iterations,
            "winners": global_z,
            "winning_bids": global_y,
            "clusters": clusters,
        }


class CBBAEnv:
    """
    与你现有框架兼容的改进版 CBBA 环境。
    main 中把
        from task_allocation import CBBAEnv, format_episode_metrics
    改成
        from CBBA import CBBAEnv, format_episode_metrics
    就可以运行。
    """

    def __init__(
        self,
        uavs: List[Uav],
        targets: List[Target],
        tasks: Optional[List[Task]] = None,
        map_size: float = 100.0,
        max_bundle_size: Optional[int] = None,
        max_consensus_rounds: int = 50,
        cluster_count: Optional[int] = None,
        debug: bool = False,
    ):
        self.init_uavs = copy.deepcopy(uavs)
        self.init_targets = copy.deepcopy(targets)
        self.map_size = map_size
        self.debug = debug
        self.solver = ImprovedCBBASolver(
            max_bundle_size=max_bundle_size,
            max_consensus_rounds=max_consensus_rounds,
            cluster_count=cluster_count,
            debug=debug,
        )
        self.reset()

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
        self.event_logs: List[str] = []
        return self

    def _log(self, *args):
        if self.debug:
            print(*args)
    def _emit_event_log(self, message: str):
        self.event_logs.append(message)
        print(message)

    def _new_blank_instance(self, cls):
        try:
            return cls.__new__(cls)
        except Exception:
            class _Blank:
                pass

            return _Blank()

    def _recompute_global_bounds(self):
        self.max_total_voyage, self.max_total_time = calculate_max_possible_voyage_time(
            self.uavs, self.targets
        )
        self.max_total_voyage = max(self.max_total_voyage, 1.0)
        self.max_total_time = max(self.max_total_time, 1.0)

    def _build_dynamic_target(
        self,
        target_prefix: str,
        location: Tuple[float, float],
        strike_ammunition: int = 3,
        recon_time: float = 5.0,
        assessment_time: float = 4.0,
    ) -> Target:
        base_prefix = target_prefix
        suffix = 1
        while (
            f"{target_prefix}S" in self.task_dict
            or f"{target_prefix}R" in self.task_dict
            or f"{target_prefix}A" in self.task_dict
        ):
            target_prefix = f"{base_prefix}_{suffix}"
            suffix += 1

        target = self._new_blank_instance(Target)
        target.id = target_prefix
        target.location = location
        target.total_time = 0.0
        target.tasks = []

        def make_task(task_id: str, task_type: int, ammunition: int, time_cost: float):
            task = self._new_blank_instance(Task)
            task.id = task_id
            task.type = task_type
            task.location = location
            task.ammunition = ammunition
            task.time = time_cost
            task.flag = False
            task.waiting_time = 0.0
            task.end_time = 0.0
            task.target = target
            task.strike = 1.0 if task_type == 1 else 0.0
            task.reconnaissance = 1.0 if task_type == 2 else 0.0
            task.assessment = 1.0 if task_type == 3 else 0.0
            return task

        strike_task = make_task(f"{target_prefix}S", 1, strike_ammunition, 0.0)
        recon_task = make_task(f"{target_prefix}R", 2, 0, recon_time)
        assess_task = make_task(f"{target_prefix}A", 3, 0, assessment_time)
        target.tasks = [strike_task, recon_task, assess_task]
        return target

    def add_dynamic_target(
        self,
        target_prefix: str = "TASKDYN01",
        location: Tuple[float, float] = (38.4, 81.2),
        strike_ammunition: int = 3,
        recon_time: float = 5.0,
        assessment_time: float = 4.0,
    ) -> Target:
        target = self._build_dynamic_target(
            target_prefix=target_prefix,
            location=location,
            strike_ammunition=strike_ammunition,
            recon_time=recon_time,
            assessment_time=assessment_time,
        )
        self.targets.append(target)
        self.tasks.extend(target.tasks)
        for task in target.tasks:
            self.task_dict[task.id] = task
        self._recompute_global_bounds()
        self._emit_event_log(
            f"[动态事件-新增目标] 新增目标 {target.id}，位置={target.location}，新增任务={[task.id for task in target.tasks]}"
        )
        return target

    def _select_crash_uav(self, crash_type: int = 2) -> Optional[Uav]:
        candidates = [uav for uav in self.uavs if uav.alive and getattr(uav, "type", None) == crash_type]
        if not candidates:
            return None
        candidates.sort(key=lambda u: (u.idx, u.id))
        return candidates[0]

    def _collect_competition_snapshot(self, tasks: List[Task]) -> Dict[str, List[dict]]:
        snapshot: Dict[str, List[dict]] = {}
        for task in tasks:
            candidates = []
            for uav in self.uavs:
                if not uav.alive:
                    continue
                if not self._static_feasible(uav, task):
                    continue
                gain, pos, delay = self.best_insertion_delay_first(uav, [], task.id)
                if pos < 0 or gain <= EPS:
                    continue
                candidates.append(
                    {
                        "uav_idx": uav.idx,
                        "uav_id": uav.id,
                        "uav_type": uav.type,
                        "gain": gain,
                        "delay": delay,
                        "insert_pos": pos,
                    }
                )
            candidates.sort(key=lambda x: (-x["gain"], x["delay"], x["uav_idx"]))
            snapshot[task.id] = candidates
        return snapshot

    def _winner_reason(self, winner_idx: Optional[int], winner_bid: float, candidates: List[dict]) -> str:
        if winner_idx is None:
            return "没有可行赢家，当前无 UAV 能满足类型/资源/航程约束。"
        winner = next((c for c in candidates if c["uav_idx"] == winner_idx), None)
        if winner is None:
            return f"该 UAV 在一致性阶段保持全局最优 winning bid={winner_bid:.4f}。"
        if len(candidates) == 1:
            return "它是唯一满足任务类型、资源与航程约束的可行竞标者。"
        runner_up = next((c for c in candidates if c["uav_idx"] != winner_idx), None)
        if runner_up is None:
            return "它是唯一有效竞标者。"
        if winner["gain"] > runner_up["gain"] + EPS:
            return (
                f"它的边际收益最高（{winner['gain']:.4f}），高于次优 {runner_up['uav_id']} "
                f"的 {runner_up['gain']:.4f}。"
            )
        if winner["delay"] < runner_up["delay"] - EPS:
            return (
                f"它与其他 UAV 收益接近，但插入延迟更小（{winner['delay']:.4f} < "
                f"{runner_up['delay']:.4f}）。"
            )
        return (
            f"它与其他 UAV 收益接近，按 CBBA 的 tie-break 规则由编号更小的 {winner['uav_id']} 获胜。"
        )

    def _print_competition_report(self, header: str, tasks: List[Task], snapshot: Dict[str, List[dict]], wave_result: dict):
        self._emit_event_log(header)
        for task in tasks:
            candidates = snapshot.get(task.id, [])
            self._emit_event_log(f"  任务 {task.id} (type={task.type}, location={task.location})")
            if not candidates:
                self._emit_event_log("    可行竞争 UAV: 无")
            else:
                for cand in candidates:
                    self._emit_event_log(
                        "    候选 -> "
                        f"{cand['uav_id']} (type={cand['uav_type']}), "
                        f"gain={cand['gain']:.4f}, delay={cand['delay']:.4f}, pos={cand['insert_pos']}"
                    )
            winner_idx = wave_result["winners"].get(task.id)
            winner_bid = wave_result["winning_bids"].get(task.id, NEG_INF)
            if winner_idx is None:
                self._emit_event_log("    最终赢家: 无")
                self._emit_event_log("    原因: 没有 UAV 成功获得该任务。")
            else:
                winner_uav = self.uavs[winner_idx]
                reason = self._winner_reason(winner_idx, winner_bid, candidates)
                self._emit_event_log(
                    f"    最终赢家: {winner_uav.id} (type={winner_uav.type}), winning_bid={winner_bid:.4f}"
                )
                self._emit_event_log(f"    原因: {reason}")

    def _merge_assignments(self, base_assignments: Dict[int, List[Task]], extra_assignments: Dict[int, List[Task]]) -> Dict[int, List[Task]]:
        merged: Dict[int, List[Task]] = {idx: list(tasks) for idx, tasks in base_assignments.items()}
        for uav_idx, task_list in extra_assignments.items():
            merged.setdefault(uav_idx, [])
            existing_ids = {task.id for task in merged[uav_idx]}
            for task in task_list:
                if task.id not in existing_ids:
                    merged[uav_idx].append(task)
                    existing_ids.add(task.id)
        return merged

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
        if cand_bid > best_bid + EPS:
            return True
        if abs(cand_bid - best_bid) <= EPS:
            if best_uav_idx is None:
                return True
            return cand_uav_idx < best_uav_idx
        return False

    # =========================
    # 创新点 1：统一异构约束的可行域掩码
    # =========================
    def _task_is_released(self, task: Task) -> bool:
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

    def _can_type_execute(self, uav: Uav, task: Task) -> bool:
        # 1: 打击型，只能做打击
        # 2: 侦察评估型，可做侦察和评估
        # 3: 通用型，可做全部任务
        if task.type == 1:
            return uav.type in (1, 3)
        if task.type == 2:
            return uav.type in (2, 3)
        if task.type == 3:
            return uav.type in (2, 3)
        return False

    def _task_requirement(self, task: Task) -> float:
        if task.type == 1:
            return max(task.strike, 0.0)
        if task.type == 2:
            return max(task.reconnaissance, 0.0)
        if task.type == 3:
            return max(task.assessment, 0.0)
        return 0.0

    def _uav_capacity(self, uav: Uav, task_type: int) -> float:
        if task_type == 1:
            return max(uav.strike, 0.0)
        if task_type == 2:
            return max(uav.reconnaissance, 0.0)
        if task_type == 3:
            return max(uav.assessment, 0.0)
        return 0.0

    def _capability_ratio(self, uav: Uav, task: Task) -> float:
        req = self._task_requirement(task)
        if req <= EPS:
            return 1.0
        cap = self._uav_capacity(uav, task.type)
        return min(cap / req, 1.0)

    def _static_feasible(self, uav: Uav, task: Task) -> bool:
        if not uav.alive:
            return False
        if task.flag:
            return False
        if not self._task_is_released(task):
            return False
        if not self._can_type_execute(uav, task):
            return False
        # if self._capability_ratio(uav, task) <= EPS:
        #     return False
        if task.type == 1 and task.ammunition > uav.ammunition + EPS:
            return False
        if task.type in (2, 3) and task.time > uav.time + EPS:
            return False
        if calculate_voyage_distance(uav, task) > uav.voyage + EPS:
            return False
        return True

    def build_feasibility_mask(
        self,
        uav: Uav,
        current_path: List[str],
        available_tasks: List[Task],
        excluded_task_ids: Optional[Set[str]] = None,
    ) -> Dict[str, bool]:
        excluded_task_ids = excluded_task_ids or set()
        mask: Dict[str, bool] = {}
        for task in available_tasks:
            if task.id in excluded_task_ids:
                mask[task.id] = False
                continue
            if not self._static_feasible(uav, task):
                mask[task.id] = False
                continue
            gain, pos, _ = self.best_insertion_delay_first(uav, current_path, task.id)
            mask[task.id] = pos >= 0 and gain > EPS
        return mask

    # =========================
    # 创新点 2：最小延迟优先、最大收益择优插入
    # =========================
    def _copy_uav_state(self, uav: Uav) -> Uav:
        return copy.deepcopy(uav)

    def _calculate_reward(
        self, uav: Uav, task: Task, waiting_time: float = 0.0
    ) -> float:
        if not self._static_feasible(uav, task):
            return -1.0

        fit_r = 0
        voyage_r = 1.0 - min(
            calculate_voyage_distance(uav, task) / self.max_total_voyage, 1.0
        )
        time_r = 1.0 - min(calculate_voyage_time(uav, task) / self.max_total_time, 1.0)
        delay_r = 1.0 - min(waiting_time / self.max_total_time, 1.0)

        alpha = 0.0
        beta = 0.50
        gamma = 0.50
        eta = 0.00
        return alpha * fit_r + beta * voyage_r + gamma * time_r + eta * delay_r

    def _evaluate_path_stats(self, uav: Uav, path_ids: List[str]) -> dict:
        sim_uav = self._copy_uav_state(uav)
        total_reward = 0.0
        total_waiting = 0.0

        for task_id in path_ids:
            task = self.task_dict[task_id]
            if not self._static_feasible(sim_uav, task):
                return {
                    "feasible": False,
                    "reward": NEG_INF,
                    "end_time": float("inf"),
                    "waiting": float("inf"),
                    "sim_uav": sim_uav,
                }

            dist = calculate_voyage_distance(sim_uav, task)
            travel_time = calculate_voyage_time(sim_uav, task)
            arrival_time = sim_uav.end_time + travel_time

            predecessor_finish = 0.0
            for prev_task in task.target.tasks:
                if prev_task.id == task.id:
                    break
                predecessor_finish = max(predecessor_finish, prev_task.end_time)

            start_time = max(arrival_time, predecessor_finish)
            waiting_time = max(0.0, start_time - arrival_time)
            finish_time = start_time

            reward = self._calculate_reward(sim_uav, task, waiting_time)
            if reward < 0:
                return {
                    "feasible": False,
                    "reward": NEG_INF,
                    "end_time": float("inf"),
                    "waiting": float("inf"),
                    "sim_uav": sim_uav,
                }

            total_reward += reward
            total_waiting += waiting_time

            sim_uav.voyage -= dist
            sim_uav.location = task.location
            sim_uav.ammunition -= task.ammunition
            sim_uav.time -= task.time
            sim_uav.task_nums += 1
            sim_uav.end_time = finish_time

        return {
            "feasible": True,
            "reward": total_reward,
            "end_time": sim_uav.end_time,
            "waiting": total_waiting,
            "sim_uav": sim_uav,
        }

    def best_insertion_delay_first(
        self,
        uav: Uav,
        current_path: List[str],
        candidate_task_id: str,
    ) -> Tuple[float, int, float]:
        base_stats = self._evaluate_path_stats(uav, current_path)
        if not base_stats["feasible"]:
            return NEG_INF, -1, float("inf")

        best_gain = NEG_INF
        best_pos = -1
        best_delay = float("inf")

        for pos in range(len(current_path) + 1):
            new_path = current_path[:pos] + [candidate_task_id] + current_path[pos:]
            new_stats = self._evaluate_path_stats(uav, new_path)
            if not new_stats["feasible"]:
                continue

            delay_inc = new_stats["end_time"] - base_stats["end_time"]
            reward_gain = new_stats["reward"] - base_stats["reward"]

            if delay_inc < best_delay - EPS:
                best_delay = delay_inc
                best_gain = reward_gain
                best_pos = pos
            elif abs(delay_inc - best_delay) <= EPS:
                if reward_gain > best_gain + EPS:
                    best_gain = reward_gain
                    best_pos = pos
                elif abs(reward_gain - best_gain) <= EPS and (
                    best_pos == -1 or pos < best_pos
                ):
                    best_pos = pos

        return best_gain, best_pos, best_delay

    # =========================
    # 执行与统计
    # =========================
    def _apply_transition_real(self, uav: Uav, task: Task):
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
            # print(
            #     f"Target {task.target.id} completed at time {task.target.total_time:.2f}"
            # )

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
                if not self._static_feasible(uav, task):
                    continue

                reward = self._calculate_reward(uav, task, waiting_time=0.0)
                if reward < 0:
                    continue

                fit = 0
                self._apply_transition_real(uav, task)

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

    def _finalize_episode_result(self) -> dict:
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
            "event_logs": self.event_logs,
        }

    def run_episode(self) -> dict:
        self.reset()
        while True:
            available_tasks = self._available_tasks()
            if not available_tasks:
                break

            wave_result = self.solver.solve_current_wave(self, available_tasks)
            self.total_cbba_iters += wave_result["iterations"]
            self.total_replan_rounds += 1

            execute_result = self.execute_assignments(wave_result["assignments"])
            self.round_history.append(
                {
                    "round": self.total_replan_rounds,
                    "available_tasks": [task.id for task in available_tasks],
                    "iterations": wave_result["iterations"],
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

        return self._finalize_episode_result()

    def run_episode_with_dynamic_events(
        self,
        target_add_round: int = 2,
        crash_round: int = 3,
        target_prefix: str = "TASKDYN01",
        target_location: Tuple[float, float] = (38.4, 81.2),
        strike_ammunition: int = 3,
        recon_time: float = 5.0,
        assessment_time: float = 4.0,
        crash_type: int = 2,
    ) -> dict:
        self.reset()
        target_added = False
        crash_done = False
        dynamic_task_ids: Set[str] = set()
        reported_dynamic_tasks: Set[str] = set()

        while True:
            next_round = self.total_replan_rounds + 1
            if (not target_added) and target_add_round is not None and next_round == target_add_round:
                target = self.add_dynamic_target(
                    target_prefix=target_prefix,
                    location=target_location,
                    strike_ammunition=strike_ammunition,
                    recon_time=recon_time,
                    assessment_time=assessment_time,
                )
                dynamic_task_ids = {task.id for task in target.tasks}
                target_added = True

            available_tasks = self._available_tasks()
            if not available_tasks:
                break

            newly_released_dynamic_tasks = [
                task
                for task in available_tasks
                if task.id in dynamic_task_ids and task.id not in reported_dynamic_tasks
            ]
            dynamic_snapshot = self._collect_competition_snapshot(newly_released_dynamic_tasks)

            wave_result = self.solver.solve_current_wave(self, available_tasks)
            self.total_cbba_iters += wave_result["iterations"]
            self.total_replan_rounds += 1

            if newly_released_dynamic_tasks:
                self._print_competition_report(
                    header=f"[动态事件-新增目标] 第 {self.total_replan_rounds} 轮中新增任务竞争情况",
                    tasks=newly_released_dynamic_tasks,
                    snapshot=dynamic_snapshot,
                    wave_result=wave_result,
                )
                reported_dynamic_tasks.update(task.id for task in newly_released_dynamic_tasks)

            assignments = {idx: list(tasks) for idx, tasks in wave_result["assignments"].items()}

            if (not crash_done) and crash_round is not None and self.total_replan_rounds == crash_round:
                crash_uav = self._select_crash_uav(crash_type=crash_type)
                if crash_uav is not None:
                    released_tasks = list(assignments.pop(crash_uav.idx, []))
                    crash_uav.alive = False
                    self._emit_event_log(
                        f"[动态事件-UAV坠毁] 第 {self.total_replan_rounds} 轮触发，坠毁 UAV={crash_uav.id} (type={crash_uav.type})"
                    )
                    self._emit_event_log(
                        f"  该 UAV 剩余未完成任务: {[task.id for task in released_tasks] if released_tasks else '无'}"
                    )
                    if released_tasks:
                        crash_snapshot = self._collect_competition_snapshot(released_tasks)
                        recovery_result = self.solver.solve_current_wave(self, released_tasks)
                        self.total_cbba_iters += recovery_result["iterations"]
                        self._print_competition_report(
                            header="[动态事件-UAV坠毁] 释放任务的二次竞争与重分配结果",
                            tasks=released_tasks,
                            snapshot=crash_snapshot,
                            wave_result=recovery_result,
                        )
                        assignments = self._merge_assignments(assignments, recovery_result["assignments"])
                crash_done = True

            execute_result = self.execute_assignments(assignments)
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

        result = self._finalize_episode_result()
        self._emit_event_log(
            f"[动态实验结束] 任务完成率={result['success_rate']:.4f} ({result['success_count']}/{result['tasks_num']})"
        )
        return result


def format_episode_metrics(ep: int, result: dict) -> str:
    return (
        f"Ep {ep} | Success: {result['success_rate']:.2f} "
        f"({result['success_count']}/{result['tasks_num']}) | "
        f"distance: {result['total_distance']:.2f}, "
        f"time: {result['total_time']:.2f} | "
        f"rounds: {result['replan_rounds']}, cbba_iters: {result['total_cbba_iters']}"
    )
