import re
import ast
import matplotlib.pyplot as plt
from matplotlib.patches import Patch


def _task_color(task_id: str):
    """按任务类型返回颜色：S-打击，R-侦察，A-评估"""
    if task_id.endswith("S"):
        return "#d62728"  # 红
    if task_id.endswith("R"):
        return "#1f77b4"  # 蓝
    if task_id.endswith("A"):
        return "#2ca02c"  # 绿
    return "#7f7f7f"


def _parse_dynamic_logs(result: dict):
    """
    从 result['event_logs'] 和 result['round_history'] 中解析：
    1. 新增目标信息
    2. 新增目标任务及赢家
    3. UAV坠毁信息
    4. 坠毁UAV已完成/未完成任务
    5. 未完成任务的重分配赢家
    6. 各受影响任务的完成轮次
    """
    event_logs = result.get("event_logs", [])
    round_history = result.get("round_history", [])

    add_round = None
    add_target_id = None
    add_location = None
    add_tasks = []

    crash_round = None
    crash_uav = None
    crash_unfinished_tasks = []

    current_round = None
    current_task = None

    # 受动态事件影响任务：release_round / winner
    task_release_round = {}
    task_winner = {}

    for line in event_logs:
        # 解析新增目标总体信息
        # [动态事件-新增目标] 新增目标 TASKDYN01，位置=(38.4, 81.2)，新增任务=['TASKDYN01S', ...]
        m = re.search(
            r"\[动态事件-新增目标\]\s*新增目标\s+(\S+)，位置=\(([^)]+)\)，新增任务=(\[.*\])",
            line,
        )
        if m:
            add_target_id = m.group(1)
            add_location = f"({m.group(2)})"
            add_tasks = ast.literal_eval(m.group(3))
            continue

        # 解析新增目标竞争轮次
        m = re.search(
            r"\[动态事件-新增目标\]\s*第\s*(\d+)\s*轮中新增任务竞争情况", line
        )
        if m:
            current_round = int(m.group(1))
            if add_round is None:
                add_round = current_round
            current_task = None
            continue

        # 解析坠毁事件
        # [动态事件-UAV坠毁] 第 3 轮触发，坠毁 UAV=UAVRA1 (type=2)
        m = re.search(
            r"\[动态事件-UAV坠毁\]\s*第\s*(\d+)\s*轮触发，坠毁 UAV=(\S+)", line
        )
        if m:
            crash_round = int(m.group(1))
            crash_uav = m.group(2)
            current_round = crash_round
            current_task = None
            continue

        # 解析坠毁UAV剩余未完成任务
        # 该 UAV 剩余未完成任务: ['TASK29A', 'TASK27A']
        if "该 UAV 剩余未完成任务:" in line:
            try:
                crash_unfinished_tasks = ast.literal_eval(line.split(":", 1)[1].strip())
            except Exception:
                crash_unfinished_tasks = []
            continue

        # 解析具体任务
        # 任务 TASKDYN01S (type=1, location=(...))
        m = re.search(r"\s*任务\s+(\S+)\s+\(type=\d+,", line)
        if m:
            current_task = m.group(1)
            if current_round is not None and current_task not in task_release_round:
                task_release_round[current_task] = current_round
            continue

        # 解析最终赢家
        # 最终赢家: UAVS2 (type=1), winning_bid=0.9597
        m = re.search(r"\s*最终赢家:\s+(\S+)\s+\(type=\d+\)", line)
        if m and current_task is not None:
            task_winner[current_task] = m.group(1)
            continue

    # 解析各任务实际完成轮次，以及如果日志没拿到赢家，则从执行记录补
    task_finish_round = {}
    for item in round_history:
        r = item.get("round")
        for rec in item.get("executed", []):
            uav_id, task_id = rec[0], rec[1]
            if task_id not in task_finish_round:
                task_finish_round[task_id] = r
            if task_id not in task_winner:
                task_winner[task_id] = uav_id

    # 解析坠毁UAV在坠毁前已完成任务
    crash_completed_tasks = []
    if crash_uav is not None and crash_round is not None:
        for item in round_history:
            r = item.get("round")
            if r >= crash_round:
                continue
            for rec in item.get("executed", []):
                uav_id, task_id = rec[0], rec[1]
                if uav_id == crash_uav and task_id not in crash_completed_tasks:
                    crash_completed_tasks.append(task_id)

    # 受影响任务 = 新增任务 + 坠毁释放任务
    affected_tasks = []
    for t in add_tasks:
        if t not in affected_tasks:
            affected_tasks.append(t)
    for t in crash_unfinished_tasks:
        if t not in affected_tasks:
            affected_tasks.append(t)

    return {
        "add_round": add_round,
        "add_target_id": add_target_id,
        "add_location": add_location,
        "add_tasks": add_tasks,
        "crash_round": crash_round,
        "crash_uav": crash_uav,
        "crash_completed_tasks": crash_completed_tasks,
        "crash_unfinished_tasks": crash_unfinished_tasks,
        "task_release_round": task_release_round,
        "task_finish_round": task_finish_round,
        "task_winner": task_winner,
        "affected_tasks": affected_tasks,
    }


def plot_dynamic_response_timeline(result: dict, save_path=None, dpi=300):
    """
    只画一张图：
    - 标出新增目标事件
    - 标出UAV坠毁事件
    - 展示新增任务/释放任务的完成过程
    - 标明处理UAV
    - 标明坠毁UAV已完成与未完成任务
    """
    data = _parse_dynamic_logs(result)

    add_round = data["add_round"]
    add_target_id = data["add_target_id"]
    add_location = data["add_location"]
    add_tasks = data["add_tasks"]

    crash_round = data["crash_round"]
    crash_uav = data["crash_uav"]
    crash_completed_tasks = data["crash_completed_tasks"]
    crash_unfinished_tasks = data["crash_unfinished_tasks"]

    task_release_round = data["task_release_round"]
    task_finish_round = data["task_finish_round"]
    task_winner = data["task_winner"]
    affected_tasks = data["affected_tasks"]

    if not affected_tasks:
        raise ValueError(
            "没有解析到受动态事件影响的任务，请检查 result['event_logs'] 是否存在。"
        )

    # 任务显示顺序：新增任务在上，坠毁释放任务在下
    ordered_tasks = []
    for t in add_tasks:
        if t in affected_tasks and t not in ordered_tasks:
            ordered_tasks.append(t)
    for t in crash_unfinished_tasks:
        if t in affected_tasks and t not in ordered_tasks:
            ordered_tasks.append(t)
    for t in affected_tasks:
        if t not in ordered_tasks:
            ordered_tasks.append(t)

    # y轴位置：中间留一格空隙，区分“新增任务”和“坠毁释放任务”
    y_map = {}
    y = 0
    for t in ordered_tasks:
        y_map[t] = y
        y += 1
        if len(add_tasks) > 0 and y == len(add_tasks):
            y += 1  # 分组留空

    max_finish_round = max(task_finish_round.values()) if task_finish_round else 1
    max_round = max(max_finish_round, add_round or 1, crash_round or 1)

    fig, ax = plt.subplots(figsize=(11, 6.8))

    # 画任务时间线
    for task_id in ordered_tasks:
        y_pos = y_map[task_id]
        release_r = task_release_round.get(
            task_id, add_round if task_id in add_tasks else crash_round
        )
        finish_r = task_finish_round.get(task_id, max_round)

        # 从 release_r 到 finish_r 画条带
        width = max(finish_r - release_r, 0) + 0.75
        ax.barh(
            y=y_pos,
            width=width,
            left=release_r,
            height=0.55,
            color=_task_color(task_id),
            alpha=0.7,
            edgecolor="black",
        )

        # 条带终点标赢家
        winner = task_winner.get(task_id, "N/A")
        ax.scatter(finish_r + 0.38, y_pos, s=28, zorder=3, color="black")
        ax.text(finish_r + 0.48, y_pos, f"{winner}", va="center", fontsize=9)

        # 起点标 release
        ax.text(
            release_r - 0.05,
            y_pos + 0.28,
            f"R{release_r}",
            ha="right",
            va="center",
            fontsize=8,
        )

    # 分组文字
    if add_tasks:
        ax.text(
            0.62,
            (y_map[add_tasks[0]] + y_map[add_tasks[-1]]) / 2,
            "New Target Tasks",
            rotation=90,
            va="center",
            ha="center",
            fontsize=10,
        )
    if crash_unfinished_tasks:
        ax.text(
            0.62,
            (y_map[crash_unfinished_tasks[0]] + y_map[crash_unfinished_tasks[-1]]) / 2,
            "Released by Crash",
            rotation=90,
            va="center",
            ha="center",
            fontsize=10,
        )

    # 事件竖线
    if add_round is not None:
        ax.axvline(add_round, linestyle="--", linewidth=1.5, color="black")
    if crash_round is not None:
        ax.axvline(crash_round, linestyle="--", linewidth=1.5, color="black")

    # 新增目标说明框
    if add_round is not None:
        lines = [f"New Target @ R{add_round}"]
        if add_target_id is not None:
            lines.append(f"{add_target_id} {add_location}")
        for t in add_tasks:
            lines.append(f"{t} -> {task_winner.get(t, 'N/A')}")
        add_box = "\n".join(lines)

        ax.text(
            add_round + 2.15,
            max(y_map.values()) - 5,
            add_box,
            fontsize=9,
            va="top",
            bbox=dict(
                boxstyle="round,pad=0.35",
                facecolor="white",
                edgecolor="black",
                alpha=0.95,
            ),
        )

    # 坠毁事件说明框
    if crash_round is not None and crash_uav is not None:
        lines = [f"UAV Crash @ R{crash_round}", f"{crash_uav} crashed"]

        if crash_completed_tasks:
            lines.append("done: " + ", ".join(crash_completed_tasks))
        else:
            lines.append("done: none")

        if crash_unfinished_tasks:
            lines.append("undone: " + ", ".join(crash_unfinished_tasks))
            for t in crash_unfinished_tasks:
                lines.append(f"{t} -> {task_winner.get(t, 'N/A')}")
        else:
            lines.append("undone: none")

        crash_box = "\n".join(lines)

        ax.text(
            crash_round + 1.15,
            max(y_map.values()) + 1.6 - 2.5,
            crash_box,
            fontsize=9,
            va="top",
            bbox=dict(
                boxstyle="round,pad=0.35",
                facecolor="white",
                edgecolor="black",
                alpha=0.95,
            ),
        )

    # 坐标轴
    ax.set_yticks([y_map[t] for t in ordered_tasks])
    ax.set_yticklabels(ordered_tasks)
    ax.set_xlabel("Replanning Round")
    ax.set_ylabel("Affected Tasks")
    ax.set_title("Dynamic Event Response Timeline")
    ax.set_xlim(0.5, max_round + 2.6)
    ax.set_xticks(list(range(1, max_round + 1)))
    ax.set_ylim(-0.8, max(y_map.values()) + 2.2)
    ax.grid(True, axis="x", linestyle="--", alpha=0.35)
    ax.invert_yaxis()  # 新增任务放上面

    # 图例
    handles = [
        Patch(facecolor="#d62728", edgecolor="black", alpha=0.7, label="Strike Task"),
        Patch(facecolor="#1f77b4", edgecolor="black", alpha=0.7, label="Recon Task"),
        Patch(facecolor="#2ca02c", edgecolor="black", alpha=0.7, label="Assess Task"),
    ]
    ax.legend(handles=handles, loc="lower right")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=dpi, bbox_inches="tight")
    plt.show()
