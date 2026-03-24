import matplotlib.pyplot as plt
from collections import defaultdict
import matplotlib.transforms as mtransforms
from matplotlib.text import Text

UAV_TYPE_NAME = {
    1: "Strike UAV",
    2: "Recon/Assess UAV",
    3: "General UAV",
}

UAV_TYPE_COLOR = {
    1: "tab:red",
    2: "tab:blue",
    3: "tab:green",
}

TASK_TYPE_NAME = {
    1: "Strike Task",
    2: "Recon Task",
    3: "Assess Task",
}

TASK_TYPE_MARKER = {
    1: "s",
    2: "^",
    3: "o",
}

TASK_TYPE_COLOR = {
    1: "tab:red",
    2: "tab:blue",
    3: "tab:green",
}


def build_route_from_history(result):
    """
    根据 round_history 中的 executed，重建每架 UAV 的真实执行顺序
    route_by_uav[uav_id] = [task_id1, task_id2, ...]
    """
    route_by_uav = defaultdict(list)
    for round_item in result["round_history"]:
        for uav_id, task_id, reward, fit in round_item["executed"]:
            route_by_uav[uav_id].append(task_id)
    return dict(route_by_uav)


def get_bounds(initial_uavs, targets, tasks, margin_ratio=0.08):
    xs = []
    ys = []

    for u in initial_uavs:
        xs.append(u.location[0])
        ys.append(u.location[1])

    for target in targets:
        xs.append(target.location[0])
        ys.append(target.location[1])

    for task in tasks:
        xs.append(task.location[0])
        ys.append(task.location[1])

    if not xs or not ys:
        return (0, 100), (0, 100)

    xmin, xmax = min(xs), max(xs)
    ymin, ymax = min(ys), max(ys)

    dx = max(1.0, xmax - xmin)
    dy = max(1.0, ymax - ymin)

    mx = dx * margin_ratio
    my = dy * margin_ratio

    return (xmin - mx, xmax + mx), (ymin - my, ymax + my)


def setup_axis(ax, initial_uavs, targets, tasks, title):
    (xmin, xmax), (ymin, ymax) = get_bounds(initial_uavs, targets, tasks)
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_title(title)
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.set_aspect("equal", adjustable="box")


def _bbox_overlaps(candidate_bbox, existing_bboxes, expand_px=2):
    expanded = mtransforms.Bbox.from_extents(
        candidate_bbox.x0 - expand_px,
        candidate_bbox.y0 - expand_px,
        candidate_bbox.x1 + expand_px,
        candidate_bbox.y1 + expand_px,
    )

    for old_bbox in existing_bboxes:
        old_expanded = mtransforms.Bbox.from_extents(
            old_bbox.x0 - expand_px,
            old_bbox.y0 - expand_px,
            old_bbox.x1 + expand_px,
            old_bbox.y1 + expand_px,
        )
        if expanded.overlaps(old_expanded):
            return True
    return False


def _build_probe_text(ax, text, color, fontsize, fontweight, zorder):
    """
    创建一个不真正加入坐标轴的 Text 探针对象，只用来测 bbox。
    这样比反复 ax.text(...)/remove() 更快。
    """
    probe = Text(
        x=0,
        y=0,
        text=str(text),
        fontsize=fontsize,
        color=color,
        fontweight=fontweight,
        zorder=zorder,
        ha="center",
        va="center",
        bbox=dict(
            boxstyle="round,pad=0.25",
            fc="white",
            ec="gray",
            alpha=0.85,
            linewidth=0.5,
        ),
    )
    probe.set_figure(ax.figure)
    probe.axes = ax
    probe.set_transform(ax.transData)
    return probe


def add_nonoverlap_text(
    ax,
    x,
    y,
    text,
    placed_bboxes,
    renderer,
    color="black",
    fontsize=8,
    fontweight=None,
    zorder=10,
):
    """
    更快版标签避让：
    1. 以上下排列为主，左右只做轻微偏移
    2. 使用真实文本框 bbox 判断是否重叠
    3. 不在候选循环里反复 draw，也不反复 add/remove artist
    """
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    xspan = xlim[1] - xlim[0]
    yspan = ylim[1] - ylim[0]

    dx = xspan * 0.006
    dy = yspan * 0.035

    candidates = [
        (0, 1.0 * dy),
        (0, -1.0 * dy),
        (0.5 * dx, 1.0 * dy),
        (-0.5 * dx, 1.0 * dy),
        (0.5 * dx, -1.0 * dy),
        (-0.5 * dx, -1.0 * dy),
        (0, 1.8 * dy),
        (0, -1.8 * dy),
        (0.8 * dx, 1.8 * dy),
        (-0.8 * dx, 1.8 * dy),
        (0.8 * dx, -1.8 * dy),
        (-0.8 * dx, -1.8 * dy),
        (0, 2.6 * dy),
        (0, -2.6 * dy),
        (1.0 * dx, 2.6 * dy),
        (-1.0 * dx, 2.6 * dy),
        (1.0 * dx, -2.6 * dy),
        (-1.0 * dx, -2.6 * dy),
        (0, 3.4 * dy),
        (0, -3.4 * dy),
    ]

    text_style = dict(
        fontsize=fontsize,
        color=color,
        fontweight=fontweight,
        zorder=zorder,
        ha="center",
        va="center",
        bbox=dict(
            boxstyle="round,pad=0.25",
            fc="white",
            ec="gray",
            alpha=0.85,
            linewidth=0.5,
        ),
    )

    probe = _build_probe_text(ax, text, color, fontsize, fontweight, zorder)

    best_pos = None
    best_bbox = None

    for ox, oy in candidates:
        tx, ty = x + ox, y + oy

        if tx < xlim[0] or tx > xlim[1] or ty < ylim[0] or ty > ylim[1]:
            continue

        probe.set_position((tx, ty))
        bbox = probe.get_window_extent(renderer=renderer)

        if not _bbox_overlaps(bbox, placed_bboxes):
            best_pos = (tx, ty)
            best_bbox = bbox
            break

    if best_pos is None:
        tx, ty = x, y + dy
        probe.set_position((tx, ty))
        best_pos = (tx, ty)
        best_bbox = probe.get_window_extent(renderer=renderer)

    artist = ax.text(best_pos[0], best_pos[1], str(text), **text_style)
    placed_bboxes.append(best_bbox)
    return artist


def plot_targets(ax, targets, placed_bboxes, renderer):
    """
    目标中心位置：金色星星
    """
    used_label = False
    for target in targets:
        x, y = target.location
        ax.scatter(
            x,
            y,
            s=220,
            marker="*",
            c="gold",
            edgecolors="black",
            linewidths=1.0,
            zorder=4,
            label="Target Center" if not used_label else None,
        )
        used_label = True

        # 如需显示 target id，可打开下面代码
        add_nonoverlap_text(
            ax,
            x,
            y,
            target.id,
            placed_bboxes,
            renderer,
            color="black",
            fontsize=8,
            zorder=5,
        )


def plot_tasks(ax, tasks, placed_bboxes, renderer, only_task_type=None):
    """
    任务点：按任务类型画不同 marker / 颜色
    """
    used_labels = set()

    for task in tasks:
        if only_task_type is not None and task.type != only_task_type:
            continue

        x, y = task.location
        label = TASK_TYPE_NAME[task.type]
        if label in used_labels:
            label = None
        else:
            used_labels.add(label)

        ax.scatter(
            x,
            y,
            s=90,
            marker=TASK_TYPE_MARKER[task.type],
            c=TASK_TYPE_COLOR[task.type],
            edgecolors="black",
            linewidths=0.8,
            alpha=0.9,
            zorder=3,
            label=label,
        )

        add_nonoverlap_text(
            ax,
            x,
            y,
            task.id,
            placed_bboxes,
            renderer,
            color="black",
            fontsize=8,
            zorder=4,
        )


def plot_uavs_and_routes(
    ax,
    initial_uavs,
    task_by_id,
    route_by_uav,
    placed_bboxes,
    renderer,
    only_task_type=None,
):
    """
    无人机起点 + 实际执行路线
    不同类型无人机用不同颜色
    """
    used_uav_type_label = set()

    for uav in initial_uavs:
        ux, uy = uav.location
        uav_type_label = UAV_TYPE_NAME.get(uav.type, f"Type {uav.type}")
        label = uav_type_label if uav_type_label not in used_uav_type_label else None
        used_uav_type_label.add(uav_type_label)

        ax.scatter(
            ux,
            uy,
            s=180,
            marker="P",
            c=UAV_TYPE_COLOR.get(uav.type, "black"),
            edgecolors="black",
            linewidths=1.0,
            zorder=6,
            label=label,
        )

        add_nonoverlap_text(
            ax,
            ux,
            uy,
            uav.id,
            placed_bboxes,
            renderer,
            color=UAV_TYPE_COLOR.get(uav.type, "black"),
            fontsize=9,
            fontweight="bold",
            zorder=7,
        )

        task_ids = route_by_uav.get(uav.id, [])
        if only_task_type is not None:
            task_ids = [
                tid for tid in task_ids if task_by_id[tid].type == only_task_type
            ]

        if not task_ids:
            continue

        points = [uav.location] + [task_by_id[tid].location for tid in task_ids]
        xs = [p[0] for p in points]
        ys = [p[1] for p in points]

        ax.plot(
            xs,
            ys,
            color=UAV_TYPE_COLOR.get(uav.type, "black"),
            linewidth=2.2,
            alpha=0.95,
            zorder=2,
        )

        for i in range(len(points) - 1):
            x1, y1 = points[i]
            x2, y2 = points[i + 1]
            ax.annotate(
                "",
                xy=(x2, y2),
                xytext=(x1, y1),
                arrowprops=dict(
                    arrowstyle="->",
                    color=UAV_TYPE_COLOR.get(uav.type, "black"),
                    lw=1.8,
                    alpha=0.9,
                ),
                zorder=2,
            )

        for step, tid in enumerate(task_ids, start=1):
            tx, ty = task_by_id[tid].location
            add_nonoverlap_text(
                ax,
                tx,
                ty,
                f"{uav.id}:{step}",
                placed_bboxes,
                renderer,
                color=UAV_TYPE_COLOR.get(uav.type, "black"),
                fontsize=8,
                zorder=7,
            )


def dedup_legend(ax):
    handles, labels = ax.get_legend_handles_labels()
    uniq = {}
    for h, l in zip(handles, labels):
        if l and l not in uniq:
            uniq[l] = h
    if uniq:
        ax.legend(uniq.values(), uniq.keys(), loc="best", fontsize=9)


def plot_overall_result(env, result, save_path, dpi=200):
    initial_uavs = env.init_uavs
    targets = env.init_targets
    tasks = env.tasks
    task_by_id = {task.id: task for task in tasks}
    route_by_uav = build_route_from_history(result)

    fig, ax = plt.subplots(figsize=(11, 8))
    setup_axis(ax, initial_uavs, targets, tasks, "CBBA UAV-Task Allocation Result")

    # 只做一次 draw，拿到 renderer，后面复用
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()

    placed_bboxes = []
    plot_targets(ax, targets, placed_bboxes, renderer)
    # plot_tasks(ax, tasks, placed_bboxes, renderer)
    plot_uavs_and_routes(
        ax, initial_uavs, task_by_id, route_by_uav, placed_bboxes, renderer
    )

    dedup_legend(ax)
    plt.tight_layout()
    plt.savefig(save_path, dpi=dpi, bbox_inches="tight")
    plt.show()


def plot_task_type_subfigures(env, result, save_path, dpi=200):
    initial_uavs = env.init_uavs
    targets = env.init_targets
    tasks = env.tasks
    task_by_id = {task.id: task for task in tasks}
    route_by_uav = build_route_from_history(result)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.6))

    for idx, task_type in enumerate([2, 1, 3]):
        ax = axes[idx]
        title = {
            2: "Recon Tasks",
            1: "Strike Tasks",
            3: "Assess Tasks",
        }[task_type]

        setup_axis(ax, initial_uavs, targets, tasks, title)

        # 每个子图各 draw 一次，拿自己的 renderer
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()

        placed_bboxes = []
        # plot_targets(ax, targets, placed_bboxes, renderer)
        plot_tasks(ax, tasks, placed_bboxes, renderer, only_task_type=task_type)
        plot_uavs_and_routes(
            ax,
            initial_uavs,
            task_by_id,
            route_by_uav,
            placed_bboxes,
            renderer,
            only_task_type=task_type,
        )

        dedup_legend(ax)

    plt.tight_layout()
    plt.savefig(save_path, dpi=dpi, bbox_inches="tight")
    plt.show()
