import matplotlib.pyplot as plt
import sys
import os

# 路径配置（确保文件读取和模块导入正常）
current_script_dir = os.path.dirname(os.path.abspath(__file__))
plt_dir = current_script_dir
my_dir = os.path.dirname(plt_dir)
sys.path.append(my_dir)

# 全局字体配置（符合期刊学术规范）
plt.rcParams["font.family"] = ["Times New Roman", "serif"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["font.size"] = 10


def get_axis_bounds(points, margin_ratio=0.08, min_span=8.0):
    """
    根据所有点的分布自适应生成坐标轴范围。
    - 不再强制从 0 开始
    - 自动留一点边距，避免点贴边
    - 当点非常集中时，保证最小显示范围，避免图被压扁
    """
    if not points:
        return (0.0, 100.0), (0.0, 100.0)

    xs = [p[0] for p in points]
    ys = [p[1] for p in points]

    xmin, xmax = min(xs), max(xs)
    ymin, ymax = min(ys), max(ys)

    dx = max(min_span, xmax - xmin)
    dy = max(min_span, ymax - ymin)

    mx = dx * margin_ratio
    my = dy * margin_ratio

    x_center = (xmin + xmax) / 2.0
    y_center = (ymin + ymax) / 2.0

    half_x = dx / 2.0 + mx
    half_y = dy / 2.0 + my

    return (x_center - half_x, x_center + half_x), (
        y_center - half_y,
        y_center + half_y,
    )


# 1. 数据文件路径处理（增加容错校验）
uav_csv = os.path.normpath(
    os.path.join(current_script_dir, "..", "data", "test", "uav.csv")
)
task_csv = os.path.normpath(
    os.path.join(current_script_dir, "..", "data", "test", "task.csv")
)

if not os.path.exists(uav_csv):
    print(f"错误：无人机数据文件不存在 → {uav_csv}")
    sys.exit(1)
if not os.path.exists(task_csv):
    print(f"错误：任务数据文件不存在 → {task_csv}")
    sys.exit(1)

# 2. 加载数据
try:
    from env import load_different_scale_csv

    uavs, tasks, targets = load_different_scale_csv(uav_csv, task_csv, 10)
except ImportError:
    print("错误：无法导入 env 模块，请检查 my 目录是否已添加到搜索路径")
    sys.exit(1)
except Exception as e:
    print(f"错误：数据加载失败 → {e}")
    sys.exit(1)

# 3. 提取坐标数据
try:
    uav_locations = [uav.location for uav in uavs]
    recon_uav = uav_locations[:3]
    strike_uav = uav_locations[3:6]
    eval_uav = uav_locations[6:10]
    target_locations = [target.location for target in targets]
except AttributeError as e:
    print(f"错误：数据对象属性异常（可能是 env 模块返回格式变化）→ {e}")
    sys.exit(1)

# 4. 绘图
plt.figure(figsize=(6, 6))

# 4.1 目标
if target_locations:
    target_x, target_y = zip(*target_locations)
    plt.scatter(
        target_x,
        target_y,
        color="black",
        s=100,
        marker="o",
        edgecolor="black",
        linewidth=1.2,
        label="Target",
    )

# 4.2 侦察型无人机
if recon_uav:
    recon_x, recon_y = zip(*recon_uav)
    plt.scatter(
        recon_x,
        recon_y,
        color="none",
        s=120,
        marker="^",
        edgecolor="black",
        linewidth=1.5,
        label="Recon UAV",
    )

# 4.3 打击型无人机
if strike_uav:
    strike_x, strike_y = zip(*strike_uav)
    plt.scatter(
        strike_x,
        strike_y,
        color="black",
        s=120,
        marker="s",
        edgecolor="black",
        linewidth=1.5,
        label="Strike UAV",
    )

# 4.4 评估型无人机
if eval_uav:
    eval_x, eval_y = zip(*eval_uav)
    plt.scatter(
        eval_x,
        eval_y,
        color="none",
        s=120,
        marker="D",
        edgecolor="black",
        linewidth=1.5,
        label="Eval UAV",
    )

# 5. 自适应坐标轴范围
all_points = []
all_points.extend(uav_locations)
all_points.extend(target_locations)
(xmin, xmax), (ymin, ymax) = get_axis_bounds(
    all_points, margin_ratio=0.08, min_span=8.0
)
plt.xlim(xmin, xmax)
plt.ylim(ymin, ymax)

# 如果你还想让刻度更规整，可以打开下面两行：
# plt.locator_params(axis='x', nbins=6)
# plt.locator_params(axis='y', nbins=6)

plt.xlabel("X Coordinate", fontsize=12, labelpad=10)
plt.ylabel("Y Coordinate", fontsize=12, labelpad=10)
plt.title(
    "Distribution of Task Targets and UAV Positions",
    fontsize=14,
    pad=20,
    fontweight="bold",
)
plt.legend(fontsize=11, loc="best", frameon=True, framealpha=0.8)
plt.grid(True, linestyle="--", alpha=0.6, color="black")
plt.gca().set_aspect("equal", adjustable="box")

# 6. 保存与显示
save_dir = os.path.join(my_dir, "outputs")
os.makedirs(save_dir, exist_ok=True)
save_path = os.path.join(save_dir, "uav_distribution.pdf")

plt.tight_layout()
plt.savefig(save_path, dpi=300, bbox_inches="tight", format="pdf")
print(f"自适应坐标图已保存 → {save_path}")

try:
    plt.show()
except Exception:
    print("提示：当前环境不支持显示图表（仅保存文件）")
