import numpy as np
import matplotlib.pyplot as plt

methods = {
    "AUCTION": {
        "voyage": [37.25, 12.15, 16.09, 24.60, 7.73, 31.42, 14.27, 25.33, 39.23, 57.88],
        "time": [9.51, 21.85, 15.91, 38.59, 28.40, 14.07, 5.94, 8.93, 19.64, 45.98],
        "tasks": [3, 1, 4, 4, 3, 3, 2, 4, 4, 2],
    },
    "CONTRACTNET": {
        "voyage": [
            13.98,
            12.15,
            33.21,
            15.43,
            41.88,
            39.21,
            14.27,
            25.33,
            48.74,
            41.97,
        ],
        "time": [9.51, 21.87, 5.96, 24.87, 27.98, 14.07, 19.54, 8.93, 37.02, 32.21],
        "tasks": [2, 1, 4, 3, 4, 3, 2, 4, 4, 3],
    },
    "CCBA_KMEANS": {
        "voyage": [
            13.98,
            12.15,
            10.13,
            15.43,
            57.14,
            24.01,
            14.27,
            25.33,
            39.23,
            41.97,
        ],
        "time": [9.51, 21.87, 24.95, 15.00, 27.98, 14.07, 43.95, 8.93, 29.54, 22.40],
        "tasks": [2, 1, 3, 3, 6, 2, 2, 4, 4, 3],
    },
    "CCBA_MY": {
        "voyage": [
            13.98,
            12.15,
            34.84,
            15.43,
            32.43,
            24.01,
            14.27,
            25.33,
            39.23,
            41.97,
        ],
        "time": [9.51, 21.87, 24.95, 15.00, 27.98, 14.07, 31.39, 8.93, 20.36, 22.40],
        "tasks": [2, 1, 4, 3, 5, 2, 2, 4, 4, 3],
    },
}

method_names = list(methods.keys())

# ===== 统计指标 =====
total_voyage = [np.sum(methods[m]["voyage"]) for m in method_names]
avg_time = [np.mean(methods[m]["time"]) for m in method_names]
max_time = [np.max(methods[m]["time"]) for m in method_names]
std_voyage = [np.std(methods[m]["voyage"]) for m in method_names]
std_time = [np.std(methods[m]["time"]) for m in method_names]
std_tasks = [np.std(methods[m]["tasks"]) for m in method_names]

# ===== 图2：平均完成时间 + 最大完成时间 =====
x = np.arange(len(method_names))
width = 0.35
plt.figure(figsize=(5, 5))
plt.bar(x - width / 2, avg_time, width, label="Average Time")
plt.bar(x + width / 2, max_time, width, label="Max Time")
plt.xticks(x, method_names)
plt.ylabel("Time")
plt.title("Comparison of Task Completion Time")
plt.legend()
plt.tight_layout()
plt.savefig("outputs/balance_time_ave.pdf", dpi=200)
plt.show()

# ===== 图3：均衡性（航程std + 时间std） =====
plt.figure(figsize=(5, 5))
plt.bar(x - width / 2, std_voyage, width, label="Voyage Std")
plt.bar(x + width / 2, std_time, width, label="Time Std")
plt.xticks(x, method_names)
plt.ylabel("Standard Deviation")
plt.title("Comparison of Stability / Balance")
plt.legend()
plt.tight_layout()
plt.savefig("outputs/balance_stability.pdf", dpi=200)
plt.show()

# ===== 图4：每架UAV的航程折线图 =====
plt.figure(figsize=(10, 5))
uav_idx = np.arange(1, 11)
for m in method_names:
    plt.plot(uav_idx, methods[m]["voyage"], marker="o", label=m)
plt.xlabel("UAV Index")
plt.ylabel("Voyage")
plt.title("Voyage Distribution Across UAVs")
plt.xticks(uav_idx)
plt.legend()
plt.tight_layout()
plt.savefig("outputs/balance_uav_voyage.pdf", dpi=200)
plt.show()

# ===== 图5：每个目标完成时间折线图 =====
plt.figure(figsize=(10, 5))
target_idx = np.arange(1, 11)
for m in method_names:
    plt.plot(target_idx, methods[m]["time"], marker="o", label=m)
plt.xlabel("Target Index")
plt.ylabel("Completion Time")
plt.title("Completion Time Distribution Across Targets")
plt.xticks(target_idx)
plt.legend()
plt.tight_layout()
plt.savefig("outputs/balance_target_time.pdf", dpi=200)
plt.show()

# ===== 输出汇总表 =====
print("\nSummary:")
for m in method_names:
    print(
        f"{m}: "
        f"total_voyage={np.sum(methods[m]['voyage']):.2f}, "
        f"avg_time={np.mean(methods[m]['time']):.2f}, "
        f"max_time={np.max(methods[m]['time']):.2f}, "
        f"voyage_std={np.std(methods[m]['voyage']):.2f}, "
        f"time_std={np.std(methods[m]['time']):.2f}, "
        f"task_std={np.std(methods[m]['tasks']):.2f}"
    )
