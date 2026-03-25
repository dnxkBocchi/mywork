import argparse
import os

from env import load_different_scale_csv
from calculate import log_all_voyage_time
from plt.plot_path import plot_overall_result, plot_task_type_subfigures

# from cbba_pro import CBBAEnv, format_episode_metrics
from cbba_pro_dynamic import CBBAEnv, format_episode_metrics

# todo:
# 3.加新增目标和坠毁的无人机处理
# 5.再找一个指标能表明分布式算法的痛点,并且我的最好
# 7.把10*10的数据放到论文里
# 相关研究扯上负载均衡
# 匹配度改名

# my
# UAVS1: tasks=['TASK06S', 'TASK08S']
# UAVS2: tasks=['TASK01S']
# UAVRA1: tasks=['TASK19R', 'TASK13R', 'TASK29A', 'TASK27A']
# UAVRA2: tasks=['TASK15R', 'TASK12R', 'TASK22A']
# UAVG1: tasks=['TASK17R', 'TASK07S', 'TASK09S', 'TASK03S', 'TASK23A']
# UAVS3: tasks=['TASK10S', 'TASK04S']
# UAVRA3: tasks=['TASK11R', 'TASK21A']
# UAVRA4: tasks=['TASK16R', 'TASK18R', 'TASK28A', 'TASK26A']
# UAVRA5: tasks=['TASK20R', 'TASK14R', 'TASK24A', 'TASK30A']
# UAVG2: tasks=['TASK05S', 'TASK02S', 'TASK25A']


# UAVS1: tasks=['TASK06S', 'TASK08S']
# UAVS2: tasks=['TASK01S']
# UAVRA1: tasks=['TASK19R', 'TASK13R', 'TASK23A']
# UAVRA2: tasks=['TASK15R', 'TASK12R', 'TASK22A']
# UAVG1: tasks=['TASK17R', 'TASK07S', 'TASK09S', 'TASK03S', 'TASK29A', 'TASK27A']
# UAVS3: tasks=['TASK10S', 'TASK04S']
# UAVRA3: tasks=['TASK11R', 'TASK21A']
# UAVRA4: tasks=['TASK16R', 'TASK18R', 'TASK28A', 'TASK26A']
# UAVRA5: tasks=['TASK20R', 'TASK14R', 'TASK24A', 'TASK30A']
# UAVG2: tasks=['TASK05S', 'TASK02S', 'TASK25A']
def parse_args():
    parser = argparse.ArgumentParser(description="Run standard CBBA on local csv data.")
    parser.add_argument("--uav_csv", type=str, default="data/test/uav.csv")
    parser.add_argument("--task_csv", type=str, default="data/test/task.csv")
    parser.add_argument("--scale", type=int, default=10)
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--print_detail", type=bool, default=False)
    parser.add_argument("--save_dir", type=str, default="outputs")
    parser.add_argument("--dpi", type=int, default=200)
    parser.add_argument("--plot", type=bool, default=True)
    parser.add_argument(
        "--max_consensus_rounds",
        type=int,
        default=50,
        help="单轮 CBBA 最大一致性迭代次数",
    )
    parser.add_argument(
        "--max_bundle_size",
        type=int,
        default=None,
        help="每架无人机 bundle 最大长度，默认不限制",
    )
    return parser.parse_args()


def print_round_history(round_history):
    for item in round_history:
        print(f"\n[Round {item['round']}]")
        print("available_tasks:", item["available_tasks"])
        print("iterations:", item["iterations"])
        print("winners:", item["winners"])
        if item["executed"]:
            print("executed:")
            for uav_id, task_id, reward, fit in item["executed"]:
                print(
                    f"  {uav_id} -> {task_id} | reward={reward:.3f} | fitness={fit:.3f}"
                )
        else:
            print("executed: []")


def print_metrics(env):
    uavs = env.uavs
    targets = env.targets
    for uav in uavs:
        print(f"{uav.id}: tasks={uav.tasks}")
    for uav in uavs:
        distance = uav._init_voyage - uav.voyage
        ammunition = uav._init_ammunition - uav.ammunition
        time = uav._init_time - uav.time
        print(
            f"{uav.id}: voyage={distance:.2f}, ammunition={ammunition:.2f}, time={time:.2f}"
        )
    for target in targets:
        print(f"{target.id}: finish_time={target.total_time:.2f}")


def run_once(ep: int, args):
    uavs, tasks, targets = load_different_scale_csv(
        args.uav_csv,
        args.task_csv,
        size=args.scale,
    )

    env = CBBAEnv(
        uavs=uavs,
        targets=targets,
        tasks=tasks,
        max_bundle_size=args.max_bundle_size,
        max_consensus_rounds=args.max_consensus_rounds,
        debug=False,
    )
    # result = env.run_episode()
    result = env.run_episode_with_dynamic_events(
        target_add_round=2,  # 第2轮触发新增目标
        crash_round=3,  # 第3轮触发无人机坠毁
        target_prefix="TASKDYN01",
        target_location=(38.4, 81.2),
        strike_ammunition=3,
        recon_time=5,
        assessment_time=4,
        crash_type=2,  # 坠毁 type=2 的无人机
    )
    print(format_episode_metrics(1, result))

    # print(format_episode_metrics(ep, result))
    print_metrics(env)
    # log_all_voyage_time(env.uavs, env.targets)

    if result["unassigned_tasks"]:
        print("unassigned_tasks:", result["unassigned_tasks"])
    else:
        print("unassigned_tasks: []")

    if args.print_detail:
        print_round_history(result["round_history"])

    if args.plot:
        os.makedirs(args.save_dir, exist_ok=True)

        # overall_path = os.path.join(args.save_dir, f"cbba_pro_allocation.pdf")
        by_type_path = os.path.join(args.save_dir, f"cbba_pro_uav_path_dynamic.pdf")

        # plot_overall_result(env, result, overall_path, dpi=args.dpi)
        plot_task_type_subfigures(env, result, by_type_path, dpi=args.dpi)

        print("\nSaved figures:")
        # print(overall_path)
        print(by_type_path)

    return result


def main():
    args = parse_args()

    print("=" * 40)
    print("CBBA_pro Test Entry")
    print(f"scale   : {args.scale}")

    for ep in range(1, args.episodes + 1):
        run_once(ep, args)


if __name__ == "__main__":
    main()
