import argparse
import os

from env import load_different_scale_csv
from calculate import log_all_voyage_time

# from task_allocation import CBBAEnv, format_episode_metrics

from cbba_pro import CBBAEnv, format_episode_metrics
from plt.plot import plot_overall_result, plot_task_type_subfigures


# todo:
# 3.加新增和坠毁的无人机处理


def parse_args():
    parser = argparse.ArgumentParser(description="Run standard CBBA on local csv data.")
    parser.add_argument("--uav_csv", type=str, default="data/test/uav.csv")
    parser.add_argument("--task_csv", type=str, default="data/test/task.csv")
    parser.add_argument("--scale", type=int, default=10)
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--print_detail", type=bool, default=False)
    parser.add_argument("--save_dir", type=str, default="outputs")
    parser.add_argument("--dpi", type=int, default=200)
    parser.add_argument("--plot", type=bool, default=False)
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
    result = env.run_episode()

    print(format_episode_metrics(ep, result))
    print(
        f"tasks_num: {result['tasks_num']} | success_count: {result['success_count']} | "
        f"fitness_count: {result['fitness_count']:.3f}"
    )

    if result["unassigned_tasks"]:
        print("unassigned_tasks:", result["unassigned_tasks"])
    else:
        print("unassigned_tasks: []")

    if args.print_detail:
        print_round_history(result["round_history"])

    if args.plot:
        os.makedirs(args.save_dir, exist_ok=True)

        overall_path = os.path.join(
            args.save_dir, f"cbba_allocation_overall_ep{ep}.png"
        )
        by_type_path = os.path.join(
            args.save_dir, f"cbba_allocation_by_task_type_ep{ep}.png"
        )

        plot_overall_result(env, result, overall_path, dpi=args.dpi)
        plot_task_type_subfigures(env, result, by_type_path, dpi=args.dpi)

        print("\nSaved figures:")
        print(overall_path)
        print(by_type_path)

    return result


def main():
    args = parse_args()

    print("=" * 80)
    print("CBBA Test Entry")
    print(f"scale   : {args.scale}")
    print(f"episodes: {args.episodes}")
    print("=" * 80)

    for ep in range(1, args.episodes + 1):
        run_once(ep, args)


if __name__ == "__main__":
    main()
