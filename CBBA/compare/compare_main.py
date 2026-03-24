import argparse
import os

import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from env import load_different_scale_csv
from task_allocation import CBBAEnv, format_episode_metrics
from cbba_basic import CBBASolver
from auction_algorithm import AuctionSolver
from contractnet_algorithm import ContractNetSolver


ALGORITHMS = ("cbba", "auction", "contractnet", "all")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compare CBBA / Auction / Contract Net on local csv data."
    )
    parser.add_argument(
        "--uav_csv", type=str, default="data/test/uav.csv", help="本地无人机 csv 路径"
    )
    parser.add_argument(
        "--task_csv", type=str, default="data/test/task.csv", help="本地任务 csv 路径"
    )
    parser.add_argument(
        "--scale", type=int, default=10, help="从 csv 中截取的目标规模 size"
    )
    parser.add_argument("--episodes", type=int, default=1, help="重复运行次数")
    parser.add_argument(
        "--algorithm", type=str, default="all", choices=ALGORITHMS, help="选择对比算法"
    )
    parser.add_argument(
        "--max_consensus_rounds", type=int, default=50, help="CBBA 最大一致性迭代次数"
    )
    parser.add_argument(
        "--max_bundle_size", type=int, default=None, help="CBBA bundle 最大长度"
    )
    parser.add_argument("--contract_passes", type=int, default=1, help="合同网公告轮数")
    parser.add_argument("--print_detail", action="store_true", help="打印每轮分配详情")
    return parser.parse_args()


def validate_paths(uav_csv: str, task_csv: str):
    if not os.path.exists(uav_csv):
        raise FileNotFoundError(f"找不到无人机文件: {uav_csv}")
    if not os.path.exists(task_csv):
        raise FileNotFoundError(f"找不到任务文件: {task_csv}")


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
        print(f"{uav.id}: voyage={distance:.2f}")
    for target in targets:
        print(f"{target.id}: finish_time={target.total_time:.2f}") 


def build_solver(args, name: str):
    if name == "cbba":
        return CBBASolver(
            max_bundle_size=args.max_bundle_size,
            max_consensus_rounds=args.max_consensus_rounds,
            debug=False,
        )
    if name == "auction":
        return AuctionSolver(debug=False)
    if name == "contractnet":
        return ContractNetSolver(max_passes=args.contract_passes, debug=False)
    raise ValueError(f"未知算法: {name}")


def run_once(ep: int, args, algo_name: str):
    uavs, tasks, targets = load_different_scale_csv(
        args.uav_csv, args.task_csv, size=args.scale
    )
    env = CBBAEnv(
        uavs=uavs,
        targets=targets,
        tasks=tasks,
        max_bundle_size=args.max_bundle_size,
        max_consensus_rounds=args.max_consensus_rounds,
        debug=False,
    )
    env.solver = build_solver(args, algo_name)
    result = env.run_episode()

    print(f"[{algo_name.upper()}] " + format_episode_metrics(ep, result))
    if algo_name == "cbba":
        print_metrics(env)
    
    if result["unassigned_tasks"]:
        print("unassigned_tasks:", result["unassigned_tasks"])
    else:
        print("unassigned_tasks: []")

    if args.print_detail:
        print_round_history(result["round_history"])

    return result


def summarize_results(all_results, algo_name: str):
    if not all_results:
        return
    avg_reward = sum(r["total_reward"] for r in all_results) / len(all_results)
    avg_success = sum(r["success_rate"] for r in all_results) / len(all_results)
    avg_fitness = sum(r["fitness_rate"] for r in all_results) / len(all_results)
    avg_distance = sum(r["total_distance"] for r in all_results) / len(all_results)
    avg_time = sum(r["total_time"] for r in all_results) / len(all_results)
    avg_rounds = sum(r["replan_rounds"] for r in all_results) / len(all_results)
    avg_iters = sum(r["total_cbba_iters"] for r in all_results) / len(all_results)

    print(
        f"[{algo_name.upper()} AVG] Avg Reward: {avg_reward:.3f} | Success: {avg_success:.2f} | "
        f"Fitness: {avg_fitness:.2f} | distance: {avg_distance:.2f}, time: {avg_time:.2f} | "
        f"rounds: {avg_rounds:.2f}, iters: {avg_iters:.2f}"
    )


def main():
    args = parse_args()
    validate_paths(args.uav_csv, args.task_csv)

    algo_list = (
        [args.algorithm]
        if args.algorithm != "all"
        else ["cbba", "auction", "contractnet"]
    )

    print("=" * 40)
    print("Task Allocation Compare Entry")
    print(f"scale   : {args.scale}")

    for algo_name in algo_list:
        algo_results = []
        print("\n" + "-" * 40)
        print(f"Running {algo_name.upper()}")
        print("-" * 40)
        for ep in range(1, args.episodes + 1):
            result = run_once(ep, args, algo_name)
            algo_results.append(result)
        if args.episodes > 1:
            summarize_results(algo_results, algo_name)


if __name__ == "__main__":
    main()
