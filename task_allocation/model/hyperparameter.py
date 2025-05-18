import argparse
import torch


def get_args():
    parser = argparse.ArgumentParser(description="Hyperparameter parser")
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0" if torch.cuda.is_available() else "cpu",
        help="Device (CPU or CUDA)",
    )

    # 基本系统主参数
    parser.add_argument("--state_dim", type=int, default=11, help="State dimension")
    parser.add_argument("--hidden_size", type=int, default=128, help="Hidden size")
    parser.add_argument("--action_num", type=int, default=5, help="Number of actions")
    parser.add_argument("--episode_number", type=int, default=200)
    parser.add_argument("--test_number", type=int, default=1, help="test number")
    parser.add_argument("--wf_number", type=int, default=10, help="Number of workflows")
    parser.add_argument("--debug", type=bool, default=True, help="Debug model")
    parser.add_argument("--buffer_size", type=int, default=10000, help="Buffer size")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size")

    # 动态参数
    parser.add_argument("--destroy_ncp", type=bool, default=False, help="destroy_ncp")
    parser.add_argument("--increase_task", type=bool, default=False, help="increase")

    # 环境超参数
    parser.add_argument("--random_seed", type=int, default=50, help="Random seed")
    parser.add_argument("--arrival_rate", type=float, default=0.1 / 60, help="second")

    # 模型超参数
    # 每隔多少个训练步骤或迭代就会将行为网络的权重复制到目标网络
    parser.add_argument("--target_update", type=int, default=20)
    # 折扣因子，用于平衡未来奖励与当前奖励的重要性
    parser.add_argument("--discount_factor", type=float, default=0.5)
    # 学习率过高，模型可能会在每次更新时过度调整。如果学习率过低，模型的学习进度可能太慢
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    # 每隔多少个 epoch 调整一次学习率
    parser.add_argument("--step_size", type=int, default=1)
    # 控制 L2 正则化项的强度，它会被加到优化器的更新步骤中，从而在每次更新权重时对权重施加衰减
    parser.add_argument("--l2_weight_decay", type=float, default=1e-4)
    # 平衡时间和成本的权重
    parser.add_argument("--beta", type=float, default=0.6)

    # PPO
    parser.add_argument("--K_epochs", type=int, default=20)
    parser.add_argument("--eps_clip", type=float, default=0.25)
    parser.add_argument("--lr_actor", type=float, default=1e-4)
    parser.add_argument("--lr_critic", type=float, default=1e-3)
    parser.add_argument("--update", type=float, default=10)

    # GAT
    parser.add_argument("--nfeat", type=int, default=8, help="Number of features.")
    parser.add_argument("--hidden", type=int, default=8, help="Number of hidden units.")
    parser.add_argument("--out_feature", type=int, default=32, help="Number of out")
    parser.add_argument("--nb_heads", type=int, default=8, help="head attentions.")
    parser.add_argument("--dropout", type=float, default=0.6)
    parser.add_argument("--alpha", type=float, default=0.2, help="leaky_relu.")

    args, unknown = parser.parse_known_args()  # 解析已知参数，忽略未知参数
    return args
