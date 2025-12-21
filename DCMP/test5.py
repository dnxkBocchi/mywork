import matplotlib.pyplot as plt
import numpy as np

# 固定随机种子，确保数据可重复
np.random.seed(42)

# 全局字体配置（符合期刊规范，避免中文依赖）
plt.rcParams["font.family"] = ["Times New Roman", "serif"]
plt.rcParams["axes.unicode_minus"] = False  # 解决负号显示异常
plt.rcParams["font.size"] = 12  # 基础字体大小，保证标签协调

# 为可复现性设定随机种子
rng = np.random.default_rng(7)

def make_epoch_data(n=200, epoch=1, max_epoch=180, noise=0.9):
    """
    生成某个 epoch 的二维样本与“置信度”分数。
    - 类间距离随 epoch 线性增大，noise 控制随机性。
    - 返回:
        X: (n,2) 坐标
        s: (n,) 颜色分数 ∈ [0,1]
    """
    # 两类各 n/2 个样本
    n1 = n // 2
    n2 = n - n1

    # 类间分离度（0~1）
    sep = epoch / max_epoch

    # 两个类的均值随 sep 分开（这会让后期更可分）
    mu1 = np.array([-1.2 - 1.8*sep, -1.0 - 1.2*sep])
    mu2 = np.array([+1.2 + 1.8*sep, +1.0 + 1.2*sep])

    # 协方差（同一套，略带各向异性）
    cov = np.array([[0.8*noise, 0.15],
                    [0.15,       0.8*noise]])

    x1 = rng.multivariate_normal(mu1, cov, size=n1)
    x2 = rng.multivariate_normal(mu2, cov, size=n2)
    X  = np.vstack([x1, x2])

    # 用一个方向向量投影出“分数”，再过 sigmoid 得到 [0,1]
    # 方向也随 epoch 稍微旋转
    theta = np.deg2rad(35 + 20*sep)
    w = np.array([np.cos(theta), np.sin(theta)])  # 分割方向
    z = X @ w                                    # 线性投影
    z = (z - z.mean()) / (z.std() + 1e-9)        # 标准化
    s = 1 / (1 + np.exp(-1.6*z))                 # sigmoid -> [0,1]
    return X, s

def plot_epochs_svg(epochs=(1, 50, 100, 180), n=220, save_path="epochs_scatter.svg"):
    # 全局外观
    plt.rcParams.update({
        "font.size": 14,
        "figure.figsize": (9, 8)  # 控制整体比例，SVG 中是矢量可任意缩放
    })

    fig, axes = plt.subplots(2, 2, constrained_layout=True)
    axes = axes.ravel()

    # 为颜色条记录所有分数（保证统一 vmin/vmax）
    all_scores = []

    # 先生成所有 epoch 的数据，便于统一颜色尺度
    data = []
    max_epoch = max(epochs)
    for ep in epochs:
        X, s = make_epoch_data(n=n, epoch=ep, max_epoch=max_epoch, noise=0.9)
        data.append((ep, X, s))
        all_scores.append(s)

    vmin, vmax = 0.0, 1.0  # 固定范围便于比较

    # 依次作图
    for ax, (ep, X, s) in zip(axes, data):
        ax.scatter(X[:, 0], X[:, 1], c=s, cmap="coolwarm", vmin=vmin, vmax=vmax, s=26)
        ax.set_title(f"Epoch = {ep}", pad=8, weight="bold")
        ax.set_aspect("equal", adjustable="box")
        ax.set_xticks([]); ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(True)  # 显示边框

    # 统一颜色条
    # 在右侧放一个竖直 colorbar，标签 0~1
    sm = plt.cm.ScalarMappable(cmap="coolwarm")
    sm.set_clim(vmin, vmax)
    cbar = fig.colorbar(sm, ax=axes, fraction=0.046, pad=0.04)
    cbar.set_ticks([0.0, 0.5, 1.0])
    cbar.set_label("score", rotation=90)

    # 保存为矢量 SVG（可无损缩放/编辑）
    fig.savefig(save_path, format="svg", bbox_inches="tight")
    print(f"Saved: {save_path}")

if __name__ == "__main__":
    plot_epochs_svg()
