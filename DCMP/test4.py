import matplotlib.pyplot as plt
import numpy as np

# 固定随机种子，确保数据可重复
np.random.seed(42)

# 全局字体配置（符合期刊规范，避免中文依赖）
plt.rcParams["font.family"] = ["Times New Roman", "serif"]
plt.rcParams["axes.unicode_minus"] = False  # 解决负号显示异常
plt.rcParams["font.size"] = 12  # 基础字体大小，保证标签协调

# ---------------- 生成与原图匹配的数据集 ----------------
# (a) Validation Set: RMSE=1.154, R=0.795
label_a = np.linspace(0, 12, 500)
noise_a = np.random.normal(0, 1.15, 500)
pred_a = 0.75 * label_a + 1.8 + noise_a  # 模拟线性关系+噪声

# (b) CASF2013: RMSE=1.267, R=0.870
label_b = np.linspace(0, 12, 300)
noise_b = np.random.normal(0, 1.26, 300)
pred_b = 0.85 * label_b + 0.8 + noise_b

# (c) CASF2016: RMSE=1.172, R=0.865
label_c = np.linspace(0, 12, 400)
noise_c = np.random.normal(0, 1.17, 400)
pred_c = 0.8 * label_c + 1.2 + noise_c

# (d) Holdout2019: RMSE=1.408, R=0.632
label_d = np.linspace(0, 12, 600)
noise_d = np.random.normal(0, 1.4, 600)
pred_d = 0.65 * label_d + 2.5 + noise_d

# ---------------- 创建2×2子图布局 ----------------
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
axes = axes.flatten()  # 转为一维数组，方便索引

# ---------------- 绘制子图(a) ----------------
axes[0].scatter(label_a, pred_a, color='red', alpha=0.5, marker='o')  # 红色圆点
coeff_a = np.polyfit(label_a, pred_a, 1)  # 拟合回归线
poly_a = np.poly1d(coeff_a)
axes[0].plot(label_a, poly_a(label_a), color='red')

axes[0].set_title('Validation Set(RMSE=1.154, R=0.795)', fontweight='bold')  # 标题加粗
axes[0].set_xlabel('Label', fontweight='bold')  # X标签加粗
axes[0].set_ylabel('Prediction', fontweight='bold')  # Y标签加粗
axes[0].grid(True)
axes[0].text(10, 1, '(a)', fontsize=24, fontweight='bold')  # 放大子图编号

# ---------------- 绘制子图(b) ----------------
axes[1].scatter(label_b, pred_b, color='blue', alpha=0.5, marker='s')  # 蓝色方块
coeff_b = np.polyfit(label_b, pred_b, 1)
poly_b = np.poly1d(coeff_b)
axes[1].plot(label_b, poly_b(label_b), color='blue')

axes[1].set_title('CASF2013(RMSE=1.267, R=0.870)', fontweight='bold')
axes[1].set_xlabel('Label', fontweight='bold')
axes[1].set_ylabel('Prediction', fontweight='bold')
axes[1].grid(True)
axes[1].text(10, 1, '(b)', fontsize=24, fontweight='bold') 

# ---------------- 绘制子图(c) ----------------
axes[2].scatter(label_c, pred_c, color='gold', alpha=0.5, marker='^')  # 黄色三角形
coeff_c = np.polyfit(label_c, pred_c, 1)
poly_c = np.poly1d(coeff_c)
axes[2].plot(label_c, poly_c(label_c), color='gold')

axes[2].set_title('CASF2016(RMSE=1.172, R=0.865)', fontweight='bold')
axes[2].set_xlabel('Label', fontweight='bold')
axes[2].set_ylabel('Prediction', fontweight='bold')
axes[2].grid(True)
axes[2].text(10, 1, '(c)', fontsize=24, fontweight='bold') 

# ---------------- 绘制子图(d) ----------------
axes[3].scatter(label_d, pred_d, color='green', alpha=0.5, marker='^')  # 绿色三角形
coeff_d = np.polyfit(label_d, pred_d, 1)
poly_d = np.poly1d(coeff_d)
axes[3].plot(label_d, poly_d(label_d), color='green')

axes[3].set_title('Holdout2019(RMSE=1.408, R=0.632)', fontweight='bold')
axes[3].set_xlabel('Label', fontweight='bold')
axes[3].set_ylabel('Prediction', fontweight='bold')
axes[3].grid(True)
axes[3].text(10, 1, '(d)', fontsize=24, fontweight='bold') 

# 调整子图间距，避免重叠
plt.tight_layout()

# 保存为SVG格式图片
plt.savefig("scatter_plots.svg", format="svg")