import matplotlib.pyplot as plt
import re
import os
import sys

def plot_from_file(file_path, save_dir):
    """
    从本地文件读取航程和时间数据，绘制散点图（黑白风格）
    
    参数:
        file_path: 数据文件路径（如 'time_voyage.txt'）
        save_dir: 图像保存路径
    """
    # 方法名称（与数据顺序对应）
    methods = ['RANDOM', 'RR', 'GA', 'PSO', 'MOPSO', 'DRL', 'GMP-DRL']
    # 设置中文字体支持，选用更美观的字体
    plt.rcParams["font.family"] = ["Times New Roman", "serif"]
    plt.rcParams["axes.unicode_minus"] = False  # 解决负号显示问题

    # 存储读取的数据
    voyage_data = []
    time_data = []
    
    # 读取文件内容
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 使用正则表达式提取所有航程和时间数据组
    # 匹配 "voyage: ..." 或 "time: ..." 开头的行
    patterns = {
        'voyage': re.findall(r'voyage:\s*([\d.,\s]+)', content),
        'time': re.findall(r'time:\s*([\d.,\s]+)', content)
    }
    
    # 处理航程数据
    for voyage_str in patterns['voyage']:
        # 移除逗号和多余空格，转换为浮点数列表
        values = list(map(float, re.sub(r'[, ]+', ' ', voyage_str).strip().split()))
        voyage_data.append(values)
    
    # 处理时间数据
    for time_str in patterns['time']:
        values = list(map(float, re.sub(r'[, ]+', ' ', time_str).strip().split()))
        time_data.append(values)
    
    # 设置样式（使用不同标记形状，不使用颜色）
    markers = ['o', 's', '^', 'D', 'v', '<', '>']  # 为每个方法分配不同的标记

    # 创建画布
    plt.figure(figsize=(6, 6))
    
    # 绘制散点图（使用黑白风格，通过不同标记区分）
    for i in range(len(methods)):
        plt.scatter(
            time_data[i],
            voyage_data[i],
            label=methods[i],
            marker=markers[i],
            alpha=0.7,
            s=60 if methods[i] != 'GMP-DRL' else 100,
            edgecolors='black',
            linewidth=0.5 if methods[i] != 'GMP-DRL' else 2,
            facecolor='none'  # 空心点，增强黑白效果
        )
    
    # 图表设置
    plt.xlabel('Tasks finish time', fontsize=12)
    plt.ylabel('UAVs voyage', fontsize=12)
    plt.title('Different methods of voyage and time relationship', fontsize=14, pad=20)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(fontsize=10, loc='best')
    
    # 调整坐标轴范围（根据数据分布自动适配）
    all_voyage = [v for sublist in voyage_data for v in sublist]
    all_time = [t for sublist in time_data for t in sublist]
    plt.xlim(min(all_time) - 10, max(all_time) + 10)
    plt.ylim(min(all_voyage) - 10, max(all_voyage) + 10)
    
    plt.tight_layout()
    plt.savefig(save_dir, dpi=300, bbox_inches='tight')  # 保存图片
    plt.show()

# 使用示例（请将路径替换为您的实际文件路径）
if __name__ == "__main__":
    current_script_dir = os.path.dirname(os.path.abspath(__file__))
    save_dir = os.path.join(current_script_dir, "..", "pic", "scatter_plot.svg")
    plot_from_file('my/plt/time_voyage.txt', save_dir)  # 假设文件与脚本在同一目录
