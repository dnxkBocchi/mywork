import matplotlib.pyplot as plt
import re

def plot_from_file(file_path):
    """
    从本地文件读取航程和时间数据，绘制散点图
    
    参数:
        file_path: 数据文件路径（如 'time_voyage.txt'）
    """
    # 方法名称（与数据顺序对应）
    methods = ['random', 'rr', 'ga', 'pso', 'mopso', 'dqn', 'my_dqn']
    
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
    
    # 设置样式（颜色和标记）
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A', '#98D8C8', '#F7DC6F', '#BB8FCE']
    markers = ['o', 's', '^', 'D', 'v', '<', '>']
    
    # 创建画布
    plt.figure(figsize=(12, 7))
    
    # 绘制散点图
    for i in range(len(methods)):
        plt.scatter(
            time_data[i],
            voyage_data[i],
            c=colors[i],
            label=methods[i],
            marker=markers[i],
            alpha=0.7,
            s=60,
            edgecolors='black',
            linewidth=0.5
        )
    
    # 图表设置
    plt.xlabel('时间 (time)', fontsize=12)
    plt.ylabel('航程 (voyage)', fontsize=12)
    plt.title('不同方法的航程与时间关系', fontsize=14, pad=20)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(title='方法', fontsize=10, loc='best')
    
    # 调整坐标轴范围（根据数据分布自动适配）
    all_voyage = [v for sublist in voyage_data for v in sublist]
    all_time = [t for sublist in time_data for t in sublist]
    plt.xlim(min(all_time) - 10, max(all_time) + 10)
    plt.ylim(min(all_voyage) - 10, max(all_voyage) + 10)
    
    plt.tight_layout()
    plt.show()

# 使用示例（请将路径替换为您的实际文件路径）
if __name__ == "__main__":
    plot_from_file('my/plt/time_voyage.txt')  # 假设文件与脚本在同一目录