import sqlite3
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict
import numpy as np

def load_misinfo_data():
    """加载misinfo.csv数据"""
    df = pd.read_csv('data/read.csv')
    # 创建post_id到label的映射
    misinfo_map = {}
    for _, row in df.iterrows():
        post_id = row['id']
        label = row['label']
        # 将post_id转换为字符串以便匹配
        misinfo_map[str(post_id)] = 'Official News' if label else 'Misinformation'
    return misinfo_map

def find_root_post(post_id, posts_dict):
    """回溯找到根源post"""
    current_post = posts_dict.get(post_id)
    if current_post is None:
        return None
    
    # 如果original_post_id为None，说明这就是根源post
    if current_post['original_post_id'] is None:
        return post_id
    
    # 递归查找根源post
    return find_root_post(current_post['original_post_id'], posts_dict)

def analyze_posts():
    """分析posts并统计时间步"""
    # 连接数据库
    conn = sqlite3.connect('data/read.db')
    cursor = conn.cursor()
    
    # 读取所有posts
    cursor.execute("SELECT post_id, user_id, original_post_id, content, created_at FROM post")
    posts_data = cursor.fetchall()
    
    # 创建posts字典
    posts_dict = {}
    for post_id, user_id, original_post_id, content, created_at in posts_data:
        posts_dict[post_id] = {
            'user_id': user_id,
            'original_post_id': original_post_id,
            'content': content,
            'created_at': created_at
        }
    
    # 加载misinfo数据
    misinfo_map = load_misinfo_data()
    
    # 通过内容匹配找到misinfo数据
    content_to_label = {}
    df = pd.read_csv('data/read.csv')
    for _, row in df.iterrows():
        content = row['source_tweet']
        label = 'Official News' if row['label'] else 'Misinformation'
        content_to_label[content] = label
    
    # 统计每个时间步的数据
    time_step_stats = defaultdict(lambda: {'Official News': 0, 'Misinformation': 0})
    
    for post_id, post_info in posts_dict.items():
        # 找到根源post
        root_post_id = find_root_post(post_id, posts_dict)
        
        if root_post_id:
            root_post = posts_dict[root_post_id]
            root_content = root_post['content']
            
            # 通过内容匹配找到标签
            if root_content in content_to_label:
                time_step = post_info['created_at']
                label = content_to_label[root_content]
                time_step_stats[time_step][label] += 1
    
    conn.close()
    return time_step_stats

def plot_results(time_step_stats):
    """绘制折线图"""
    # 准备数据
    time_steps = sorted(time_step_stats.keys())
    official_news_counts = [time_step_stats[ts]['Official News'] for ts in time_steps]
    misinformation_counts = [time_step_stats[ts]['Misinformation'] for ts in time_steps]
    
    # 创建图表
    plt.figure(figsize=(12, 8))
    plt.plot(time_steps, official_news_counts, 'b-', label='Official News', linewidth=2, marker='o')
    plt.plot(time_steps, misinformation_counts, 'r-', label='Misinformation', linewidth=2, marker='s')
    
    # 设置图表属性
    plt.xlabel('Time Step', fontsize=12)
    plt.ylabel('Number of Posts', fontsize=12)
    plt.title('Posts Distribution Over Time Steps', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    
    # 设置x轴标签
    plt.xticks(rotation=45)
    
    # 调整布局
    plt.tight_layout()
    
    # 保存图表
    plt.savefig('result/result.png',
                dpi=300, bbox_inches='tight')
    
    # 显示图表
    plt.show()
    
    # 打印统计信息
    print("时间步统计:")
    print("=" * 50)
    for ts in time_steps:
        print(f"时间步 {ts}:")
        print(f"  Official News: {time_step_stats[ts]['Official News']}")
        print(f"  Misinformation: {time_step_stats[ts]['Misinformation']}")
        print()

if __name__ == "__main__":
    print("开始分析posts数据...")
    time_step_stats = analyze_posts()
    print("分析完成，开始绘制图表...")
    plot_results(time_step_stats)
    print("图表已保存为 result.png")