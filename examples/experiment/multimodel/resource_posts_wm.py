import sqlite3
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict
import numpy as np
import argparse

def load_misinfo_data():
    """加载misinfo.csv数据"""
    df = pd.read_csv('data/normal.csv')
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

def analyze_posts(db_path):
    """Analyze posts in a given SQLite DB and aggregate counts per time step."""
    # Connect DB
    conn = sqlite3.connect(db_path)
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
    df = pd.read_csv('data/normal.csv')
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

def compute_modd_series(time_step_stats):
    """Compute D_M(t), D_O(t) and Δ_MO(t) series from counts dict."""
    time_steps = sorted(time_step_stats.keys())
    official_news_counts = [time_step_stats[ts]['Official News'] for ts in time_steps]
    misinformation_counts = [time_step_stats[ts]['Misinformation'] for ts in time_steps]
    total_counts = [o + m for o, m in zip(official_news_counts, misinformation_counts)]
    dm = [(m / t) if t > 0 else 0.0 for m, t in zip(misinformation_counts, total_counts)]
    do = [(o / t) if t > 0 else 0.0 for o, t in zip(official_news_counts, total_counts)]
    delta_mo = [m_val - o_val for m_val, o_val in zip(dm, do)]
    return time_steps, dm, do, delta_mo, official_news_counts, misinformation_counts, total_counts

def plot_results(time_step_stats):
    """Plot metrics per paper: D_M(t), D_O(t) and MODD Δ_MO(t).

    - D_M(t) = N_M(t) / N_total(t)
    - D_O(t) = N_O(t) / N_total(t)
    - Δ_MO(t) = D_M(t) - D_O(t)
    """
    # Prepare data and compute metrics
    time_steps, dm, do, delta_mo, official_news_counts, misinformation_counts, total_counts = compute_modd_series(time_step_stats)

    # Create figure
    plt.figure(figsize=(12, 8))
    plt.plot(time_steps, dm, 'r-', label='D_M(t): Misinformation PDF', linewidth=2, marker='s')
    plt.plot(time_steps, do, 'b-', label='D_O(t): Official News PDF', linewidth=2, marker='o')
    plt.plot(time_steps, delta_mo, 'k--', label='Δ_MO(t) = D_M - D_O', linewidth=2, marker='^')

    # Axes and aesthetics
    plt.xlabel('Time Step', fontsize=12)
    plt.ylabel('Proportion / Difference', fontsize=12)
    plt.title('PDFs and MODD over Time Steps', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)

    # Rotate x-ticks for readability
    plt.xticks(rotation=45)

    # Layout
    plt.ylim(-1.0, 1.0)
    plt.tight_layout()

    # Save figure
    plt.savefig('result/2.png',
                dpi=300, bbox_inches='tight')

    # Show
    plt.show()

    # Print detailed stats per time step
    print('Per-time-step metrics:')
    print('=' * 50)
    for idx, ts in enumerate(time_steps):
        print(f"Time step {ts}:")
        print(f"  Counts -> Official: {official_news_counts[idx]}, Misinformation: {misinformation_counts[idx]}, Total: {total_counts[idx]}")
        print(f"  D_M: {dm[idx]:.4f}, D_O: {do[idx]:.4f}, Δ_MO: {delta_mo[idx]:.4f}")
        print()

def plot_modd_comparison(stats_a, stats_b, stats_c, label_a='with HCI', label_b='w/o HCI', label_c='None', save_path='/home/lsj/Projects/Gitself/oasis_main/visualization/twitter_simulation/human_interect/modd_compare.png'):
    """Plot Δ_MO comparison for two configurations on aligned time steps.

    Only the MODD curves are drawn to replicate the ablation style figure.
    """
    # Build aligned time steps
    all_steps = sorted(set(list(stats_a.keys()) + list(stats_b.keys())))

    # Helper to get counts list per steps
    def get_counts(stats):
        o_counts = [stats.get(ts, {'Official News': 0, 'Misinformation': 0})['Official News'] for ts in all_steps]
        m_counts = [stats.get(ts, {'Official News': 0, 'Misinformation': 0})['Misinformation'] for ts in all_steps]
        totals = [o + m for o, m in zip(o_counts, m_counts)]
        dm = [(m / t) if t > 0 else 0.0 for m, t in zip(m_counts, totals)]
        do = [(o / t) if t > 0 else 0.0 for o, t in zip(o_counts, totals)]
        delta = [m_val - o_val for m_val, o_val in zip(dm, do)]
        return delta

    delta_a = get_counts(stats_a)
    delta_b = get_counts(stats_b)
    delta_c = get_counts(stats_c)

    plt.figure(figsize=(12, 6))
    plt.plot(all_steps, delta_a, 'b-', label=label_a, linewidth=2, marker='o')
    plt.plot(all_steps, delta_b, 'r-', label=label_b, linewidth=2, marker='s')
    plt.plot(all_steps, delta_c, 'y-', label=label_c, linewidth=2, marker='p')

    plt.xlabel('Time Step', fontsize=12)
    plt.ylabel('MODD', fontsize=12)
    plt.title('MODD Comparison over Time Steps', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=0)
    plt.ylim(-1.0, 1.0)
    plt.tight_layout()

    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"Figure saved: {save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Compute and visualize misinformation metrics.')
    parser.add_argument('--db_a', type=str, default='data/normal.db', help='Path to SQLite DB A (e.g., with HCI)')
    parser.add_argument('--db_b', type=str, default='data/read.db', help='Path to SQLite DB B (e.g., w/o HCI)')
    parser.add_argument('--db_c', type=str, default='data/multimodel.db', help='Path to SQLite DB C (e.g., w/o HCI)')
    parser.add_argument('--label_a', type=str, default='normal', help='Legend label for DB A')
    parser.add_argument('--label_b', type=str, default='read', help='Legend label for DB B')
    parser.add_argument('--label_c', type=str, default='multimodel', help='Legend label for DB C')
    parser.add_argument('--save', type=str, default='result/new.png', help='Output image path for comparison figure')
    args = parser.parse_args()

    if args.db_a and args.db_b:
        print('Start analyzing two DBs for comparison ...')
        stats_a = analyze_posts(args.db_a)
        stats_b = analyze_posts(args.db_b)
        stats_c = analyze_posts(args.db_c)
        print('Analysis done, plotting comparison ...')
        plot_modd_comparison(stats_a, stats_b, stats_c, args.label_a, args.label_b, args.label_c, args.save)
    elif args.db_a:
        print('Start analyzing single DB ...')
        stats = analyze_posts(args.db_a)
        print('Analysis done, plotting PDFs and MODD ...')
        plot_results(stats)
    else:
        print('Please provide --db_a (and optionally --db_b) to proceed.')