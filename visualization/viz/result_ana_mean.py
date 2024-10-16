import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import sys
sys.path.append("visualization")
from graph import prop_graph
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import pickle
from typing import List

all_topic_df = pd.read_csv("data/label_clean_v7.csv")

def load_list(path):
    # 从文件中读取列表
    with open(path, 'rb') as file:
        loaded_list = pickle.load(file)
    return loaded_list

def get_stat_list(prop_g:prop_graph):
    _, node_nums = prop_g.plot_scale_time()
    node_nums += [node_nums[-1]]*(300-len(node_nums))
    _, depth_list = prop_g.plot_depth_time()
    depth_list += [depth_list[-1]]*(300-len(depth_list))
    _, max_breadth_list = prop_g.plot_max_breadth_time()
    max_breadth_list += [max_breadth_list[-1]]*(300-len(max_breadth_list))

    return [node_nums, depth_list, max_breadth_list]

def get_xdb_data(db_paths, topic_name):
    
    source_tweet_content = all_topic_df[all_topic_df["topic_name"] == topic_name]["source_tweet"].item()
    stats = []

    real_stat_list = []

    for index, stat in enumerate(["scale", "depth", "max_breadth"]):
        real_data_root = Path(f"data/real_world_data//real_data_{stat}")
        real_data_root.mkdir(parents=True, exist_ok=True)
        pkl_path = os.path.join(real_data_root, f"{topic_name}.pkl")
        Y_real = load_list(pkl_path)
        Y_real += [Y_real[-1]]*(300-len(Y_real))
        real_stat_list.append(Y_real)

    for db_path in db_paths:
        pg = prop_graph(source_tweet_content, db_path, viz=False)
        try:
            pg.build_graph()
            stats.append(get_stat_list(pg))
        except Exception as e:
            zero_stats = [[0]*300]*3
            stats.append(zero_stats)
            print(e)    

    stats.append(real_stat_list)
    return stats


def get_all_xdb_data(db_folders:List):
    topics = os.listdir(f"data/simu_db/{db_folders[0]}")
    topics = [topic.split(".")[0] for topic in topics]
    # len(db_folders) == 多少种 recsys的setting + real
    all_scale_lists = [[] for _ in range(len(db_folders) + 1)]
    all_depth_lists = [[] for _ in range(len(db_folders) + 1)]
    all_mb_lists = [[] for _ in range(len(db_folders) + 1)]
    # all_sv_lists = [[] for _ in range(len(db_folders) + 1)]

    for topic in tqdm(topics):
        db_paths = []
        for db_folder in db_folders:
            db_paths.append(f"data/simu_db/{db_folder}/{topic}.db")
        try:
            simu_data = get_xdb_data(db_paths,topic_name=topic)
            for db_index in range(len(db_folders)+1):
                all_scale_lists[db_index].append(simu_data[db_index][0][0:150])
                all_depth_lists[db_index].append(simu_data[db_index][1][0:150])
                all_mb_lists[db_index].append(simu_data[db_index][2][0:150])
                # all_sv_lists[db_index].append(simu_data[db_index][3][0:150])
        except Exception as e:
            print(f"Fail at topic {topic}, because {e}")
            # raise e
    all_scale_lists = np.array(all_scale_lists)
    all_depth_lists = np.array(all_depth_lists)
    all_mb_lists = np.array(all_mb_lists)
    # all_sv_lists = np.array(all_sv_lists)

    # return [[all_scale_lists[index], all_depth_lists[index], all_mb_lists[index], all_sv_lists[index]] for index in range(len(all_scale_lists))]
    return [[all_scale_lists[index], all_depth_lists[index], all_mb_lists[index]] for index in range(len(all_scale_lists))]

def plot_trend(db_folders:List, db_types:List):

    stats = get_all_xdb_data(db_folders)
    stats_name = ["scale", "depth", "max breadth"]
    # 绘制图表
    fig, axes = plt.subplots(1, 3, figsize=(21, 7))
    for stat_index, stat_name in enumerate(stats_name):
        # 确定子图的位置
        ax = axes[stat_index]
        colors = ["blue", "red", "orange", "magenta", "green", "purple", "orange"]

        for db_index, db_type in enumerate(db_types):
            # 计算均值和置信区间
            mean_values = np.mean(stats[db_index][stat_index], axis=0)
            std_dev = np.std(stats[db_index][stat_index], axis=0)
            confidence_interval = 1.96 * (std_dev / np.sqrt(stats[db_index][stat_index].shape[0]))

            # 绘制 db_type 数据
            ax.plot(mean_values, label=db_type, color=colors[db_index])
            ax.fill_between(range(stats[db_index][stat_index].shape[1]), mean_values - confidence_interval, mean_values + confidence_interval, color=colors[db_index], alpha=0.2, label=f'{db_type} 95% Confidence Interval')

        ax.set_xlabel('Time/minute', fontsize=22)
        ax.set_ylabel(stat_name, fontsize=22)
        ax.set_title(f'Trend of {stat_name} Over Time', fontsize=22)
        # 设置每个子图的x轴和y轴刻度字体大小
        ax.tick_params(axis='x', labelsize=20)
        ax.tick_params(axis='y', labelsize=20)
        ax.grid(True)
        # ax.legend()

    # 从其中一个子图获取句柄和标签
    handles, labels = ax.get_legend_handles_labels()

    # 创建共享图例，放置在整个图的下方
    fig.legend(handles, labels, loc='lower center', fontsize=20, ncol=2)
    # plt.subplots_adjust(left=0.05, right=0.95, top=0.9, bottom=0.15, wspace=0.4) 
    plt.tight_layout(rect=[0, 0.15, 1, 1])
    # plt.tight_layout()
    file_name = ""
    for type in db_types:
        file_name += f"{type}--"
    file_name += "all_stats.png"
    save_dir = Path(f"visualization/viz_results/compare_real/{file_name}")
    # save_dir = Path(f"visualization/viz_results/GENSS_/{file_name}")
    save_dir.parent.mkdir(parents=True, exist_ok=True)

    plt.savefig(save_dir)
    plt.show()

if __name__ == "__main__":

    plot_trend(db_folders=["yaml_200"], 
               db_types=["OASIS","Real"]) 