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
    # _, sv_list = prop_g.plot_structural_virality_time()
    # sv_list += [sv_list[-1]]*(300-len(sv_list))

    return [node_nums, depth_list, max_breadth_list]

def get_xdb_data(db_paths, topic_name):
 
    source_tweet_content = all_topic_df[all_topic_df["topic_name"] == topic_name]["source_tweet"].item()
    stats = []
    for db_path in db_paths:
        pg = prop_graph(source_tweet_content, db_path, viz=False)
        try:
            pg.build_graph()
            stats.append(get_stat_list(pg))
        except:
            zero_stats = [[0]*300]*3
            stats.append(zero_stats)
            # print("e")    
    
    real_stat_list = []

    # for index, stat in enumerate(["scale", "depth", "max_breadth", "structural_virality"]):
    for index, stat in enumerate(["scale", "depth", "max_breadth"]):
        real_data_root = Path(f"visualization/real_data//real_data_{stat}")
        real_data_root.mkdir(parents=True, exist_ok=True)
        pkl_path = os.path.join(real_data_root, f"{topic_name}.pkl")
        Y_real = load_list(pkl_path)
        # Y_real = np.array([Y_real[:300]]).reshape(-1, 1)
        Y_real += [Y_real[-1]]*(300-len(Y_real))
        # Y_real = np.array([Y_real[:300]]).reshape(-1, 1)
        real_stat_list.append(Y_real)
    stats.append(real_stat_list)

    return stats


def get_all_xdb_data(db_folders:List):
    # topics = os.listdir("visualization/simu_db/recsys_ablation/without_rec")
    topics = os.listdir(f"visualization/simu_db/{db_folders[0]}")
    topics = [topic.split(".")[0] for topic in topics]
    # topics = ["False_Business_0"]
    # len(db_folders) == 多少种 recsys的setting + real
    all_scale_lists = [[] for _ in range(len(db_folders) + 1)]
    all_depth_lists = [[] for _ in range(len(db_folders) + 1)]
    all_mb_lists = [[] for _ in range(len(db_folders) + 1)]
    # all_sv_lists = [[] for _ in range(len(db_folders) + 1)]

    for topic in tqdm(topics):
        db_paths = []
        for db_folder in db_folders:
            db_paths.append(f"visualization/simu_db/{db_folder}/{topic}.db")
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

def plot_rmse(db_folders:List, db_types:List):
    stats = get_all_xdb_data(db_folders)
    # stats_names = ["scale", "depth", "max breadth", "structural virality"]
    stats_names = ["scale", "depth", "max breadth"]
    # stats_names = ["scale"]
    # stats_names = ["depth", "max breadth"]
    # stats_names = ["scale", "max breadth"]
    # 绘制图表
    # plt.figure(figsize=(14, 7))
    fig, axes = plt.subplots(1, 3, figsize=(28, 7))
    markers = ['o', '^', 's', 'D', 'v', '*']
    for stat_index, stat_name in enumerate(stats_names):
        # 确定子图的位置
        # row = stat_index // 2
        # col = stat_index % 2
        # ax = axes[row, col]
        ax = axes[stat_index]
        # ax = axes  # 三种stat的loss画到一张图里
        # stat_index += 1 # when stats_names = ["depth", "max breadth"]
        if stat_name == "max breadth":
            stat_index = 2
        colors = ["blue", "red", "orange", "magenta", "green", "purple", "orange"]
       
        for type_index, db_type in enumerate(db_types):
            topic_rmse_losses = []
            topic_rmse_losses_per_min = []
            for topic_idx in range(len(stats[0][stat_index])):
                simu_arr = np.array(stats[type_index][stat_index][topic_idx])
                real_arr = np.array(stats[-1][stat_index][topic_idx])
                # 算出来t时刻的RMSE后，还需要除以真实传播过程中所涉及的最大用户数，算百分比偏差
                rmse_loss_per_min = np.abs(simu_arr - real_arr) / real_arr.max()
                rmse_loss = np.sqrt(np.mean((simu_arr - real_arr) ** 2)) / real_arr.max()
                topic_rmse_losses.append(rmse_loss)
                topic_rmse_losses_per_min.append(rmse_loss_per_min)

            topic_rmse_losses_per_min = np.array(topic_rmse_losses_per_min)
            rmse_losses = np.mean(np.array(topic_rmse_losses))
            print(f"{db_type}_{stat_name} rmse loss: {rmse_losses}")
            rmse_losses_per_min = np.mean(topic_rmse_losses_per_min, axis=0)
            # ax.plot(rmse_losses, label=db_type, color=colors[type_index], marker=markers[type_index], markevery=3)
            # ax.plot(rmse_losses, label=f"{stat_name}", color=colors[stat_index], marker=markers[stat_index], markevery=3)  # 三种stat的loss画到一张图里
            ax.plot(rmse_losses_per_min, label=f"{db_type}", color=colors[type_index], marker=markers[type_index], markevery=3)  # rec ablation
            

        ax.set_xlabel('Time/minute', fontsize=22)  # 设置X轴标签的字体大小
        if stat_index == 0:
            ax.set_ylabel("Loss", fontsize=22)

        ax.grid(True)
        ax.set_title(f'Trend of {stat_name} Normalized RMSE Over Time', fontsize=22)
        # 设置每个子图的x轴和y轴刻度字体大小
        ax.tick_params(axis='x', labelsize=20)
        ax.tick_params(axis='y', labelsize=20)


    # ax.set_title(f'Trend of scale Normalized RMSE Over Time', fontsize=22) # 三种stat的loss画到一张图里，或者是只画一张图
    # ax.legend(fontsize=22, loc='lower right')   
    

    # 从其中一个子图获取句柄和标签
    handles, labels = ax.get_legend_handles_labels()

    # 创建共享图例，放置在整个图的下方
    fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, 0), fontsize=22, ncol=3)
    plt.tight_layout(rect=[0, 0.13, 1, 1])
    # plt.tight_layout()
    file_name = ""
    for type in db_types:
        if "w/o" in type:
            type = type.replace("w/o", "without")
        file_name += f"{type}--"
    file_name += "all_stats.png"
    save_dir = Path(f"loss/compare_model/{file_name}")
    save_dir.parent.mkdir(parents=True, exist_ok=True)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.savefig(save_dir)
    plt.show()

if __name__ == "__main__":
    # plot_rmse(db_folders=["time_ablation", "topic200-fan_score_log1000"], 
    #          db_types=["OASIS w/o temporal feature", "OASIS"])
    # plot_rmse(db_folders=["topic200-fan_score_log1000",], 
    #            db_types=["OASIS", "Real"])
    # plot_rmse(db_folders=["recsys_ablation/without_rec", "recsys_ablation/twhin_rec_no_fan_score_no_like_score", "recsys_ablation/twhin_rec_30_fan_score_log1000_no_like_score", "recsys_ablation/twhin_rec_fan_score_log1000_like_score", "recsys_ablation/ST_rec_fan_score_log1000_like_score"], 
    #           db_types=["No recsys", "twhin no fan&like", "twhin no like", "twhin", "ST", "Real"])
    # plot_rmse(db_folders=["recsys_ablation/no_rec", "recsys_ablation/ST", "recsys_ablation/BERT", "recsys_ablation/TWHIN"], 
    #         db_types=["Without Recsys","Recsys based on paraphrase-MiniLM-L6-v2", "Recsys based on BERT-base-multilingual-cased", "Recsys based on TWHIN-BERT"])
    plot_rmse(db_folders=["model_ablation/llama3-8b", "model_ablation/internlm2-chat-20b", "model_ablation/qwen1.5", ], 
              db_types=["llama3-8b", "internlm2-chat-20b",  "qwen1.5-7B-Chat"])