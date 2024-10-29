# =========== Copyright 2023 @ CAMEL-AI.org. All Rights Reserved. ===========
# Licensed under the Apache License, Version 2.0 (the “License”);
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an “AS IS” BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =========== Copyright 2023 @ CAMEL-AI.org. All Rights Reserved. ===========
import os
import pickle
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
# sys.path.append("visualization/twitter_simulation")
from graph import prop_graph
from tqdm import tqdm

all_topic_df = pd.read_csv("data/twitter_dataset/all_topics.csv")


def load_list(path):
    # load real world propagation data from file
    with open(path, "rb") as file:
        loaded_list = pickle.load(file)
    return loaded_list


def get_stat_list(prop_g: prop_graph):
    _, node_nums = prop_g.plot_scale_time()
    node_nums += [node_nums[-1]] * (300 - len(node_nums))
    _, depth_list = prop_g.plot_depth_time()
    depth_list += [depth_list[-1]] * (300 - len(depth_list))
    _, max_breadth_list = prop_g.plot_max_breadth_time()
    max_breadth_list += [max_breadth_list[-1]] * (300 - len(max_breadth_list))

    return [node_nums, depth_list, max_breadth_list]


def get_xdb_data(db_paths, topic_name):
    source_tweet_content = all_topic_df[all_topic_df["topic_name"] ==
                                        topic_name]["source_tweet"].item()
    stats = []
    real_stat_list = []

    for index, stat in enumerate(["scale", "depth", "max_breadth"]):
        real_data_root = Path(
            f"data/twitter_dataset/real_world_prop_data/real_data_{stat}")
        real_data_root.mkdir(parents=True, exist_ok=True)
        pkl_path = os.path.join(real_data_root, f"{topic_name}.pkl")
        Y_real = load_list(pkl_path)
        Y_real += [Y_real[-1]] * (300 - len(Y_real))
        real_stat_list.append(Y_real)

    for db_path in db_paths:
        pg = prop_graph(source_tweet_content, db_path, viz=False)
        try:
            pg.build_graph()
            stats.append(get_stat_list(pg))
        except Exception as e:
            zero_stats = [[0] * 300] * 3
            stats.append(zero_stats)
            print(e)

    stats.append(real_stat_list)
    return stats


def get_all_xdb_data(db_folders: List):
    topics = os.listdir(f"data/simu_db/{db_folders[0]}")
    topics = [topic.split(".")[0] for topic in topics]
    # len(db_folders) == simulation results + real world propagation data  OR
    # len(db_folders) == different simulation settings +  real world
    # propagation data
    all_scale_lists = [[] for _ in range(len(db_folders) + 1)]
    all_depth_lists = [[] for _ in range(len(db_folders) + 1)]
    all_mb_lists = [[] for _ in range(len(db_folders) + 1)]

    for topic in tqdm(topics):
        db_paths = []
        for db_folder in db_folders:
            db_paths.append(f"data/simu_db/{db_folder}/{topic}.db")
        try:
            simu_data = get_xdb_data(db_paths, topic_name=topic)
            for db_index in range(len(db_folders) + 1):
                all_scale_lists[db_index].append(simu_data[db_index][0][0:150])
                all_depth_lists[db_index].append(simu_data[db_index][1][0:150])
                all_mb_lists[db_index].append(simu_data[db_index][2][0:150])
        except Exception as e:
            print(f"Fail at topic {topic}, because {e}")
            # raise e
    all_scale_lists = np.array(all_scale_lists)
    all_depth_lists = np.array(all_depth_lists)
    all_mb_lists = np.array(all_mb_lists)

    return [[
        all_scale_lists[index], all_depth_lists[index], all_mb_lists[index]
    ] for index in range(len(all_scale_lists))]


def plot_trend(db_folders: List, db_types: List):
    stats = get_all_xdb_data(db_folders)
    stats_name = ["scale", "depth", "max breadth"]

    fig, axes = plt.subplots(1, 3, figsize=(21, 7))
    for stat_index, stat_name in enumerate(stats_name):
        ax = axes[stat_index]
        colors = [
            "blue", "red", "orange", "magenta", "green", "purple", "orange"
        ]

        for db_index, db_type in enumerate(db_types):
            # calculate mean and confidence interval
            mean_values = np.mean(stats[db_index][stat_index], axis=0)
            std_dev = np.std(stats[db_index][stat_index], axis=0)
            confidence_interval = 1.96 * (
                std_dev / np.sqrt(stats[db_index][stat_index].shape[0]))

            ax.plot(mean_values, label=db_type, color=colors[db_index])
            ax.fill_between(
                range(stats[db_index][stat_index].shape[1]),
                mean_values - confidence_interval,
                mean_values + confidence_interval,
                color=colors[db_index],
                alpha=0.2,
                label=f"{db_type} 95% Confidence Interval",
            )

        ax.set_xlabel("Time/minute", fontsize=22)
        ax.set_ylabel(stat_name, fontsize=22)
        ax.set_title(f"Trend of {stat_name} Over Time", fontsize=22)

        ax.tick_params(axis="x", labelsize=20)
        ax.tick_params(axis="y", labelsize=20)
        ax.grid(True)

    handles, labels = ax.get_legend_handles_labels()

    fig.legend(handles, labels, loc="lower center", fontsize=20, ncol=2)
    plt.tight_layout(rect=[0, 0.15, 1, 1])
    file_name = ""
    for type in db_types:
        file_name += f"{type}--"
    file_name += "all_stats.png"
    save_dir = Path(f"visualization/twitter_simulation/align_with_real_world"
                    f"/results/{file_name}")
    save_dir.parent.mkdir(parents=True, exist_ok=True)

    plt.savefig(save_dir)
    plt.show()


if __name__ == "__main__":
    plot_trend(db_folders=["yaml_200"], db_types=["OASIS", "Real"])
