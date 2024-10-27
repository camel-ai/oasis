import json
import sqlite3

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats


class Database:

    def __init__(self, db_path):
        self.conn = sqlite3.connect(db_path)

    def get_score_comment_id(self, comment_id):
        cursor = self.conn.cursor()
        """
        根据评论ID查询特定评论的分数（点赞数减去踩数）。
        """
        # 准备SQL查询，计算得分（点赞数 - 踩数）
        query = """
        SELECT (num_likes - num_dislikes) AS score
        FROM comment
        WHERE comment_id = ?
        """
        # 执行查询
        cursor.execute(query, (comment_id, ))
        # 获取查询结果
        result = cursor.fetchone()
        # 如果查询有结果，返回得分；否则返回None
        if result:
            return result[0]
        else:
            return None


def get_result(comment_id_lst, db_path):
    db = Database(db_path)

    result_lst = []

    for track_comment_id in comment_id_lst:
        result = db.get_score_comment_id(track_comment_id)
        if result is None:
            print(f"Comment with id:{track_comment_id} not found.")
            result_lst.append(result)
        else:
            result_lst.append(result)
    return result_lst


# 计算均值和95%置信区间的函数
def mean_confidence_interval(data, confidence=0.95):
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), stats.sem(a)
    h = se * stats.t.ppf((1 + confidence) / 2., n - 1)
    return m, m - h, m + h


def visualization(up_result, down_result, control_result, exp_name,
                  folder_path):
    # 计算每组数据的均值和置信区间
    up_mean, up_ci_low, up_ci_high = mean_confidence_interval(up_result)
    down_mean, down_ci_low, down_ci_high = mean_confidence_interval(
        down_result)
    control_mean, control_ci_low, control_ci_high = mean_confidence_interval(
        control_result)

    # 绘图
    labels = ['Down', 'Control', 'Up']
    means = [down_mean, control_mean, up_mean]
    conf_intervals = [(down_ci_low, down_ci_high),
                      (control_ci_low, control_ci_high),
                      (up_ci_low, up_ci_high)]

    x_pos = range(len(labels))  # x位置

    fig, ax = plt.subplots()

    # 绘制条形图
    # 注意：yerr的计算方式也需要调整，以确保误差条与相应的均值对应
    ax.bar(labels,
           means,
           color='skyblue',
           yerr=np.transpose(
               [[mean - ci_low, ci_high - mean]
                for mean, (ci_low, ci_high) in zip(means, conf_intervals)]),
           capsize=10)

    # 在条形上方添加点表示均值
    for i, mean in enumerate(means):
        ax.plot(x_pos[i], mean, 'ro')  # 'ro'表示红色的圆点

    ax.set_ylabel('Scores')
    ax.set_title('Mean Scores with 95% Confidence Intervals')

    # 保存图像，确保目录存在或者调整为正确的路径
    plt.savefig(f"{folder_path}/"
                f"score_{exp_name}.png")

    plt.show()


def main(exp_info_file_path, db_path, exp_name, folder_path):
    with open(exp_info_file_path, 'r') as file:
        exp_info = json.load(file)

    up_result = get_result(exp_info["up_comment_id"], db_path)
    down_result = get_result(exp_info["down_comment_id"], db_path)
    control_result = get_result(exp_info["control_comment_id"], db_path)
    print('up_result:', up_result, 'down_result:', down_result,
          'control_result', control_result)
    visualization(up_result, down_result, control_result, exp_name,
                  folder_path)


if __name__ == "__main__":
    main(exp_info_file_path=(
        './experiments/reddit_herding_effect/results_analysis/'
        'result_data/exp_info.json'),
         db_path=('./experiments/reddit_herding_effect/results_analysis/'
                  'result_data/mock_reddit_06-30_06-33-29.db'))
