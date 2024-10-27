import json
import sqlite3
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm

# 1. 追踪所有被实验的comment_id的tweet_id
# 2. 追踪对应tweet的第一个refresh看到的user对应的trace位置
# 3. 追踪在这个user下一次refresh之前是否对comment做出了反应
# 4. 如果有反应，追踪停止；如果无反应，可以继续追踪下一个看到的user，回到2


class Database:

    def __init__(self, db_path):
        self.conn = sqlite3.connect(db_path)

    def find_user_by_comment_id(self, comment_id, start_at):
        cursor = self.conn.cursor()
        # 查询action为'refresh'的记录，按created_at排序
        query = '''
        SELECT user_id, created_at, info
        FROM trace
        WHERE action = 'refresh'
        AND created_at > ?
        ORDER BY created_at
        '''
        cursor.execute(query, (start_at, ))
        results = cursor.fetchall()

        # 遍历查询结果
        for result in results:
            user_id, created_at, info = result
            # 解析info字段中的JSON数据
            info_data = json.loads(info)
            posts = info_data.get("posts", [])

            # 检查是否有匹配的comment_id
            for post in posts:
                comments = post.get("comments", [])
                for comment in comments:
                    if comment.get("comment_id") == comment_id:
                        return user_id, created_at

        # 没有人看过这条评论，返回None
        return None

    def track_ope_before_next_refresh(self, track_user_id, start_time,
                                      track_comment_id):
        cursor = self.conn.cursor()

        # 获取下一条action='refresh'的记录的created_at时间
        query_next_refresh_time = '''
        SELECT created_at
        FROM trace
        WHERE action = 'refresh' AND created_at > ? AND user_id = ?
        ORDER BY created_at
        LIMIT 1
        '''
        cursor.execute(query_next_refresh_time, (start_time, track_user_id))
        next_refresh_time = cursor.fetchone()

        # print(start_time, next_refresh_time)

        # 如果没有找到下一条refresh记录，则不限制结束时间
        if not next_refresh_time:
            end_time = '9999-12-31 23:59:59'
        else:
            end_time = next_refresh_time[0]

        # 检查在时间区间内是否有like的记录
        query_like = '''
        SELECT 1
        FROM comment_like
        WHERE user_id = ? AND comment_id = ? AND created_at > ? AND created_at < ?
        LIMIT 1
        '''
        cursor.execute(query_like,
                       (track_user_id, track_comment_id, start_time, end_time))
        if cursor.fetchone():
            return "like_comment"

        # 检查在时间区间内是否有dislike的记录
        query_dislike = '''
        SELECT 1
        FROM comment_dislike
        WHERE user_id = ? AND comment_id = ? AND created_at > ? AND created_at < ?
        LIMIT 1
        '''
        cursor.execute(query_dislike,
                       (track_user_id, track_comment_id, start_time, end_time))
        if cursor.fetchone():
            return "dislike_comment"

        # 如果没有找到任何记录，返回'no action'
        return "no action"


def get_result(comment_id_lst, db_path, exp_start_time):
    db = Database(db_path)

    like_lst, dislike_lst, no_action_lst = [], [], []

    for track_comment_id in comment_id_lst:
        # print(track_comment_id)
        start_time = exp_start_time
        # print(start_time)
        while True:
            read_result = db.find_user_by_comment_id(track_comment_id,
                                                     start_time)
            # print(read_result)
            # 这条评论没有被阅读过，不计入分析
            if not read_result:
                break

            # 从之前的start_time开始，第一个看到comment的user_id和看到的时间
            track_user_id, start_time = read_result

            ope_type = db.track_ope_before_next_refresh(
                track_user_id, start_time, track_comment_id)
            # print(ope_type, track_comment_id, track_user_id)

            if ope_type == "like_comment":
                like_lst.append((track_user_id, track_comment_id))
                break
            elif ope_type == "dislike_comment":
                dislike_lst.append((track_user_id, track_comment_id))
                break
            else:
                # 如果没有操作，可以继续追踪其他用户的反应
                no_action_lst.append((track_user_id, track_comment_id))
                break
    # 记录对应操作的user_id和comment_id
    return like_lst, dislike_lst, no_action_lst


def analyze_probability(data,
                        save_file_path,
                        action_type='like',
                        figsize=(8, 6)):
    # 计算概率
    probabilities = {}
    for key, value in data.items():
        total = sum(value.values())
        probabilities[key] = value[action_type] / total if total > 0 else 0

    # 计算95%置信区间
    conf_intervals = {}
    for key, value in data.items():
        n = sum(value.values())
        p = probabilities[key]
        z = norm.ppf(0.975)  # 95%置信水平对应的z值
        se = np.sqrt(p * (1 - p) / n)
        conf_intervals[key] = (max(0, p - z * se), min(1, p + z * se))

    # 准备绘图数据
    labels = list(probabilities.keys())
    probs = [probabilities[key] for key in labels]
    conf_ints = [conf_intervals[key] for key in labels]

    # 计算yerr
    yerr = [[prob - low for prob, (low, high) in zip(probs, conf_ints)],
            [high - prob for prob, (low, high) in zip(probs, conf_ints)]]

    # 绘制概率和置信区间
    plt.figure(figsize=figsize)
    plt.bar(labels,
            probs,
            color='skyblue',
            label=f'{action_type.capitalize()} Probability')
    plt.errorbar(labels, probs, yerr=yerr, fmt='o', color='red', capsize=5)

    # 在control组的概率处加一条虚线
    control_probability = probabilities['control']
    plt.axhline(y=control_probability,
                color='gray',
                linestyle='--',
                label='Control Probability')

    plt.xlabel('Comment Type')
    plt.ylabel(f'Probability of {action_type.capitalize()}s')
    plt.title(
        f'Probability of {action_type.capitalize()}s with 95% Confidence Intervals'
    )
    plt.legend()

    # 调整y轴范围以便更清晰地展示完整的置信区间
    y_min = min(prob - low
                for prob, (low, high) in zip(probs, conf_ints)) - 0.05
    y_max = max(high for _, (low, high) in zip(probs, conf_ints)) + 0.05
    plt.ylim(max(0, y_min), min(1, y_max))

    plt.savefig(save_file_path)
    plt.show()


def main(exp_info_file_path, db_path, exp_name, folder_path):
    with open(exp_info_file_path, 'r') as file:
        exp_info = json.load(file)

    exp_start_time = datetime(2024, 8, 6, 8, 0)

    like_up, dislike_up, no_action_up = get_result(exp_info["up_comment_id"],
                                                   db_path, exp_start_time)
    like_down, dislike_down, no_action_down = get_result(
        exp_info["down_comment_id"], db_path, exp_start_time)
    like_control, dislike_control, no_action_control = get_result(
        exp_info["control_comment_id"], db_path, exp_start_time)

    data = {
        'downvote': {
            'like': len(like_down),
            'dislike': len(dislike_down),
            'no_action': len(no_action_down)
        },
        'control': {
            'like': len(like_control),
            'dislike': len(dislike_control),
            'no_action': len(no_action_control)
        },
        'upvote': {
            'like': len(like_up),
            'dislike': len(dislike_up),
            'no_action': len(no_action_up)
        }
    }

    print(data)

    # 分析并绘制like的概率
    analyze_probability(data,
                        action_type='like',
                        save_file_path=f'{folder_path}/like_{exp_name}.png')

    # 分析并绘制dislike的概率
    analyze_probability(data,
                        action_type='dislike',
                        save_file_path=f'{folder_path}/dislike_{exp_name}.png')


if __name__ == "__main__":
    main(exp_info_file_path=(
        './experiments/reddit_herding_effect/results_analysis/'
        'result_data/exp_info.json'),
         db_path=('./experiments/reddit_herding_effect/results_analysis/'
                  'result_data/mock_reddit_06-30_06-33-29.db'))
