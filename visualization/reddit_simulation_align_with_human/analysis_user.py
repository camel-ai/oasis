import sqlite3

import matplotlib.pyplot as plt
import numpy as np


def main(db_path):
    # db的文件名代表实验开始的时间
    file_name = db_path[:-3]
    # 图像保存路径
    image_path = f"analysis_result/user_like_{file_name}.png"

    # 连接到SQLite数据库
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # 修改查询：返回用户ID，并排除ID为0和1的用户
    query = """
    SELECT u.user_id,
        COALESCE(l.likes, 0) AS likes,
        COALESCE(d.dislikes, 0) AS dislikes
    FROM user u
    LEFT JOIN (
        SELECT user_id, COUNT(*) AS likes
        FROM comment_like
        GROUP BY user_id
    ) l ON u.user_id = l.user_id
    LEFT JOIN (
        SELECT user_id, COUNT(*) AS dislikes
        FROM comment_dislike
        GROUP BY user_id
    ) d ON u.user_id = d.user_id
    WHERE u.user_id > 2
    ORDER BY u.user_id ASC
    """

    cursor.execute(query)
    data = cursor.fetchall()

    # 关闭数据库连接
    conn.close()

    # 解析查询结果
    user_ids = [row[0] for row in data]
    likes = [row[1] for row in data]
    dislikes = [row[2] for row in data]

    # 计算均值和方差
    likes_mean = np.mean(likes)
    likes_var = np.var(likes)
    dislikes_mean = np.mean(dislikes)
    dislikes_var = np.var(dislikes)

    # 绘图
    fig, ax = plt.subplots(figsize=(12, 8))

    # 设置位置和宽度
    x = np.arange(len(user_ids))  # 使用用户ID作为x轴
    width = 0.35

    # 绘制条形图
    ax.bar(x - width / 2, likes, width, label="Likes", color="blue")
    ax.bar(x + width / 2, dislikes, width, label="Dislikes", color="red")

    # 添加均值和方差线条
    ax.axhline(likes_mean, color="blue", linestyle="dashed", linewidth=1)
    ax.text(
        0,
        likes_mean * 1.1,
        f"Mean: {likes_mean:.2f}, Var: {likes_var:.2f}",
        color="blue",
    )

    ax.axhline(dislikes_mean, color="red", linestyle="dashed", linewidth=1)
    ax.text(
        0,
        dislikes_mean * 1.1,
        f"Mean: {dislikes_mean:.2f}, Var: {dislikes_var:.2f}",
        color="red",
    )

    # 调整y轴的上限，以便清晰显示均值和方差线
    y_max = max(max(likes), max(dislikes)) * 1.2
    ax.set_ylim(0, y_max)

    # 添加一些文本标签
    ax.set_xlabel("User ID")
    ax.set_ylabel("Frequency")
    ax.set_title("Likes and Dislikes by User ID with Mean and Variance")
    ax.set_xticks(x)
    ax.set_xticklabels(user_ids, rotation=45, ha="right")
    ax.legend()

    # 保存图像
    plt.tight_layout()
    plt.savefig(image_path)

    # 显示图像
    plt.show()


if __name__ == "__main__":
    main(db_path=("./experiments/reddit_herding_effect/results_analysis/"
                  "result_data/mock_reddit_06-30_06-33-29.db"))
