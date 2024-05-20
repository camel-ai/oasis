import asyncio
import os.path as osp
import os
import sqlite3
import random
import pytest

from datetime import datetime
from twitter.channel import Twitter_Channel
from twitter.twitter import Twitter
from twitter.typing import ActionType
from test.show_db import print_db_contents


parent_folder = osp.dirname(osp.abspath(__file__))
test_db_filepath = osp.join(parent_folder, "test.db")


# 定义一个fixture来初始化数据库和Twitter实例
@pytest.fixture
def setup_db():
    # 测试前确保test.db不存在
    if os.path.exists(test_db_filepath):
        os.remove(test_db_filepath)


@pytest.mark.asyncio
async def test_update_rec_table(setup_db):
    channel = Twitter_Channel()
    infra = Twitter(test_db_filepath, channel)
    # 在测试开始之前，将3个用户插入到user表中
    conn = sqlite3.connect(test_db_filepath)
    cursor = conn.cursor()
    cursor.execute(
        ("INSERT INTO user "
            "(agent_id, user_name, num_followings, num_followers) "
            "VALUES (?, ?, ?, ?)"),
        (1, "user1", 0, 0)
    )
    cursor.execute(
        ("INSERT INTO user "
            "(agent_id, user_name, num_followings, num_followers) "
            "VALUES (?, ?, ?, ?)"),
        (2, "user2", 2, 4)
    )
    cursor.execute(
        ("INSERT INTO user "
            "(agent_id, user_name, num_followings, num_followers) "
            "VALUES (?, ?, ?, ?)"),
        (3, "user3", 3, 5)
    )
    conn.commit()

    # 在测试开始之前，将60条推文用户插入到tweet表中
    for i in range(1, 61):  # 生成60条tweet
        user_id = i % 3 + 1  # 循环使用用户ID 1, 2, 3
        content = f"Tweet content for tweet {i}"  # 简单生成不同的内容
        created_at = datetime.now()
        num_likes = random.randint(0, 100)  # 随机生成点赞数

        cursor.execute(
            ("INSERT INTO tweet "
             "(user_id, content, created_at, num_likes) "
             "VALUES (?, ?, ?, ?)"),
            (user_id, content, created_at, num_likes)
        )
    conn.commit()

    task = asyncio.create_task(infra.running())
    await channel.write_to_receive_queue((None, None, ActionType.UPDATE_REC))
    await channel.write_to_receive_queue((None, None, ActionType.EXIT))
    await task

    for i in range(1, 4):
        cursor.execute(
            "SELECT tweet_id FROM rec WHERE user_id = ?", (i,))
        tweets = cursor.fetchall()  # 获取所有记录
        assert len(tweets) == 50, f"User {user_id} does not have 50 tweets."
        tweet_ids = [tweet[0] for tweet in tweets]
        is_unique = len(tweet_ids) == len(set(tweet_ids))
        print(set(tweet_ids))
        assert is_unique, f"User {user_id} has duplicate tweet_ids."

    print_db_contents(test_db_filepath)
