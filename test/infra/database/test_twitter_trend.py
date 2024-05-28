# File: ./test/infra/test_twitter_trend.py
import os
import os.path as osp
import sqlite3
from datetime import timedelta

import pytest

from twitter.twitter import Twitter  # 确保从你的模块中导入Twitter类

parent_folder = osp.dirname(osp.abspath(__file__))
test_db_filepath = osp.join(parent_folder, "test.db")


class MockChannel:

    def __init__(self):
        self.call_count = 0
        self.messages = []  # 用于存储发送的消息

    async def receive_from(self):
        # 第一次调用返回搜索用户的指令
        if self.call_count == 0:
            self.call_count += 1
            return ('id_', (1, None, "trend"))
        else:
            return ('id_', (None, None, "exit"))

    async def send_to(self, message):
        self.messages.append(message)  # 存储消息以便后续断言
        # 对搜索用户的结果进行断言
        if self.call_count == 1:
            # 验证搜索成功且找到至少一个匹配用户
            assert message[2]["success"] is True, "Trend should be successful"
            assert message[2]["tweets"][0]["content"] == "Tweet 6"
            assert message[2]["tweets"][1]["content"] == "Tweet 5"
            assert message[2]["tweets"][2]["content"] == "Tweet 4"


# 定义一个fixture来初始化数据库和Twitter实例
@pytest.fixture
def setup_twitter():
    # 测试前确保test.db不存在
    if os.path.exists(test_db_filepath):
        os.remove(test_db_filepath)

    # 创建数据库和表
    db_path = test_db_filepath

    # 初始化Twitter实例
    mock_channel = MockChannel()
    twitter_instance = Twitter(db_path, mock_channel)
    return twitter_instance


@pytest.mark.asyncio
async def test_search_user(setup_twitter):
    try:
        twitter = setup_twitter

        # 在测试开始之前，将1个用户插入到user表中
        conn = sqlite3.connect(test_db_filepath)
        cursor = conn.cursor()
        cursor.execute(
            ("INSERT INTO user "
             "(user_id, agent_id, user_name, num_followings, num_followers) "
             "VALUES (?, ?, ?, ?, ?)"), (1, 1, "user1", 0, 0))
        conn.commit()

        # 在测试开始之前，将tweet插入到tweet表中
        conn = sqlite3.connect(test_db_filepath)
        cursor = conn.cursor()

        today = twitter.start_time
        # 生成从今天开始往前数10天的时间戳列表
        tweets_info = [
            (1, f'Tweet {9-i}',
             (today - timedelta(days=9 - i)).strftime('%Y-%m-%d %H:%M:%S.%f'),
             9 - i) for i in range(10)
        ]

        cursor.executemany(
            "INSERT INTO tweet (user_id, content, created_at, num_likes) "
            "VALUES (?, ?, ?, ?)", tweets_info)
        conn.commit()

        await twitter.running()

        # 验证跟踪表(trace)是否正确记录了操作
        cursor.execute("SELECT * FROM trace WHERE action='trend'")
        assert cursor.fetchone() is not None, "trend action not traced"

    finally:
        conn.close()
        # 清理
        # if os.path.exists(test_db_filepath):
        #     os.remove(test_db_filepath)
