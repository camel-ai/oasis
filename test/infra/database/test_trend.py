import os
import os.path as osp
import sqlite3
from datetime import datetime, timedelta

import pytest

from social_simulation.social_platform.platform import Platform

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
            print(message[2])
            # 验证搜索成功且找到至少一个匹配用户
            assert message[2]["success"] is True, "Trend should be successful"
            assert message[2]["posts"][0]["content"] == "Post 6"
            assert message[2]["posts"][1]["content"] == "Post 5"
            assert message[2]["posts"][2]["content"] == "Post 4"
            print(message[2]["posts"])


@pytest.fixture
def setup_platform():
    # 测试前确保test.db不存在
    if os.path.exists(test_db_filepath):
        os.remove(test_db_filepath)

    # 创建数据库和表
    db_path = test_db_filepath

    mock_channel = MockChannel()
    instance = Platform(db_path, mock_channel)
    return instance


@pytest.mark.asyncio
async def test_search_user(setup_platform):
    try:
        platform = setup_platform

        # 在测试开始之前，将1个用户插入到user表中
        conn = platform.db
        cursor = conn.cursor()
        cursor.execute(
            ("INSERT INTO user "
             "(user_id, agent_id, user_name, num_followings, num_followers) "
             "VALUES (?, ?, ?, ?, ?)"), (1, 1, "user1", 0, 0))
        conn.commit()

        # 在测试开始之前，将post插入到post表中
        conn = platform.db
        cursor = conn.cursor()

        today = platform.start_time
        # 生成从今天开始往前数10天的时间戳列表
        posts_info = [
            (1, f'Post {9-i}',
             (today - timedelta(days=9 - i)).strftime('%Y-%m-%d %H:%M:%S.%f'),
             (9 - i), 0) for i in range(10)
        ]

        cursor.executemany(
            "INSERT INTO post (user_id, content, created_at, num_likes, "
            "num_dislikes) VALUES (?, ?, ?, ?, ?)", posts_info)
        conn.commit()

        comments_info = [(i + 1, 1, 'Comment', datetime.now())
                         for i in range(10)]

        cursor.executemany(
            "INSERT INTO comment (post_id, user_id, content, created_at) "
            "VALUES (?, ?, ?, ?)", comments_info)
        conn.commit()

        await platform.running()

        # 验证跟踪表(trace)是否正确记录了操作
        cursor.execute("SELECT * FROM trace WHERE action='trend'")
        assert cursor.fetchone() is not None, "trend action not traced"

    finally:
        conn.close()
        # 清理
        if os.path.exists(test_db_filepath):
            os.remove(test_db_filepath)
