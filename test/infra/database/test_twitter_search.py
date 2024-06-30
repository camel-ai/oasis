# File: ./test/infra/test_twitter_search.py
import os
import os.path as osp
import sqlite3

import pytest

from social_platform.twitter import Twitter  # 确保从你的模块中导入Twitter类

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
            return ('id_', (1, "bob", "search_user"))  # 假设搜索关键词为"bob"
        if self.call_count == 1:
            self.call_count += 1
            return ('id_', (2, "Bob", "search_tweets"))  # 假设搜索关键词为"bob"
        # 后续调用返回退出指令
        else:
            return ('id_', (None, None, "exit"))

    async def send_to(self, message):
        self.messages.append(message)  # 存储消息以便后续断言
        # 对搜索用户的结果进行断言
        if self.call_count == 1:
            # 验证搜索成功且找到至少一个匹配用户
            assert message[2]["success"] is True, "Search should be successful"
            assert len(
                message[2]["users"]) > 0, "Should find at least one user"
            # 你可以添加更多的断言来验证返回的用户信息是否正确
            assert message[2]["users"][0]["user_name"] == \
                   "user2", "The first matching user should be 'user2'"
        if self.call_count == 2:
            assert message[2]["success"] is True, "Search should be successful"
            assert len(
                message[2]["tweets"]) > 0, "Should find at least one tweet"
            assert message[2]["tweets"][0]["content"] == "Bob's first tweet!"
            assert message[2]["tweets"][0]['comments'][0]['content'] == (
                "Alice's comment")
            assert message[2]["tweets"][0]['comments'][1]['content'] == (
                "Bob's comment")
            assert message[2]["tweets"][0]['comments'][2]['content'] == (
                "Charlie's comment")


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

        # 在测试开始之前，将几个用户插入到user表中
        conn = sqlite3.connect(test_db_filepath)
        cursor = conn.cursor()
        users_info = [(1, 1, "user1", "Alice", "Bio of Alice",
                       "2023-01-01 12:00:00", 10, 5),
                      (2, 2, "user2", "Bob", "Bio of Bob",
                       "2023-01-02 12:00:00", 15, 8),
                      (3, 3, "user3", "Charlie", "Bio of Charlie",
                       "2023-01-03 12:00:00", 20, 12)]
        cursor.executemany("INSERT INTO user VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                           users_info)
        tweets_info = [
            # (user_id, content, created_at, num_likes)
            (1, "Hello World from Alice!", "2023-01-01 13:00:00", 100, 2),
            (2, "Bob's first tweet!", "2023-01-02 14:00:00", 150, 2),
            (3, "Charlie says hi!", "2023-01-03 15:00:00", 200, 3)
        ]
        # 假设 cursor 是你已经创建并与数据库建立连接的游标对象
        cursor.executemany(
            ("INSERT INTO tweet (user_id, content, created_at, num_likes, "
             "num_dislikes) VALUES (?, ?, ?, ?, ?)"), tweets_info)
        conn.commit()

        comments_info = [
            # (tweet_id, user_id, content)
            (2, 1, "Alice's comment", "2023-01-01 13:05:00", 5, 1),
            (2, 2, "Bob's comment", "2023-01-02 14:10:00", 3, 0),
            (2, 3, "Charlie's comment", "2023-01-03 15:20:00", 8, 2)
        ]
        # 假设 cursor 是你已经创建并与数据库建立连接的游标对象
        cursor.executemany(
            "INSERT INTO comment (tweet_id, user_id, content, created_at, "
            "num_likes, num_dislikes) VALUES (?, ?, ?, ?, ?, ?)",
            comments_info)
        conn.commit()

        await twitter.running()

        # 验证跟踪表(trace)是否正确记录了操作
        cursor.execute("SELECT * FROM trace WHERE action='search_user'")
        assert cursor.fetchone() is not None, "search_user action not traced"

        # 验证跟踪表(trace)是否正确记录了操作
        cursor.execute("SELECT * FROM trace WHERE action='search_tweets'")
        assert cursor.fetchone() is not None, "search_tweet action not traced"

    finally:
        conn.close()
        # 清理
        if os.path.exists(test_db_filepath):
            os.remove(test_db_filepath)
