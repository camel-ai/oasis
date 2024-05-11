'''
测试通过：create_tweet, like, unlike 函数
'''

import os
import sqlite3

import pytest

import os.path as osp
from SandboxTwitterModule.infra import Twitter  # 确保从你的模块中导入Twitter类


parent_folder = osp.dirname(osp.abspath(__file__))
test_db_filepath = osp.join(parent_folder, "test.db")


# 定义一个fixture来初始化数据库和Twitter实例
@pytest.fixture
def setup_twitter():
    # 测试前确保test.db不存在
    if os.path.exists(test_db_filepath):
        os.remove(test_db_filepath)

    # 创建数据库和表
    db_path = test_db_filepath

    # 初始化Twitter实例
    twitter_instance = Twitter(db_path)
    return twitter_instance


class MockChannel:

    def __init__(self):
        self.call_count = 0
        self.messages = []  # 用于存储发送的消息

    async def receive_from(self, group):
        # 第一次调用返回创建推文的指令
        if self.call_count == 0:
            self.call_count += 1
            return 1, "This is a test tweet", "create_tweet"
        # 第二次调用返回点赞操作的指令
        elif self.call_count == 1:
            self.call_count += 1
            return 1, 1, "like"
        elif self.call_count == 2:
            self.call_count += 1
            return 2, 1, "like"
        elif self.call_count == 3:
            self.call_count += 1
            return 2, 1, "unlike"
        # 返回退出指令
        else:
            return None, None, "exit"

    async def send_to(self, group, message):
        self.messages.append(message)  # 存储消息以便后续断言
        if self.call_count == 1:
            # 对创建推文的成功消息进行断言
            assert message[1]["success"] is True
            assert "tweet_id" in message[1]
        elif self.call_count == 2:
            # 对点赞操作的成功消息进行断言
            assert message[1]["success"] is True
            assert "like_id" in message[1]
        elif self.call_count == 3:
            assert message[1]["success"] is True
            assert "like_id" in message[1]
        elif self.call_count == 4:
            assert message[1]["success"] is True
            assert "like_id" in message[1]


@pytest.mark.asyncio
async def test_create_like_unlike_tweet(setup_twitter):
    try:
        twitter = setup_twitter
        mock_channel = MockChannel()

        # 在测试开始之前，将2个用户插入到user表中
        conn = sqlite3.connect(test_db_filepath)
        cursor = conn.cursor()
        cursor.execute(
            ("INSERT INTO user "
             "(user_id, agent_id, user_name, num_followings, num_followers) "
             "VALUES (?, ?, ?, ?, ?)"),
            (1, 1, "user1", 0, 0)
        )
        cursor.execute(
            ("INSERT INTO user "
             "(user_id, agent_id, user_name, num_followings, num_followers) "
             "VALUES (?, ?, ?, ?, ?)"),
            (2, 2, "user2", 2, 4)
        )
        conn.commit()

        await twitter.running(mock_channel)

        # 验证数据库中是否正确插入了数据
        conn = sqlite3.connect(test_db_filepath)
        cursor = conn.cursor()

        # 验证推文表(tweet)是否正确插入了数据
        cursor.execute("SELECT * FROM tweet")
        tweets = cursor.fetchall()
        assert len(tweets) == 1
        tweet = tweets[0]
        assert tweet[1] == 1  # 假设用户ID是1
        assert tweet[2] == "This is a test tweet"
        assert tweet[4] == 1  # num_likes

        # 验证推文表(tweet)是否正确插入了数据
        cursor.execute("SELECT * FROM like")
        likes = cursor.fetchall()
        assert len(likes) == 1

        # 验证跟踪表(trace)是否正确记录了创建推文和点赞操作
        cursor.execute("SELECT * FROM trace WHERE action='create_tweet'")
        assert cursor.fetchone() is not None, "Create tweet action not traced"

        cursor.execute("SELECT * FROM trace WHERE action='like'")
        results = cursor.fetchall()
        assert results is not None, "Like tweet action not traced"
        assert len(results) == 2

        cursor.execute("SELECT * FROM trace WHERE action='unlike'")
        results = cursor.fetchall()
        assert results is not None, "Unlike tweet action not traced"
        assert results[0][0] == 2  # `user_id`
        assert results[0][-1] == "{'tweet_id': 1, 'like_id': 2}"

        # 验证点赞表(like)是否正确插入了数据
        cursor.execute("SELECT * FROM like WHERE tweet_id=1 AND user_id=1")
        assert cursor.fetchone() is not None, "Like record not found"

    finally:
        # 清理
        conn.close()
        if os.path.exists(test_db_filepath):
            os.remove(test_db_filepath)
