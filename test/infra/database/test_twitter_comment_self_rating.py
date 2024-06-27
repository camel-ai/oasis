# File: ./test/infra/test_twitter_tweet.py
import os
import os.path as osp
import sqlite3

import pytest

from twitter.twitter import Twitter  # 确保从你的模块中导入Twitter类

parent_folder = osp.dirname(osp.abspath(__file__))
test_db_filepath = osp.join(parent_folder, "test.db")


class MockChannel:

    def __init__(self):
        self.call_count = 0
        self.messages = []  # 用于存储发送的消息

    async def receive_from(self):
        # 第1次调用返回创建推文的指令
        if self.call_count == 0:
            self.call_count += 1
            return ('id_', (1, "Test tweet", "create_tweet"))
        # 第2次调用返回创建评论的指令
        if self.call_count == 1:
            self.call_count += 1
            return ('id_', (1, (1, "Test Comment"), "create_comment"))
        # 第3次调用返回点赞评论的指令
        if self.call_count == 2:
            self.call_count += 1
            return ('id_', (1, 1, "like_comment"))
        if self.call_count == 3:
            self.call_count += 1
            return ('id_', (2, 1, "like_comment"))
        if self.call_count == 4:
            self.call_count += 1
            return ('id_', (1, 1, "dislike_comment"))
        if self.call_count == 5:
            self.call_count += 1
            return ('id_', (2, 1, "dislike_comment"))
        else:
            return ('id_', (None, None, "exit"))

    async def send_to(self, message):
        self.messages.append(message)  # 存储消息以便后续断言
        if self.call_count == 1:
            # 对创建推文的成功消息进行断言
            assert message[2]["success"] is True
            assert "tweet_id" in message[2]
        elif self.call_count == 2:
            # 对点赞操作的成功消息进行断言
            assert message[2]["success"] is True
            assert "comment_id" in message[2]
        elif self.call_count == 3:
            assert message[2]["success"] is False
            assert message[2]["error"] == (
                'Users are not allowed to like/dislike their own comments.')
        elif self.call_count == 4:
            # 对点赞操作的成功消息进行断言
            assert message[2]["success"] is True
            assert "comment_like_id" in message[2]
        elif self.call_count == 5:
            # 对取消点赞操作的成功消息进行断言
            assert message[2]["success"] is False
            assert message[2]["error"] == (
                'Users are not allowed to like/dislike their own comments.')
        elif self.call_count == 6:
            # 对点赞操作的成功消息进行断言
            assert message[2]["success"] is True
            assert "comment_dislike_id" in message[2]


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
    twitter_instance = Twitter(db_path, mock_channel, allow_self_rating=False)
    return twitter_instance


@pytest.mark.asyncio
async def test_comment(setup_twitter):
    try:
        twitter = setup_twitter

        # 在测试开始之前，将2个用户插入到user表中
        conn = sqlite3.connect(test_db_filepath)
        cursor = conn.cursor()
        cursor.execute(
            ("INSERT INTO user "
             "(user_id, agent_id, user_name, num_followings, num_followers) "
             "VALUES (?, ?, ?, ?, ?)"), (1, 1, "user1", 0, 0))
        cursor.execute(
            ("INSERT INTO user "
             "(user_id, agent_id, user_name, num_followings, num_followers) "
             "VALUES (?, ?, ?, ?, ?)"), (2, 2, "user2", 2, 4))
        conn.commit()

        await twitter.running()

    finally:
        # 清理
        conn.close()
        if os.path.exists(test_db_filepath):
            os.remove(test_db_filepath)
