# File: test_user_signup.py
import os
import sqlite3

import pytest

import os.path as osp
from SandboxTwitterModule.infra import Twitter  # 确保从你的模块中导入Twitter类


parent_folder = osp.dirname(osp.abspath(__file__))
test_db_filepath = osp.join(parent_folder, "test.db")


class MockChannel:

    def __init__(self):
        self.call_count = 0
        self.messages = []  # 用于存储发送的消息

    async def receive_from(self):
        # 第一次调用返回创建推文的指令
        if self.call_count == 0:
            self.call_count += 1
            return ('id_', (1, ("alice0101", "Alice", "A girl."), "sign_up"))
        # 第二次调用返回点赞操作的指令
        elif self.call_count == 1:
            self.call_count += 1
            return ('id_', (2, ("bubble", "Bob", "A boy."), "sign_up"))
        elif self.call_count == 2:
            self.call_count += 1
            return ('id_', (1, ("alan", "Alan", "A kid."), "sign_up"))
        # 返回退出指令
        else:
            return ('id_', (None, None, "exit"))

    async def send_to(self, message):
        self.messages.append(message)  # 存储消息以便后续断言
        if self.call_count == 1:
            # 对创建推文的成功消息进行断言
            assert message[2]["success"] is True
            assert "user_id" in message[2]
        elif self.call_count == 2:
            # 对点赞操作的成功消息进行断言
            assert message[2]["success"] is True
            assert "user_id" in message[2]
        elif self.call_count == 3:
            assert message[2]["success"] is False
            assert "error" in message[2]
            assert message[2]["error"] == (
                "Agent 1 have already signed up with user id: 1"
            )


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
async def test_create_like_unlike_tweet(setup_twitter):
    try:
        twitter = setup_twitter

        await twitter.running()

        # 验证数据库中是否正确插入了数据
        conn = sqlite3.connect(test_db_filepath)
        cursor = conn.cursor()

        # 验证推文表(tweet)是否正确插入了数据
        cursor.execute("SELECT * FROM user")
        users = cursor.fetchall()
        assert len(users) == 2
        assert users[0][1] == 1
        assert users[0][2] == "alice0101"
        assert users[0][3] == "Alice"
        assert users[0][4] == "A girl."
        assert users[1][1] == 2
        assert users[1][2] == "bubble"
        assert users[1][3] == "Bob"
        assert users[1][4] == "A boy."

        # 验证跟踪表(trace)是否正确记录了创建推文和点赞操作
        cursor.execute("SELECT * FROM trace WHERE action ='sign_up'")
        results = cursor.fetchall()
        assert len(results) == 2

    finally:
        conn.close()
        if os.path.exists(test_db_filepath):
            os.remove(test_db_filepath)
