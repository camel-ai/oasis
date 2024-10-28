import os
import os.path as osp
import sqlite3

import pytest

from social_simulation.social_platform.platform import Platform

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
            return ("id_", (1, ("alice0101", "Alice", "A girl."), "sign_up"))
        # 第二次调用返回点赞操作的指令
        elif self.call_count == 1:
            self.call_count += 1
            return ("id_", (2, ("bubble", "Bob", "A boy."), "sign_up"))
        elif self.call_count == 2:
            self.call_count += 1
            return ("id_", (1, "This is a test post", "create_post"))
        # elif self.call_count == 3:
        #     self.call_count += 1
        #     return ('id_', (3, "This is a test post", "create_post"))
        else:
            return ("id_", (None, None, "exit"))

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
            assert message[2]["success"] is True
            assert "post_id" in message[2]
        # elif self.call_count == 4:
        #     assert message[2]["success"] is False
        #     assert message[2]["error"] == (
        #         "Agent 3 have not signed up and have no user id.")


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
async def test_signup_create_post(setup_platform):
    try:
        platform = setup_platform

        await platform.running()

        # 验证数据库中是否正确插入了数据
        conn = sqlite3.connect(test_db_filepath)
        cursor = conn.cursor()

        # 验证推文表(post)是否正确插入了数据
        cursor.execute("SELECT * FROM user")
        users = cursor.fetchall()
        assert len(users) == 2
        assert users[0][1] == 1
        assert users[0][3] == "Alice"
        assert users[1][1] == 2
        assert users[1][3] == "Bob"

        # 验证推文表(post)是否正确插入了数据
        cursor.execute("SELECT * FROM post")
        posts = cursor.fetchall()
        assert len(posts) == 1
        post = posts[0]
        assert post[1] == 1  # 假设用户ID是1
        assert post[2] == "This is a test post"
        assert post[4] == 0

        # 验证跟踪表(trace)是否正确记录了创建注册操作
        cursor.execute("SELECT * FROM trace WHERE action ='sign_up'")
        results = cursor.fetchall()
        assert len(results) == 2

        # 验证跟踪表(trace)是否正确记录了创建推文操作
        cursor.execute("SELECT * FROM trace WHERE action='create_post'")
        assert cursor.fetchone() is not None, "Create post action not traced"

    finally:
        pass
        conn.close()
        if os.path.exists(test_db_filepath):
            os.remove(test_db_filepath)
