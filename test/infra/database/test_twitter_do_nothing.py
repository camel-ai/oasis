# File: ./test/infra/test_twitter_refresh.py
import os
import os.path as osp
import sqlite3

import pytest

from twitter.twitter import Twitter  # 确保从你的模块中导入Twitter类
from twitter.typing import ActionType

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
            return ('id_', (1, None, ActionType.DO_NOTHING))
        else:
            return ('id_', (None, None, ActionType.EXIT))

    async def send_to(self, message):
        self.messages.append(message)
        if self.call_count == 1:
            assert message[2]["success"] is True


# 定义一个fixture来初始化数据库和Twitter实例
@pytest.fixture
def setup_twitter():
    # 测试前确保test.db不存在
    if os.path.exists(test_db_filepath):
        os.remove(test_db_filepath)

    # 创建数据库和表
    db_path = test_db_filepath
    mock_channel = MockChannel()
    # 初始化Twitter实例
    twitter_instance = Twitter(db_path, mock_channel)
    return twitter_instance


@pytest.mark.asyncio
async def test_refresh(setup_twitter):
    try:
        twitter = setup_twitter

        # 在测试开始之前，将1个用户插入到user表中
        conn = sqlite3.connect(test_db_filepath)
        cursor = conn.cursor()
        cursor.execute(
            ("INSERT INTO user (user_id, agent_id, user_name, bio, "
             "num_followings, num_followers) VALUES (?, ?, ?, ?, ?, ?)"),
            (1, 1, "user1", "This is test bio for user 1", 0, 0))
        conn.commit()

        await twitter.running()
        # 验证跟踪表(trace)是否正确记录了操作
        cursor.execute("SELECT * FROM trace WHERE action='do_nothing'")
        assert cursor.fetchone() is not None, "trend action not traced"

    finally:
        conn.close()
        # 清理
        if os.path.exists(test_db_filepath):
            os.remove(test_db_filepath)
