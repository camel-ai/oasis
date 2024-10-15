import os
import os.path as osp
import sqlite3
from datetime import datetime

import pytest

from social_simulation.social_platform.platform import Platform
from social_simulation.social_platform.typing import ActionType
from social_simulation.testing.show_db import print_db_contents

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
            return ('id_', (None, None, ActionType.UPDATE_REC_TABLE))
        if self.call_count == 1:
            self.call_count += 1
            return ('id_', (0, None, ActionType.REFRESH))
        else:
            return ('id_', (None, None, ActionType.EXIT))

    async def send_to(self, message):
        self.messages.append(message)  # 存储消息以便后续断言
        # 对refresh的结果进行断言
        if self.call_count == 2:
            # 验证refresh成功
            # print_db_contents(test_db_filepath)
            print(self.messages)
            assert message[2]["success"] is True
            assert len(message[2]["posts"]) == 1
            print(message[2]["posts"])
            # 然后检查 'posts' 列表中的每个条目
            for post in message[2].get('posts', []):
                assert post.get('post_id') is not None
                assert post.get('user_id') is not None
                assert post.get('content') is not None
                assert post.get('created_at') is not None
                assert post.get('num_likes') is not None
                assert post.get('comments') is not None


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
async def test_refresh(setup_platform):
    try:
        platform = setup_platform

        # 在测试开始之前，将1个用户插入到user表中
        conn = sqlite3.connect(test_db_filepath)
        cursor = conn.cursor()
        cursor.execute(
            ("INSERT INTO user (user_id, agent_id, user_name, bio, "
             "num_followings, num_followers) VALUES (?, ?, ?, ?, ?, ?)"),
            (0, 0, "user0", "This is test bio for user 0", 0, 0))
        conn.commit()

        # 在测试开始之前，将post插入到post表中
        conn = sqlite3.connect(test_db_filepath)
        cursor = conn.cursor()

        # 在测试开始之前，将60条推文用户插入到post表中
        for i in range(60):  # 生成60条post
            user_id = i % 3  # 循环使用用户ID 0, 1, 2
            content = f"Post content for post {i}"  # 简单生成不同的内容
            comment_content = f"Comment content for post {i}"
            created_at = datetime.now()

            cursor.execute(("INSERT INTO post (user_id, content, created_at, "
                            "num_likes, num_dislikes) VALUES (?, ?, ?, ?, ?)"),
                           (user_id, content, created_at, 0, 0))
            cursor.execute(("INSERT INTO comment (post_id, user_id, content, "
                            "created_at) VALUES (?, ?, ?, ?)"),
                           (i, user_id, comment_content, created_at))
        conn.commit()
        # print_db_contents(test_db_filepath)
        await platform.running()
        # 验证跟踪表(trace)是否正确记录了操作
        cursor.execute("SELECT * FROM trace WHERE action='refresh'")
        assert cursor.fetchone() is not None, "refresh action not traced"

    finally:
        conn.close()
        # 清理
        if os.path.exists(test_db_filepath):
            os.remove(test_db_filepath)
