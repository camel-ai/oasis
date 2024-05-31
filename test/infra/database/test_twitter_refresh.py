# File: ./test/infra/test_twitter_refresh.py
import os
import os.path as osp
import sqlite3
from datetime import datetime
from test.show_db import print_db_contents

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
            return ('id_', (None, None, ActionType.UPDATE_REC))
        if self.call_count == 1:
            self.call_count += 1
            return ('id_', (1, None, ActionType.REFRESH))
        else:
            return ('id_', (None, None, ActionType.EXIT))

    async def send_to(self, message):
        self.messages.append(message)  # 存储消息以便后续断言
        # 对搜索用户的结果进行断言
        if self.call_count == 2:
            # 验证搜索成功且找到至少一个匹配用户
            print_db_contents(test_db_filepath)
            assert message[2]["success"] is True, "Trend should be successful"
            assert len(message[2]["tweets"]) == 5

            # 然后检查 'tweets' 列表中的每个条目
            for tweet in message[2].get('tweets', []):
                assert tweet.get('tweet_id') is not None
                assert tweet.get('user_id') is not None
                assert tweet.get('content') is not None
                assert tweet.get('created_at') is not None
                assert tweet.get('num_likes') is not None


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
            ("INSERT INTO user "
             "(user_id, agent_id, user_name, bio,  num_followings, num_followers) "
             "VALUES (?, ?, ?, ?, ? , ?)"), (1, 1, "user1", "This is test bio for user 1", 0, 0))
        conn.commit()

        # 在测试开始之前，将tweet插入到tweet表中
        conn = sqlite3.connect(test_db_filepath)
        cursor = conn.cursor()

        # 在测试开始之前，将60条推文用户插入到tweet表中
        for i in range(1, 61):  # 生成60条tweet
            user_id = i % 3 + 1  # 循环使用用户ID 1, 2, 3
            content = f"Tweet content for tweet {i}"  # 简单生成不同的内容
            created_at = datetime.now()

            cursor.execute(("INSERT INTO tweet "
                            "(user_id, content, created_at, num_likes) "
                            "VALUES (?, ?, ?, ?)"),
                           (user_id, content, created_at, 0))
        conn.commit()
        print_db_contents(test_db_filepath)
        await twitter.running()
        # 验证跟踪表(trace)是否正确记录了操作
        cursor.execute("SELECT * FROM trace WHERE action='refresh'")
        assert cursor.fetchone() is not None, "trend action not traced"

    finally:
        conn.close()
        # 清理
        if os.path.exists(test_db_filepath):
            os.remove(test_db_filepath)
