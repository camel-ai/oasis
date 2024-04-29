import os
import sqlite3

import pytest

from core.twitter import Twitter  # 确保从你的模块中导入Twitter类

test_db_filepath = "info_server/test/test.db"


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


class MockChannelFollow:

    def __init__(self):
        self.call_count = 0
        self.messages = []  # 用于存储发送的消息

    async def receive_from(self, group):
        # 第一次调用返回关注操作的指令
        if self.call_count == 0:
            self.call_count += 1
            return 1, 2, "follow"  # 假设用户1关注用户2
        if self.call_count == 1:
            self.call_count += 1
            return 1, 3, "follow"  # 假设用户1关注用户3
        if self.call_count == 2:
            self.call_count += 1
            return 1, 3, "unfollow"  # 假设用户1取消关注用户3
        if self.call_count == 3:
            self.call_count += 1
            return 2, 1, "mute"  # 假设用户2禁言用户1
        if self.call_count == 4:
            self.call_count += 1
            return 2, 3, "mute"  # 假设用户2禁言用户3
        if self.call_count == 5:
            self.call_count += 1
            return 2, 3, "unmute"  # 假设用户2取消禁言用户3
        # 返回退出指令
        else:
            return None, None, "exit"

    async def send_to(self, group, message):
        self.messages.append(message)  # 存储消息以便后续断言
        if self.call_count == 1:
            # 对关注操作的成功消息进行断言
            assert message[1]["success"] is True
            assert "follow_id" in message[1]
        if self.call_count == 2:
            # 对关注操作的成功消息进行断言
            assert message[1]["success"] is True
            assert "follow_id" in message[1]
        if self.call_count == 3:
            # 对取消关注操作的成功消息进行断言
            assert message[1]["success"] is True
            assert "follow_id" in message[1]
        if self.call_count == 4:
            # 对禁言操作的成功消息进行断言
            assert message[1]["success"] is True
            assert "mute_id" in message[1]
        if self.call_count == 5:
            # 对禁言操作的成功消息进行断言
            assert message[1]["success"] is True
            assert "mute_id" in message[1]
        if self.call_count == 6:
            # 对取消禁言操作的成功消息进行断言
            assert message[1]["success"] is True
            assert "mute_id" in message[1]


@pytest.mark.asyncio
async def test_follow_user(setup_twitter):
    try:
        twitter = setup_twitter
        mock_channel = MockChannelFollow()

        # 在测试开始之前，将3个用户插入到user表中
        conn = sqlite3.connect(test_db_filepath)
        cursor = conn.cursor()
        cursor.execute(
            ("INSERT INTO user "
             "(user_id, user_name, num_followings, num_followers) "
             "VALUES (?, ?, ?, ?)"),
            (1, "user1", 0, 0)
        )
        cursor.execute(
            ("INSERT INTO user "
             "(user_id, user_name, num_followings, num_followers) "
             "VALUES (?, ?, ?, ?)"),
            (2, "user2", 2, 4)
        )
        cursor.execute(
            ("INSERT INTO user "
             "(user_id, user_name, num_followings, num_followers) "
             "VALUES (?, ?, ?, ?)"),
            (3, "user3", 3, 5)
        )
        conn.commit()

        await twitter.running(mock_channel)

        # 验证数据库中是否正确插入了数据

        # 验证关注表(follow)是否正确插入了数据
        cursor.execute(
            "SELECT * FROM follow WHERE follower_id=1 AND followee_id=2")
        assert cursor.fetchone() is not None, "Follow record not found"

        # 验证跟踪表(trace)是否正确记录了关注操作
        cursor.execute("SELECT * FROM trace WHERE action='follow'")
        assert cursor.fetchone() is not None, "Follow action not traced"

        # 验证关注表(follow)是否正确删除了数据
        cursor.execute(
            "SELECT * FROM follow WHERE follower_id=1 AND followee_id=3")
        assert cursor.fetchone() is None, "unfollow record not deleted"

        # 验证跟踪表(trace)是否正确记录了取消关注操作
        cursor.execute("SELECT * FROM trace WHERE action='unfollow'")
        assert cursor.fetchone() is not None, "Follow action not traced"

        # 验证禁言表(mute)是否正确插入了数据
        cursor.execute("SELECT * FROM mute WHERE muter_id=2 AND mutee_id=1")
        assert cursor.fetchone() is not None, "Mute record not found"

        # 验证禁言表(mute)是否正确删除了数据
        cursor.execute("SELECT * FROM mute WHERE muter_id=2 AND mutee_id=3")
        assert cursor.fetchone() is None, "Unmute record not found"

        # 验证跟踪表(trace)是否正确记录了禁言操作
        cursor.execute("SELECT * FROM trace WHERE action='mute'")
        assert cursor.fetchone() is not None, "Mute action not traced"

        # 验证跟踪表(trace)是否正确记录了取消禁言操作
        cursor.execute("SELECT * FROM trace WHERE action='unmute'")
        assert cursor.fetchone() is not None, "Unmute action not traced"

        # 验证user表的num_followings和num_followers是否正确更新
        cursor.execute(
            "SELECT num_followings FROM user WHERE user_id = ?",
            (1,)
        )
        result = cursor.fetchone()
        assert result == (1,), "follow action not update user table"
        cursor.execute(
            "SELECT num_followers FROM user WHERE user_id = ?",
            (3,)
        )
        result = cursor.fetchone()
        assert result == (5,), "Unfollow action not update user table"
    finally:
        # 清理
        conn.close()
        # if os.path.exists(test_db_filepath):
        #     os.remove(test_db_filepath)
