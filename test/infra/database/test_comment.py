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
        # 第1次调用返回创建推文的指令
        if self.call_count == 0:
            self.call_count += 1
            return ('id_', (1, "Test post", "create_post"))
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
            return ('id_', (2, 1, "unlike_comment"))
        if self.call_count == 5:
            self.call_count += 1
            return ('id_', (1, 1, "dislike_comment"))
        if self.call_count == 6:
            self.call_count += 1
            return ('id_', (2, 1, "dislike_comment"))
        if self.call_count == 7:
            self.call_count += 1
            return ('id_', (2, 1, "undo_dislike_comment"))
        else:
            return ('id_', (None, None, "exit"))

    async def send_to(self, message):
        self.messages.append(message)  # 存储消息以便后续断言
        if self.call_count == 1:
            # 对创建推文的成功消息进行断言
            assert message[2]["success"] is True
            assert "post_id" in message[2]
        elif self.call_count == 2:
            # 对点赞操作的成功消息进行断言
            assert message[2]["success"] is True
            assert "comment_id" in message[2]
        elif self.call_count == 3:
            # 对点赞操作的成功消息进行断言
            assert message[2]["success"] is True
            assert "comment_like_id" in message[2]
        elif self.call_count == 4:
            # 对点赞操作的成功消息进行断言
            assert message[2]["success"] is True
            assert "comment_like_id" in message[2]
        elif self.call_count == 5:
            # 对取消点赞操作的成功消息进行断言
            assert message[2]["success"] is True
            assert "comment_like_id" in message[2]
        elif self.call_count == 6:
            # 对点赞操作的成功消息进行断言
            assert message[2]["success"] is True
            assert "comment_dislike_id" in message[2]
        elif self.call_count == 7:
            # 对点赞操作的成功消息进行断言
            assert message[2]["success"] is True
            assert "comment_dislike_id" in message[2]
        elif self.call_count == 8:
            # 对点赞操作的成功消息进行断言
            assert message[2]["success"] is True
            assert "comment_dislike_id" in message[2]


@pytest.fixture
def setup_platform():
    # 测试前确保test.db不存在
    if os.path.exists(test_db_filepath):
        os.remove(test_db_filepath)

    # 创建数据库和表
    db_path = test_db_filepath

    mock_channel = MockChannel()
    platform_instance = Platform(db_path, mock_channel)
    return platform_instance


@pytest.mark.asyncio
async def test_comment(setup_platform):
    try:
        platform = setup_platform

        # 在测试开始之前，将2个用户插入到user表中
        conn = platform.db
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

        await platform.running()

        # 验证数据库中是否正确插入了数据
        conn = sqlite3.connect(test_db_filepath)
        cursor = conn.cursor()

        # 验证推文表(post)是否正确插入了数据
        cursor.execute("SELECT * FROM comment")
        comments = cursor.fetchall()
        assert len(comments) == 1  # 一条test post，一条repost
        comment = comments[0]
        assert comment[1] == 1  # post ID是1
        assert comment[2] == 1  # user ID是1
        assert comment[3] == "Test Comment"
        assert comment[5] == 1  # num_likes
        assert comment[6] == 1  # num_dislikes

        # 验证comment_like表是否正确插入了数据
        cursor.execute("SELECT * FROM comment_like")
        comment_likes = cursor.fetchall()
        assert len(comment_likes) == 1

        # 验证dislike表是否正确插入了数据
        cursor.execute("SELECT * FROM comment_dislike")
        dislikes = cursor.fetchall()
        assert len(dislikes) == 1

        # 验证跟踪表(trace)是否正确记录了创建推文和点赞操作
        cursor.execute("SELECT * FROM trace WHERE action='create_comment'")
        assert cursor.fetchone() is not None, "Create post action not traced"

        cursor.execute("SELECT * FROM trace WHERE action='like_comment'")
        results = cursor.fetchall()
        assert results is not None, "Like comment action not traced"
        assert len(results) == 2

        cursor.execute("SELECT * FROM trace WHERE action='unlike_comment'")
        results = cursor.fetchall()
        assert results is not None, "Unlike comment action not traced"
        assert results[0][0] == 2  # `user_id`
        assert results[0][-1] == '{"comment_id": 1, "comment_like_id": 2}'

        cursor.execute("SELECT * FROM trace WHERE action='dislike_comment'")
        results = cursor.fetchall()
        assert results is not None, "Dislike comment action not traced"
        assert len(results) == 2

        cursor.execute(
            "SELECT * FROM trace WHERE action='undo_dislike_comment'")
        results = cursor.fetchall()
        assert results is not None, "Undo dislike comment action not traced"
        assert results[0][0] == 2  # `user_id`
        assert results[0][-1] == '{"comment_id": 1, "comment_dislike_id": 2}'

        # 验证comment like表是否正确插入了数据
        cursor.execute("SELECT * FROM comment_like WHERE comment_id=1 AND "
                       "user_id=1")
        assert cursor.fetchone() is not None, "Comment like record not found"

        # 验证comment_dislike表是否正确插入了数据
        cursor.execute("SELECT * FROM comment_dislike WHERE comment_id=1 AND "
                       "user_id=1")
        fetched_record = cursor.fetchone()
        assert fetched_record is not None, "Comment dislike record not found"

    finally:
        # 清理
        conn.close()
        if os.path.exists(test_db_filepath):
            os.remove(test_db_filepath)
