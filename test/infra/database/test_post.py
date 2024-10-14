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
            return ('id_', (1, "This is a test post", "create_post"))
        # 第二次调用返回点赞操作的指令
        elif self.call_count == 1:
            self.call_count += 1
            return ('id_', (1, 1, "like_post"))
        elif self.call_count == 2:
            self.call_count += 1
            return ('id_', (2, 1, "like_post"))
        elif self.call_count == 3:
            self.call_count += 1
            return ('id_', (2, 1, "unlike_post"))
        elif self.call_count == 4:
            self.call_count += 1
            return ('id_', (1, 1, "dislike_post"))
        elif self.call_count == 5:
            self.call_count += 1
            return ('id_', (2, 1, "dislike_post"))
        elif self.call_count == 6:
            self.call_count += 1
            return ('id_', (2, 1, "undo_dislike_post"))
        # 调用返回转推操作的指令
        elif self.call_count == 7:
            self.call_count += 1
            return ('id_', (2, 1, "repost"))
        # 返回退出指令
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
            assert "like_id" in message[2]
        elif self.call_count == 3:
            assert message[2]["success"] is True
            assert "like_id" in message[2]
        elif self.call_count == 4:
            assert message[2]["success"] is True
            assert "like_id" in message[2]
        elif self.call_count == 5:
            # 对点赞操作的成功消息进行断言
            assert message[2]["success"] is True
            assert "dislike_id" in message[2]
        elif self.call_count == 6:
            assert message[2]["success"] is True
            assert "dislike_id" in message[2]
        elif self.call_count == 7:
            assert message[2]["success"] is True
            assert "dislike_id" in message[2]
        elif self.call_count == 8:
            # 对转推的成功消息进行断言
            assert message[2]["success"] is True
            assert "post_id" in message[2]


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
async def test_create_repost_like_unlike_post(setup_platform):
    try:
        platform = setup_platform

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

        await platform.running()

        # 验证数据库中是否正确插入了数据
        conn = sqlite3.connect(test_db_filepath)
        cursor = conn.cursor()

        # 验证推文表(post)是否正确插入了数据
        cursor.execute("SELECT * FROM post")
        posts = cursor.fetchall()
        assert len(posts) == 2  # 一条test post，一条repost
        post = posts[0]
        assert post[1] == 1  # 假设用户ID是1
        assert post[2] == "This is a test post"
        assert post[4] == 1  # num_likes
        assert post[5] == 1  # num_dislikes

        repost = posts[1]
        rt_content = ("user2 repost from user1. "
                      "original_post: This is a test post")
        assert repost[1] == 2  # 转发用户ID为2
        assert repost[2] == rt_content

        # 验证like表是否正确插入了数据
        cursor.execute("SELECT * FROM like")
        likes = cursor.fetchall()
        assert len(likes) == 1

        # 验证dislike表是否正确插入了数据
        cursor.execute("SELECT * FROM dislike")
        dislikes = cursor.fetchall()
        assert len(dislikes) == 1

        # 验证跟踪表(trace)是否正确记录了创建推文和点赞操作
        cursor.execute("SELECT * FROM trace WHERE action='create_post'")
        assert cursor.fetchone() is not None, "Create post action not traced"

        cursor.execute("SELECT * FROM trace WHERE action='repost'")
        assert cursor.fetchone() is not None, "Repost action not traced"

        cursor.execute("SELECT * FROM trace WHERE action='like_post'")
        results = cursor.fetchall()
        assert results is not None, "Like post action not traced"
        assert len(results) == 2

        cursor.execute("SELECT * FROM trace WHERE action='unlike_post'")
        results = cursor.fetchall()
        assert results is not None, "Unlike post action not traced"
        assert results[0][0] == 2  # `user_id`
        assert results[0][-1] == '{"post_id": 1, "like_id": 2}'

        cursor.execute("SELECT * FROM trace WHERE action='dislike_post'")
        results = cursor.fetchall()
        assert results is not None, "Dislike post action not traced"
        assert len(results) == 2

        cursor.execute("SELECT * FROM trace WHERE action='undo_dislike_post'")
        results = cursor.fetchall()
        assert results is not None, "Undo dislike post action not traced"
        assert results[0][0] == 2  # `user_id`
        assert results[0][-1] == '{"post_id": 1, "dislike_id": 2}'

        # 验证点赞表(like)是否正确插入了数据
        cursor.execute("SELECT * FROM like WHERE post_id=1 AND user_id=1")
        assert cursor.fetchone() is not None, "Like record not found"

        # 验证点踩表(dislike)是否正确插入了数据
        cursor.execute("SELECT * FROM dislike WHERE post_id=1 AND user_id=1")
        assert cursor.fetchone() is not None, "Like record not found"

    finally:
        # 清理
        conn.close()
        if os.path.exists(test_db_filepath):
            os.remove(test_db_filepath)
