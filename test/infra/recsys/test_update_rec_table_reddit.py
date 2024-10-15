import asyncio
import os
import os.path as osp
import sqlite3
from datetime import datetime

import pytest

from social_simulation.social_platform.channel import Channel
from social_simulation.social_platform.platform import Platform
from social_simulation.social_platform.typing import ActionType
from social_simulation.testing.show_db import print_db_contents

parent_folder = osp.dirname(osp.abspath(__file__))
test_db_filepath = osp.join(parent_folder, "test.db")


@pytest.fixture
def setup_db():
    # 测试前确保test.db不存在
    if os.path.exists(test_db_filepath):
        os.remove(test_db_filepath)


@pytest.mark.asyncio
async def test_update_rec_table(setup_db):
    try:
        channel = Channel()
        infra = Platform(
            test_db_filepath, channel, recsys_type='reddit', max_rec_post_len = 50)
        # 在测试开始之前，将3个用户插入到user表中
        conn = sqlite3.connect(test_db_filepath)
        cursor = conn.cursor()
        cursor.execute(
            ("INSERT INTO user "
             "(agent_id, user_name, bio, num_followings, num_followers) "
             "VALUES (?, ?, ?, ?, ?)"),
            (0, "user1", "This is test bio for user1", 0, 0))
        cursor.execute(
            ("INSERT INTO user "
             "(agent_id, user_name, bio, num_followings, num_followers) "
             "VALUES (?, ?, ?, ?, ?)"),
            (1, "user2", "This is test bio for user2", 2, 4))
        cursor.execute(
            ("INSERT INTO user "
             "(agent_id, user_name, bio, num_followings, num_followers) "
             "VALUES (?, ?, ?, ?, ?)"),
            (2, "user3", "This is test bio for user3", 3, 5))
        conn.commit()

        # 在测试开始之前，将60条推文用户插入到post表中
        for i in range(1, 61):  # 生成60条post
            user_id = i % 3 + 1  # 循环使用用户ID 1, 2, 3
            content = f"Post content for post {i}"  # 简单生成不同的内容
            created_at = datetime(2024, 6, 27, i % 24, 0, 0, 123456)
            num_likes = i  # 随机生成点赞数

            cursor.execute(("INSERT INTO post "
                            "(user_id, content, created_at, num_likes) "
                            "VALUES (?, ?, ?, ?)"),
                           (user_id, content, created_at, num_likes))
        conn.commit()

        task = asyncio.create_task(infra.running())
        await channel.write_to_receive_queue(
            (None, None, ActionType.UPDATE_REC_TABLE))
        await channel.write_to_receive_queue((None, None, ActionType.EXIT))
        await task

        # print_db_contents(test_db_filepath)

        for i in range(3):
            cursor.execute("SELECT post_id FROM rec WHERE user_id = ?", (i, ))
            posts = cursor.fetchall()  # 获取所有记录
            assert len(posts) == 50, f"User {user_id} doesn't have 50 posts."
            post_ids = [post[0] for post in posts]
            print(post_ids)
            is_unique = len(post_ids) == len(set(post_ids))
            assert is_unique, f"User {user_id} has duplicate post_ids."

    finally:
        conn.close()
        # 清理
        if os.path.exists(test_db_filepath):
            os.remove(test_db_filepath)
