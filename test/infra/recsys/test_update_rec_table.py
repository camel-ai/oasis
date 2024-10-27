import asyncio
import os
import os.path as osp
import random
import sqlite3

import pytest

from social_simulation.social_platform.channel import Channel
from social_simulation.social_platform.platform import Platform
from social_simulation.social_platform.recsys import reset_globals
from social_simulation.social_platform.typing import ActionType
from social_simulation.testing.show_db import print_db_contents

parent_folder = osp.dirname(osp.abspath(__file__))
test_db_filepath = osp.join(parent_folder, "test.db")


@pytest.fixture
def setup_db():
    if os.path.exists(test_db_filepath):
        os.remove(test_db_filepath)


@pytest.mark.asyncio
async def test_update_rec_table(setup_db):
    try:
        channel = Channel()
        recsys_type = "twhin-bert"
        infra = Platform(test_db_filepath,
                         channel,
                         recsys_type=recsys_type,
                         refresh_rec_post_count=50,
                         max_rec_post_len=50)
        if recsys_type == "twhin-bert":
            reset_globals()
        # 在测试开始之前，将3个用户插入到user表中
        conn = sqlite3.connect(test_db_filepath)
        cursor = conn.cursor()
        cursor.execute((
            "INSERT INTO user "
            "(user_id, agent_id, user_name, bio, num_followings, num_followers) "
            "VALUES (?, ?, ?, ?, ?, ?)"),
                       (0, 0, "user1", "This is test bio for user1", 0, 0))
        cursor.execute((
            "INSERT INTO user "
            "(user_id, agent_id, user_name, bio, num_followings, num_followers) "
            "VALUES (?, ?, ?, ?, ?, ?)"),
                       (1, 1, "user2", "This is test bio for user2", 2, 4))
        cursor.execute((
            "INSERT INTO user "
            "(user_id, agent_id, user_name, bio, num_followings, num_followers) "
            "VALUES (?, ?, ?, ?, ?, ?)"),
                       (2, 2, "user3", "This is test bio for user3", 3, 5))
        conn.commit()

        # 在测试开始之前，将60条推文用户插入到post表中
        for i in range(60):  # 生成60条post
            user_id = i % 3  # 循环使用用户ID 0, 1, 2
            content = f"Post content for post {i}"  # 简单生成不同的内容
            created_at = "0"
            num_likes = random.randint(0, 100)  # 随机生成点赞数

            cursor.execute(("INSERT INTO post "
                            "(user_id, content, created_at, num_likes) "
                            "VALUES (?, ?, ?, ?)"),
                           (user_id, content, created_at, num_likes))
        conn.commit()

        os.environ["SANDBOX_TIME"] = '0'

        task = asyncio.create_task(infra.running())
        await channel.write_to_receive_queue(
            (None, None, ActionType.UPDATE_REC_TABLE))
        await channel.write_to_receive_queue((None, None, ActionType.EXIT))
        await task

        for i in range(3):
            cursor.execute("SELECT post_id FROM rec WHERE user_id = ?", (i, ))
            posts = cursor.fetchall()  # 获取所有记录
            # ! Number of available posts for recommendation =
            # total posts - posts from current user !
            # 实际上推特里是可以看到自己发的推文的
            assert len(posts) == 50, f"User {user_id} doesn't have 50 posts."
            post_ids = [post[0] for post in posts]
            is_unique = len(post_ids) == len(set(post_ids))
            print(set(post_ids))
            assert is_unique, f"User {user_id} has duplicate post_ids."

        print_db_contents(test_db_filepath)

    finally:
        conn.close()
        # 清理
        if os.path.exists(test_db_filepath):
            os.remove(test_db_filepath)
