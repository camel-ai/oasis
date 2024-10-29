# =========== Copyright 2023 @ CAMEL-AI.org. All Rights Reserved. ===========
# Licensed under the Apache License, Version 2.0 (the “License”);
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an “AS IS” BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =========== Copyright 2023 @ CAMEL-AI.org. All Rights Reserved. ===========
import asyncio
import os
import os.path as osp
import sqlite3
from datetime import datetime

import pytest

from oasis.social_platform.channel import Channel
from oasis.social_platform.platform import Platform
from oasis.social_platform.typing import ActionType

parent_folder = osp.dirname(osp.abspath(__file__))
test_db_filepath = osp.join(parent_folder, "test.db")


@pytest.fixture
def setup_db():
    # Ensure test.db does not exist before the test
    if os.path.exists(test_db_filepath):
        os.remove(test_db_filepath)


@pytest.mark.asyncio
async def test_update_rec_table(setup_db):
    try:
        channel = Channel()
        infra = Platform(test_db_filepath,
                         channel,
                         recsys_type="reddit",
                         max_rec_post_len=50)
        # Insert 3 users into the user table before the test starts
        conn = sqlite3.connect(test_db_filepath)
        cursor = conn.cursor()
        cursor.execute(
            ("INSERT INTO user "
             "(agent_id, user_name, bio, num_followings, num_followers) "
             "VALUES (?, ?, ?, ?, ?)"),
            (0, "user1", "This is test bio for user1", 0, 0),
        )
        cursor.execute(
            ("INSERT INTO user "
             "(agent_id, user_name, bio, num_followings, num_followers) "
             "VALUES (?, ?, ?, ?, ?)"),
            (1, "user2", "This is test bio for user2", 2, 4),
        )
        cursor.execute(
            ("INSERT INTO user "
             "(agent_id, user_name, bio, num_followings, num_followers) "
             "VALUES (?, ?, ?, ?, ?)"),
            (2, "user3", "This is test bio for user3", 3, 5),
        )
        conn.commit()

        # Insert 60 tweet users into the post table before the test starts
        for i in range(1, 61):  # Generate 60 posts
            user_id = i % 3 + 1  # Cycle through user IDs 1, 2, 3
            # Simply generate different content
            content = f"Post content for post {i}"
            created_at = datetime(2024, 6, 27, i % 24, 0, 0, 123456)
            num_likes = i  # Randomly generate the number of likes

            cursor.execute(
                ("INSERT INTO post "
                 "(user_id, content, created_at, num_likes) "
                 "VALUES (?, ?, ?, ?)"),
                (user_id, content, created_at, num_likes),
            )
        conn.commit()

        task = asyncio.create_task(infra.running())
        await channel.write_to_receive_queue(
            (None, None, ActionType.UPDATE_REC_TABLE))
        await channel.write_to_receive_queue((None, None, ActionType.EXIT))
        await task

        # print_db_contents(test_db_filepath)

        for i in range(3):
            cursor.execute("SELECT post_id FROM rec WHERE user_id = ?", (i, ))
            posts = cursor.fetchall()  # Fetch all records
            assert len(posts) == 50, f"User {user_id} doesn't have 50 posts."
            post_ids = [post[0] for post in posts]
            print(post_ids)
            is_unique = len(post_ids) == len(set(post_ids))
            assert is_unique, f"User {user_id} has duplicate post_ids."

    finally:
        conn.close()
        # Cleanup
        if os.path.exists(test_db_filepath):
            os.remove(test_db_filepath)
