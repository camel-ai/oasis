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
import os
import os.path as osp
import sqlite3

import pytest

from oasis.social_platform.platform import Platform

parent_folder = osp.dirname(osp.abspath(__file__))
test_db_filepath = osp.join(parent_folder, "test.db")


class MockChannel:

    def __init__(self):
        self.call_count = 0
        self.messages = []  # Used to store the sent message.

    async def receive_from(self):
        if self.call_count == 0:
            self.call_count += 1
            return ("id_", (1, "Test post", "create_post"))
        if self.call_count == 1:
            self.call_count += 1
            return ("id_", (1, (1, "Test Comment"), "create_comment"))
        if self.call_count == 2:
            self.call_count += 1
            return ("id_", (1, 1, "like_comment"))
        if self.call_count == 3:
            self.call_count += 1
            return ("id_", (2, 1, "like_comment"))
        if self.call_count == 4:
            self.call_count += 1
            return ("id_", (2, 1, "unlike_comment"))
        if self.call_count == 5:
            self.call_count += 1
            return ("id_", (1, 1, "dislike_comment"))
        if self.call_count == 6:
            self.call_count += 1
            return ("id_", (2, 1, "dislike_comment"))
        if self.call_count == 7:
            self.call_count += 1
            return ("id_", (2, 1, "undo_dislike_comment"))
        else:
            return ("id_", (None, None, "exit"))

    async def send_to(self, message):
        # Store the message for subsequent assertions.
        self.messages.append(message)
        if self.call_count == 1:
            assert message[2]["success"] is True
            assert "post_id" in message[2]
        elif self.call_count == 2:
            assert message[2]["success"] is True
            assert "comment_id" in message[2]
        elif self.call_count == 3:
            assert message[2]["success"] is True
            assert "comment_like_id" in message[2]
        elif self.call_count == 4:
            assert message[2]["success"] is True
            assert "comment_like_id" in message[2]
        elif self.call_count == 5:
            assert message[2]["success"] is True
            assert "comment_like_id" in message[2]
        elif self.call_count == 6:
            assert message[2]["success"] is True
            assert "comment_dislike_id" in message[2]
        elif self.call_count == 7:
            assert message[2]["success"] is True
            assert "comment_dislike_id" in message[2]
        elif self.call_count == 8:
            assert message[2]["success"] is True
            assert "comment_dislike_id" in message[2]


@pytest.fixture
def setup_platform():
    if os.path.exists(test_db_filepath):
        os.remove(test_db_filepath)

    db_path = test_db_filepath

    mock_channel = MockChannel()
    platform_instance = Platform(db_path, mock_channel)
    return platform_instance


@pytest.mark.asyncio
async def test_comment(setup_platform):
    try:
        platform = setup_platform

        conn = sqlite3.connect(test_db_filepath)
        cursor = conn.cursor()
        cursor.execute(
            ("INSERT INTO user "
             "(user_id, agent_id, user_name, num_followings, num_followers) "
             "VALUES (?, ?, ?, ?, ?)"),
            (1, 1, "user1", 0, 0),
        )
        cursor.execute(
            ("INSERT INTO user "
             "(user_id, agent_id, user_name, num_followings, num_followers) "
             "VALUES (?, ?, ?, ?, ?)"),
            (2, 2, "user2", 2, 4),
        )
        conn.commit()

        await platform.running()

        conn = sqlite3.connect(test_db_filepath)
        cursor = conn.cursor()

        cursor.execute("SELECT * FROM comment")
        comments = cursor.fetchall()
        assert len(comments) == 1  # A test post，A repost
        comment = comments[0]
        assert comment[1] == 1  # post ID is 1
        assert comment[2] == 1  # user ID is 1
        assert comment[3] == "Test Comment"
        assert comment[5] == 1  # num_likes
        assert comment[6] == 1  # num_dislikes

        cursor.execute("SELECT * FROM comment_like")
        comment_likes = cursor.fetchall()
        assert len(comment_likes) == 1

        cursor.execute("SELECT * FROM comment_dislike")
        dislikes = cursor.fetchall()
        assert len(dislikes) == 1

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

        cursor.execute("SELECT * FROM comment_like WHERE comment_id=1 AND "
                       "user_id=1")
        assert cursor.fetchone() is not None, "Comment like record not found"

        cursor.execute("SELECT * FROM comment_dislike WHERE comment_id=1 AND "
                       "user_id=1")
        fetched_record = cursor.fetchone()
        assert fetched_record is not None, "Comment dislike record not found"

    finally:
        conn.close()
        if os.path.exists(test_db_filepath):
            os.remove(test_db_filepath)
