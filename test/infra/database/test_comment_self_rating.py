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
        # The first call returns the command to create a post
        if self.call_count == 0:
            self.call_count += 1
            return ("id_", (1, "Test post", "create_post"))
        # The second call returns the command to create a comment.
        if self.call_count == 1:
            self.call_count += 1
            return ("id_", (1, (1, "Test Comment"), "create_comment"))
        # The third call returns the command to like a comment.
        if self.call_count == 2:
            self.call_count += 1
            return ("id_", (1, 1, "like_comment"))
        if self.call_count == 3:
            self.call_count += 1
            return ("id_", (2, 1, "like_comment"))
        if self.call_count == 4:
            self.call_count += 1
            return ("id_", (1, 1, "dislike_comment"))
        if self.call_count == 5:
            self.call_count += 1
            return ("id_", (2, 1, "dislike_comment"))
        else:
            return ("id_", (None, None, "exit"))

    async def send_to(self, message):
        # Store the message for subsequent assertions.
        self.messages.append(message)
        if self.call_count == 1:
            assert message[2]["success"] is True
            assert "post_id" in message[2]
        elif self.call_count == 2:
            # Assert the success message for creating a post.
            assert message[2]["success"] is True
            assert "comment_id" in message[2]
        elif self.call_count == 3:
            assert message[2]["success"] is False
            assert message[2]["error"] == (
                "Users are not allowed to like/dislike their own comments.")
        elif self.call_count == 4:
            # Assert the success message for liking a post.
            assert message[2]["success"] is True
            assert "comment_like_id" in message[2]
        elif self.call_count == 5:
            # Assert the success message for unliking a post.
            assert message[2]["success"] is False
            assert message[2]["error"] == (
                "Users are not allowed to like/dislike their own comments.")
        elif self.call_count == 6:
            assert message[2]["success"] is True
            assert "comment_dislike_id" in message[2]


@pytest.fixture
def setup_platform():
    if os.path.exists(test_db_filepath):
        os.remove(test_db_filepath)

    db_path = test_db_filepath

    mock_channel = MockChannel()
    platform_instance = Platform(db_path,
                                 mock_channel,
                                 allow_self_rating=False)
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

    finally:
        conn.close()
        if os.path.exists(test_db_filepath):
            os.remove(test_db_filepath)
