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
from datetime import datetime

import pytest

from oasis.social_platform.platform import Platform
from oasis.social_platform.typing import ActionType

parent_folder = osp.dirname(osp.abspath(__file__))
test_db_filepath = osp.join(parent_folder, "test.db")


class MockChannel:

    def __init__(self):
        self.call_count = 0
        self.messages = []  # Used to store sent messages

    async def receive_from(self):
        # Returns the command to search for a user on the first call
        if self.call_count == 0:
            self.call_count += 1
            return ("id_", (None, None, ActionType.UPDATE_REC_TABLE))
        if self.call_count == 1:
            self.call_count += 1
            return ("id_", (0, None, ActionType.REFRESH))
        else:
            return ("id_", (None, None, ActionType.EXIT))

    async def send_to(self, message):
        self.messages.append(message)  # Store message for later assertion
        # Asserts the result of user refresh
        if self.call_count == 2:
            # Verify refresh success
            # print_db_contents(test_db_filepath)
            assert message[2]["success"] is True
            print(message[2]["posts"])
            assert len(message[2]["posts"]) == 1
            # Then check each entry in the 'posts' list
            for post in message[2].get("posts", []):
                assert post.get("post_id") is not None
                assert post.get("user_id") is not None
                assert post.get("content") is not None
                assert post.get("created_at") is not None
                assert post.get("score") == -1
                assert post.get("comments")[0].get("score") == -2


@pytest.fixture
def setup_platform():
    # Ensure test.db does not exist before testing
    if os.path.exists(test_db_filepath):
        os.remove(test_db_filepath)

    # Create database and tables
    db_path = test_db_filepath
    mock_channel = MockChannel()
    # Initialize Platform instance
    platform_instance = Platform(db_path, mock_channel, show_score=True)
    return platform_instance


@pytest.mark.asyncio
async def test_refresh(setup_platform):
    try:
        platform = setup_platform

        # Insert 1 user into the user table before the test starts
        conn = sqlite3.connect(test_db_filepath)
        cursor = conn.cursor()
        cursor.execute(
            ("INSERT INTO user (user_id, agent_id, user_name, bio, "
             "num_followings, num_followers) VALUES (?, ?, ?, ?, ?, ?)"),
            (0, 0, "user0", "This is test bio for user 0", 0, 0),
        )
        conn.commit()

        # Insert post into the post table before the test starts
        conn = sqlite3.connect(test_db_filepath)
        cursor = conn.cursor()

        # Insert 60 tweet users into the post table before the test starts
        for i in range(60):  # Generate 60 posts
            user_id = i % 3  # Cycle through user IDs 1, 2, 3
            # Simply generate different content
            content = f"Posts content for post {i}"
            comment_content = f"Comment content for post {i}"
            created_at = datetime.now()

            cursor.execute(
                ("INSERT INTO post (user_id, content, created_at, "
                 "num_likes, num_dislikes) VALUES (?, ?, ?, ?, ?)"),
                (user_id, content, created_at, 0, 1),
            )
            cursor.execute(
                ("INSERT INTO comment (post_id, user_id, content, "
                 "created_at, num_likes, num_dislikes) VALUES "
                 "(?, ?, ?, ?, ?, ?)"),
                (i, user_id, comment_content, created_at, 0, 2),
            )
        conn.commit()
        # print_db_contents(test_db_filepath)
        await platform.running()
        # Verify that the trace table correctly recorded the operation
        cursor.execute("SELECT * FROM trace WHERE action='refresh'")
        assert cursor.fetchone() is not None, "refresh action not traced"

    finally:
        conn.close()
        # Cleanup
        if os.path.exists(test_db_filepath):
            os.remove(test_db_filepath)
