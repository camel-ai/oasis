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
from datetime import datetime, timedelta

import pytest

from oasis.social_platform.platform import Platform

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
            return ("id_", (1, None, "trend"))
        else:
            return ("id_", (None, None, "exit"))

    async def send_to(self, message):
        self.messages.append(message)  # Store message for later assertion
        # Asserts the result of the user search
        if self.call_count == 1:
            print(message[2])
            # Verify the search was successful and at least one matching user
            # was found
            assert message[2]["success"] is True, "Trend should be successful"
            assert message[2]["posts"][0]["content"] == "Post 6"
            print(message[2]["posts"])


@pytest.fixture
def setup_platform():
    # Ensure test.db does not exist before testing
    if os.path.exists(test_db_filepath):
        os.remove(test_db_filepath)

    # Create database and tables
    db_path = test_db_filepath

    mock_channel = MockChannel()
    instance = Platform(db_path, mock_channel)
    return instance


@pytest.mark.asyncio
async def test_search_user(setup_platform):
    try:
        platform = setup_platform

        # Insert 1 user into the user table before the test starts
        conn = sqlite3.connect(test_db_filepath)
        cursor = conn.cursor()
        cursor.execute(
            ("INSERT INTO user "
             "(user_id, agent_id, user_name, num_followings, num_followers) "
             "VALUES (?, ?, ?, ?, ?)"),
            (1, 1, "user1", 0, 0),
        )
        conn.commit()

        # Insert post into the post table before the test starts
        conn = sqlite3.connect(test_db_filepath)
        cursor = conn.cursor()

        today = platform.start_time
        # Generate a list of timestamps starting from today and going back 10
        # days
        posts_info = [(
            1,
            f"Post {9-i}",
            (today - timedelta(days=9 - i)).strftime("%Y-%m-%d %H:%M:%S.%f"),
            (9 - i),
            0,
        ) for i in range(10)]

        cursor.executemany(
            "INSERT INTO post (user_id, content, created_at, num_likes, "
            "num_dislikes) VALUES (?, ?, ?, ?, ?)",
            posts_info,
        )
        conn.commit()

        comments_info = [(i + 1, 1, "Comment", datetime.now())
                         for i in range(10)]

        cursor.executemany(
            "INSERT INTO comment (post_id, user_id, content, created_at) "
            "VALUES (?, ?, ?, ?)",
            comments_info,
        )
        conn.commit()

        await platform.running()

        # Verify that the trace table correctly recorded the operation
        cursor.execute("SELECT * FROM trace WHERE action='trend'")
        assert cursor.fetchone() is not None, "trend action not traced"

    finally:
        conn.close()
        # Cleanup
        if os.path.exists(test_db_filepath):
            os.remove(test_db_filepath)
