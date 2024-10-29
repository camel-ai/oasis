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
        self.messages = []  # Used to store sent messages

    async def receive_from(self):
        # Returns the command to search users on the first call
        if self.call_count == 0:
            self.call_count += 1
            return ("id_", (1, "bob", "search_user")
                    )  # Assuming the search keyword is "bob"
        if self.call_count == 1:
            self.call_count += 1
            return ("id_", (2, "Bob", "search_posts")
                    )  # Assuming the search keyword is "bob"
        # Returns the exit command on subsequent calls
        else:
            return ("id_", (None, None, "exit"))

    async def send_to(self, message):
        self.messages.append(message)  # Store messages for later assertions
        # Assert on the results of searching users
        if self.call_count == 1:
            # Verify the search was successful and found at least one
            # matching user
            assert message[2]["success"] is True, "Search should be successful"
            assert len(
                message[2]["users"]) > 0, "Should find at least one user"
            # You can add more assertions here to verify the correctness of
            # the returned user information
            assert (message[2]["users"][0]["user_name"] == "user2"
                    ), "The first matching user should be 'user2'"
        if self.call_count == 2:
            assert message[2]["success"] is True, "Search should be successful"
            assert len(
                message[2]["posts"]) > 0, "Should find at least one post"
            assert message[2]["posts"][0]["content"] == "Bob's first post!"
            assert message[2]["posts"][0]["comments"][0]["content"] == (
                "Alice's comment")
            assert message[2]["posts"][0]["comments"][1]["content"] == (
                "Bob's comment")
            assert message[2]["posts"][0]["comments"][2]["content"] == (
                "Charlie's comment")


@pytest.fixture
def setup_platform():
    # Ensure test.db does not exist before the test
    if os.path.exists(test_db_filepath):
        os.remove(test_db_filepath)

    # Create the database and table
    db_path = test_db_filepath

    mock_channel = MockChannel()
    instance = Platform(db_path, mock_channel)
    return instance


@pytest.mark.asyncio
async def test_search_user(setup_platform):
    try:
        platform = setup_platform

        # Insert several users into the user table before the test starts
        conn = sqlite3.connect(test_db_filepath)
        cursor = conn.cursor()
        users_info = [
            (1, 1, "user1", "Alice", "Bio of Alice", "2023-01-01 12:00:00", 10,
             5),
            (2, 2, "user2", "Bob", "Bio of Bob", "2023-01-02 12:00:00", 15, 8),
            (3, 3, "user3", "Charlie", "Bio of Charlie", "2023-01-03 12:00:00",
             20, 12),
        ]
        cursor.executemany("INSERT INTO user VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                           users_info)
        posts_info = [
            # (user_id, content, created_at, num_likes)
            (1, "Hello World from Alice!", "2023-01-01 13:00:00", 100, 2),
            (2, "Bob's first post!", "2023-01-02 14:00:00", 150, 2),
            (3, "Charlie says hi!", "2023-01-03 15:00:00", 200, 3),
        ]
        # Assuming cursor is your cursor object already created and connected
        # to the database
        cursor.executemany(
            ("INSERT INTO post (user_id, content, created_at, num_likes, "
             "num_dislikes) VALUES (?, ?, ?, ?, ?)"),
            posts_info,
        )
        conn.commit()

        comments_info = [
            # (post_id, user_id, content)
            (2, 1, "Alice's comment", "2023-01-01 13:05:00", 5, 1),
            (2, 2, "Bob's comment", "2023-01-02 14:10:00", 3, 0),
            (2, 3, "Charlie's comment", "2023-01-03 15:20:00", 8, 2),
        ]
        # Assuming cursor is your cursor object already created and connected
        # to the database
        cursor.executemany(
            "INSERT INTO comment (post_id, user_id, content, created_at, "
            "num_likes, num_dislikes) VALUES (?, ?, ?, ?, ?, ?)",
            comments_info,
        )
        conn.commit()

        await platform.running()

        # Verify that the trace table correctly recorded the action
        cursor.execute("SELECT * FROM trace WHERE action='search_user'")
        assert cursor.fetchone() is not None, "search_user action not traced"

        # Verify that the trace table correctly recorded the action
        cursor.execute("SELECT * FROM trace WHERE action='search_posts'")
        assert cursor.fetchone() is not None, "search_post action not traced"

    finally:
        conn.close()
        # Cleanup
        if os.path.exists(test_db_filepath):
            os.remove(test_db_filepath)
