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
        self.messages = []
        self.call_count = 0

    async def receive_from(self):
        if self.call_count == 0:
            self.call_count += 1
            return ("id_", (0, ("Alice", "A", "Bio A"), "sign_up"))
        elif self.call_count == 1:
            self.call_count += 1
            return ("id_", (1, ("Bob", "B", "Bio B"), "sign_up"))
        elif self.call_count == 2:
            self.call_count += 1
            return ("id_", (2, ("Charlie", "C", "Bio C"), "sign_up"))
        elif self.call_count == 3:
            self.call_count += 1
            return ("id_", (0, "AI Discussion Group", "create_group"))
        elif self.call_count == 4:
            self.call_count += 1
            return ("id_", (1, 1, "join_group"))
        elif self.call_count == 5:
            self.call_count += 1
            return ("id_", (2, 1, "join_group"))
        elif self.call_count == 6:
            self.call_count += 1
            return ("id_", (0, (1, "Hello everyone!"), "send_to_group"))
        else:
            return ("id_", (None, None, "exit"))

    async def send_to(self, message):
        self.messages.append(message)


@pytest.fixture
def setup_platform():
    # Remove the test database file if it exists
    if os.path.exists(test_db_filepath):
        os.remove(test_db_filepath)

    # Create a mock channel and initialize the platform instance
    mock_channel = MockChannel()
    instance = Platform(test_db_filepath, channel=mock_channel)
    return instance


@pytest.mark.asyncio
async def test_group_functionality(setup_platform):
    try:
        platform = setup_platform
        await platform.running()

        # Connect to the SQLite database for verification
        conn = sqlite3.connect(test_db_filepath)
        cursor = conn.cursor()

        # Verify user registration
        cursor.execute("SELECT * FROM user")
        users = cursor.fetchall()
        assert len(users) == 3  # Expecting 3 registered users

        # Verify group creation
        cursor.execute("SELECT * FROM chat_group")
        groups = cursor.fetchall()
        assert len(groups) == 1  # One group should be created
        assert groups[0][1] == "AI Discussion Group"  # Check group name

        # Verify members joining the group
        cursor.execute("SELECT * FROM group_members WHERE group_id = 1")
        members = cursor.fetchall()
        assert len(members) == 3  # All three users joined the group

        # Verify message sending
        cursor.execute(
            "SELECT * FROM group_messages WHERE content = 'Hello everyone!'")
        messages = cursor.fetchall()
        assert len(messages) == 1  # One message with this content should exist

    finally:
        # Ensure database connection is closed after test
        conn.close()
        # Clean up: remove the test database file
        if os.path.exists(test_db_filepath):
            os.remove(test_db_filepath)
