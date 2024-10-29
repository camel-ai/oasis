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
        if self.call_count == 0:
            self.call_count += 1
            return ("id_", (1, ("alice0101", "Alice", "A girl."), "sign_up"))
        elif self.call_count == 1:
            self.call_count += 1
            return ("id_", (2, ("bubble", "Bob", "A boy."), "sign_up"))
        # Returns the exit command
        else:
            return ("id_", (None, None, "exit"))

    async def send_to(self, message):
        self.messages.append(message)  # Store messages for later assertions
        if self.call_count == 1:
            print(message[2])
            assert message[2]["success"] is True
            assert "user_id" in message[2]
        elif self.call_count == 2:
            assert message[2]["success"] is True
            assert "user_id" in message[2]


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
async def test_create_like_unlike_post(setup_platform):
    try:
        platform = setup_platform

        await platform.running()

        # Verify if the data is correctly inserted into the database
        conn = sqlite3.connect(test_db_filepath)
        cursor = conn.cursor()

        # Verify if posts are correctly inserted into the post table
        cursor.execute("SELECT * FROM user")
        users = cursor.fetchall()
        assert len(users) == 2
        assert users[0][1] == 1
        assert users[0][2] == "alice0101"
        assert users[0][3] == "Alice"
        assert users[0][4] == "A girl."
        assert users[1][1] == 2
        assert users[1][2] == "bubble"
        assert users[1][3] == "Bob"
        assert users[1][4] == "A boy."

        # Verify if the trace table correctly recorded the sign-up operations
        cursor.execute("SELECT * FROM trace WHERE action ='sign_up'")
        results = cursor.fetchall()
        assert len(results) == 2

    finally:
        conn.close()
        if os.path.exists(test_db_filepath):
            os.remove(test_db_filepath)
