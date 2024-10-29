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
        # On the first call, return the command for the follow operation
        if self.call_count == 0:
            self.call_count += 1
            return ("id_", (1, 2, "follow"))  # Assuming user 1 follows user 2
        if self.call_count == 1:
            self.call_count += 1
            return ("id_", (1, 3, "follow"))  # Assuming user 1 follows user 3
        if self.call_count == 2:
            self.call_count += 1
            return ("id_", (1, 3, "unfollow")
                    )  # Assuming user 1 unfollows user 3
        if self.call_count == 3:
            self.call_count += 1
            return ("id_", (2, 1, "mute"))  # Assuming user 2 mutes user 1
        if self.call_count == 4:
            self.call_count += 1
            return ("id_", (2, 3, "mute"))  # Assuming user 2 mutes user 3
        if self.call_count == 5:
            self.call_count += 1
            return ("id_", (2, 3, "unmute"))  # Assuming user 2 unmutes user 3
        # Returns the exit command afterwards
        else:
            return ("id_", (None, None, "exit"))

    async def send_to(self, message):
        self.messages.append(message)  # Store messages for later assertions
        if self.call_count == 1:
            # Assert on the success message of the follow operation
            assert message[2]["success"] is True
            assert "follow_id" in message[2]
        if self.call_count == 2:
            # Assert on the success message of the follow operation
            assert message[2]["success"] is True
            assert "follow_id" in message[2]
        if self.call_count == 3:
            # Assert on the success message of the unfollow operation
            assert message[2]["success"] is True
            assert "follow_id" in message[2]
        if self.call_count == 4:
            # Assert on the success message of the mute operation
            assert message[2]["success"] is True
            assert "mute_id" in message[2]
        if self.call_count == 5:
            # Assert on the success message of the mute operation
            assert message[2]["success"] is True
            assert "mute_id" in message[2]
        if self.call_count == 6:
            # Assert on the success message of the unmute operation
            assert message[2]["success"] is True
            assert "mute_id" in message[2]


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
async def test_follow_user(setup_platform):
    try:
        platform = setup_platform

        # Insert 3 users into the user table before the test starts
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
        cursor.execute(
            ("INSERT INTO user "
             "(user_id, agent_id, user_name, num_followings, num_followers) "
             "VALUES (?, ?, ?, ?, ?)"),
            (3, 3, "user3", 3, 5),
        )
        conn.commit()

        await platform.running()

        # Verify if the data was correctly inserted into the database

        # Verify if the follow table has the correct data inserted
        cursor.execute(
            "SELECT * FROM follow WHERE follower_id=1 AND followee_id=2")
        assert cursor.fetchone() is not None, "Follow record not found"

        # Verify if the trace table correctly recorded the follow operation
        cursor.execute("SELECT * FROM trace WHERE action='follow'")
        assert cursor.fetchone() is not None, "Follow action not traced"

        # Verify if the follow table correctly deleted the data
        cursor.execute(
            "SELECT * FROM follow WHERE follower_id=1 AND followee_id=3")
        assert cursor.fetchone() is None, "Unfollow record not deleted"

        # Verify if the trace table correctly recorded the unfollow operation
        cursor.execute("SELECT * FROM trace WHERE action='unfollow'")
        assert cursor.fetchone() is not None, "Unfollow action not traced"

        # Verify if the mute table has the correct data inserted
        cursor.execute("SELECT * FROM mute WHERE muter_id=2 AND mutee_id=1")
        assert cursor.fetchone() is not None, "Mute record not found"

        # Verify if the mute table correctly deleted the data
        cursor.execute("SELECT * FROM mute WHERE muter_id=2 AND mutee_id=3")
        assert cursor.fetchone() is None, "Unmute record not deleted"

        # Verify if the trace table correctly recorded the mute operation
        cursor.execute("SELECT * FROM trace WHERE action='mute'")
        assert cursor.fetchone() is not None, "Mute action not traced"

        # Verify if the trace table correctly recorded the unmute operation
        cursor.execute("SELECT * FROM trace WHERE action='unmute'")
        assert cursor.fetchone() is not None, "Unmute action not traced"

    finally:
        # Cleanup
        conn.close()
        if os.path.exists(test_db_filepath):
            os.remove(test_db_filepath)
