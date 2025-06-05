# =========== Copyright 2023 @ CAMEL-AI.org. All Rights Reserved. ===========
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =========== Copyright 2023 @ CAMEL-AI.org. All Rights Reserved. ===========
import os
import os.path as osp
import sqlite3

import pytest

from oasis.social_platform.platform import Platform
from oasis.social_platform.typing import ActionType

parent_folder = osp.dirname(osp.abspath(__file__))
test_db_filepath = osp.join(parent_folder, "test.db")


class MockChannel:
    def __init__(self):
        self.call_count = 0
        self.messages = []

    async def receive_from(self):
        # Simulate a series of operations: create different types of posts and report them
        if self.call_count == 0:
            self.call_count += 1
            return ("id_", (1, "This is a common post", "create_post"))
        elif self.call_count == 1:
            self.call_count += 1
            return ("id_", (2, 1, "repost"))  # Repost the first post
        elif self.call_count == 2:
            self.call_count += 1
            return ("id_", (3, (1, "This is a quote comment"), "quote_post"))  # Quote the first post
        elif self.call_count == 3:
            self.call_count += 1
            return ("id_", (2, (1, "Inappropriate content"), "report_post"))  # Report common post
        elif self.call_count == 4:
            self.call_count += 1
            return ("id_", (3, (2, "Spam content"), "report_post"))  # Report repost
        elif self.call_count == 5:
            self.call_count += 1
            return ("id_", (1, (3, "Misinformation"), "report_post"))  # Report quote post
        else:
            return ("id_", (None, None, "exit"))

    async def send_to(self, message):
        self.messages.append(message)
        if self.call_count == 1:
            # Verify common post creation success
            assert message[2]["success"] is True
            assert "post_id" in message[2]
        elif self.call_count == 2:
            # Verify repost success
            assert message[2]["success"] is True
            assert "post_id" in message[2]
        elif self.call_count == 3:
            # Verify quote post success
            assert message[2]["success"] is True
            assert "post_id" in message[2]
        elif self.call_count == 4:
            # Verify report on common post success
            assert message[2]["success"] is True
            assert "report_id" in message[2]
        elif self.call_count == 5:
            # Verify report on repost success
            assert message[2]["success"] is True
            assert "report_id" in message[2]
        elif self.call_count == 6:
            # Verify report on quote post success
            assert message[2]["success"] is True
            assert "report_id" in message[2]


@pytest.fixture
def setup_platform():
    # Ensure test database doesn't exist
    if os.path.exists(test_db_filepath):
        os.remove(test_db_filepath)

    # Create database and tables
    db_path = test_db_filepath
    mock_channel = MockChannel()
    instance = Platform(db_path=db_path, channel=mock_channel)
    return instance


@pytest.mark.asyncio
async def test_report_post(setup_platform):
    try:
        platform = setup_platform

        # Insert test users
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
            (2, 2, "user2", 0, 0),
        )
        cursor.execute(
            ("INSERT INTO user "
             "(user_id, agent_id, user_name, num_followings, num_followers) "
             "VALUES (?, ?, ?, ?, ?)"),
            (3, 3, "user3", 0, 0),
        )
        conn.commit()

        await platform.running()

        # Verify report records in database
        cursor.execute("SELECT * FROM report ORDER BY created_at")
        reports = cursor.fetchall()
        
        # Verify number of report records
        assert len(reports) == 3, "Should have 3 report records"
        
        # Verify report on common post
        assert reports[0][1] == 2, "Reporter user ID should be 2"
        assert reports[0][2] == 1, "Reported post ID should be 1"
        assert reports[0][3] == "Inappropriate content", "Report reason doesn't match"
        assert reports[0][4] is not None, "Creation time should not be empty"
        
        # Verify report on repost
        assert reports[1][1] == 3, "Reporter user ID should be 3"
        assert reports[1][2] == 2, "Reported post ID should be 2"
        assert reports[1][3] == "Spam content", "Report reason doesn't match"
        assert reports[1][4] is not None, "Creation time should not be empty"

        # Verify report on quote post
        assert reports[2][1] == 1, "Reporter user ID should be 1"
        assert reports[2][2] == 3, "Reported post ID should be 3"
        assert reports[2][3] == "Misinformation", "Report reason doesn't match"
        assert reports[2][4] is not None, "Creation time should not be empty"

        # Verify report comments
        cursor.execute("SELECT * FROM comment WHERE user_id = 0 ORDER BY created_at")
        report_comments = cursor.fetchall()
        
        # Should have one comment per reported post
        assert len(report_comments) == 3, "Should have 3 report comments"
        
        # Verify comment contents
        for comment in report_comments:
            assert "This post has been reported" in comment[3], "Comment should mention report"
            assert comment[4] is not None, "Comment creation time should not be empty"

    finally:
        # Cleanup
        conn.close()
        if os.path.exists(test_db_filepath):
            os.remove(test_db_filepath)