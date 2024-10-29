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
from oasis.social_platform.typing import ActionType

parent_folder = osp.dirname(osp.abspath(__file__))
test_db_filepath = osp.join(parent_folder, "test.db")


class MockChannel:

    def __init__(self):
        self.call_count = 0
        self.messages = []

    async def receive_from(self):
        if self.call_count == 0:
            self.call_count += 1
            return ("id_", (1, None, ActionType.DO_NOTHING))
        else:
            return ("id_", (None, None, ActionType.EXIT))

    async def send_to(self, message):
        self.messages.append(message)
        if self.call_count == 1:
            assert message[2]["success"] is True


@pytest.fixture
def setup_platform():
    if os.path.exists(test_db_filepath):
        os.remove(test_db_filepath)

    db_path = test_db_filepath
    mock_channel = MockChannel()

    platform_instance = Platform(db_path, mock_channel)
    return platform_instance


@pytest.mark.asyncio
async def test_refresh(setup_platform):
    try:
        platform = setup_platform

        conn = sqlite3.connect(test_db_filepath)
        cursor = conn.cursor()
        cursor.execute(
            ("INSERT INTO user (user_id, agent_id, user_name, bio, "
             "num_followings, num_followers) VALUES (?, ?, ?, ?, ?, ?)"),
            (1, 1, "user1", "This is test bio for user 1", 0, 0),
        )
        conn.commit()

        await platform.running()
        cursor.execute("SELECT * FROM trace WHERE action='do_nothing'")
        assert cursor.fetchone() is not None, "trend action not traced"

    finally:
        conn.close()
        # 清理
        if os.path.exists(test_db_filepath):
            os.remove(test_db_filepath)
