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
        self.messages = []  # Used to store sent messages

    async def receive_from(self):
        if self.call_count == 0:
            self.call_count += 1
            return ("id_", (1, ('apple', 1),
                            ActionType.PURCHASE_PRODUCT.value))
        if self.call_count == 1:
            self.call_count += 1
            return ("id_", (2, ('apple', 2),
                            ActionType.PURCHASE_PRODUCT.value))
        if self.call_count == 2:
            self.call_count += 1
            return ("id_", (2, ('banana', 1),
                            ActionType.PURCHASE_PRODUCT.value))
        if self.call_count == 3:
            self.call_count += 1
            return ("id_", (2, ('orange', 1),
                            ActionType.PURCHASE_PRODUCT.value))
        else:
            return ("id_", (None, None, "exit"))

    async def send_to(self, message):
        self.messages.append(message)  # Store message for later assertion
        if self.call_count == 1:
            print(message[2])
            msg = "Purchase apple from user 1 failed"
            assert message[2]["success"] is True, msg
            assert message[2]["product_id"] == 1
        elif self.call_count == 2:
            msg = "Purchase apple from user 2 failed"
            assert message[2]["success"] is True, msg
            assert message[2]["product_id"] == 1
        elif self.call_count == 3:
            msg = "Purchase banana from user 2 failed"
            assert message[2]["success"] is True, msg
            assert message[2]["product_id"] == 2
        elif self.call_count == 4:
            assert message[2]["success"] is False


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
        cursor.execute(
            ("INSERT INTO user "
             "(user_id, agent_id, user_name, num_followings, num_followers) "
             "VALUES (?, ?, ?, ?, ?)"),
            (2, 2, "user2", 0, 0),
        )
        conn.commit()

        conn = sqlite3.connect(test_db_filepath)
        cursor = conn.cursor()

        await platform.sign_up_product(1, "apple")
        await platform.sign_up_product(2, "banana")
        # print_db_contents(test_db_filepath)

        await platform.running()

        # Verify that the trace table correctly recorded the operation
        cursor.execute(
            "SELECT * FROM product WHERE product_name='apple' and sales=3")
        assert cursor.fetchone() is not None, "apple sales is not 3"
        cursor.execute(
            "SELECT * FROM product WHERE product_name='banana' and sales=1")
        assert cursor.fetchone() is not None, "banana sales is not 1"

    finally:
        conn.close()
        # Cleanup
        if os.path.exists(test_db_filepath):
            os.remove(test_db_filepath)
