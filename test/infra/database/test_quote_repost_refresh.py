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
        # The first call returns the command to create a post
        if self.call_count == 0:
            self.call_count += 1
            return ("id_", (1, "This is a test post", "create_post"))
        elif self.call_count == 1:
            self.call_count += 1
            return ("id_", (1, 1, "like_post"))
        elif self.call_count == 2:
            self.call_count += 1
            return ("id_", (None, None, ActionType.UPDATE_REC_TABLE))
        elif self.call_count == 3:
            self.call_count += 1
            return ("id_", (1, None, ActionType.REFRESH))
        elif self.call_count == 4:
            self.call_count += 1
            return ("id_", (1, (1, "a comment"), "create_comment"))
        elif self.call_count == 5:
            self.call_count += 1
            return ("id_", (2, 1, "repost"))
        elif self.call_count == 6:
            self.call_count += 1
            return ("id_", (1, (1, 'I like the post.'), "quote_post"))
        elif self.call_count == 7:
            self.call_count += 1
            return ("id_", (2, (2, 'I quote to the reposted post.'),
                            "quote_post"))
        elif self.call_count == 8:
            self.call_count += 1
            return ("id_", (1, 4, "repost"))
        elif self.call_count == 9:
            self.call_count += 1
            return ("id_", (2, (4, 'I quote to the quoted post.'),
                            "quote_post"))
        elif self.call_count == 10:
            self.call_count += 1
            return ("id_", (None, None, ActionType.UPDATE_REC_TABLE))
        elif self.call_count == 11:
            self.call_count += 1
            return ("id_", (1, None, ActionType.REFRESH))
        else:
            return ("id_", (None, None, "exit"))

    async def send_to(self, message):
        # Store the message for subsequent assertions
        self.messages.append(message)
        if self.call_count == 1:
            assert message[2]["success"] is True
        if self.call_count == 2:
            assert message[2]["success"] is True
        elif self.call_count == 4:
            assert message[2]["success"] is True
            assert len(message[2]["posts"]) == 1
            post = message[2]["posts"][0]
            assert post['comments'] == []
            assert post['content'] == 'This is a test post'
            assert post['created_at'] is not None
            assert post['num_dislikes'] == 0
            assert post['num_likes'] == 1
            assert post['num_shares'] == 0
            assert post['post_id'] == 1
            assert post['user_id'] == 1
        elif self.call_count == 5:
            assert message[2]["success"] is True
        elif self.call_count == 6:
            # Assert the success message for the like operation
            assert message[2]["success"] is True
        elif self.call_count == 7:
            assert message[2]["success"] is True
        elif self.call_count == 8:
            assert message[2]["success"] is True
        elif self.call_count == 9:
            # Assert the success message for a repost
            assert message[2]["success"] is True
        elif self.call_count == 10:
            # Assert the success message for a repost
            assert message[2]["success"] is True
        elif self.call_count == 12:
            # Assert the success message for a repost
            assert message[2]["success"] is True
            posts = message[2]["posts"]
            assert len(posts) == 6
            # Post 1
            assert posts[0]['post_id'] == 1
            assert posts[0]['user_id'] == 1
            assert posts[0]['content'] == 'This is a test post'
            assert posts[0]['num_likes'] == 1
            assert posts[0]['num_shares'] == 4
            assert len(posts[0]['comments']) == 1
            assert posts[0]['comments'][0]['comment_id'] == 1
            assert posts[0]['comments'][0]['content'] == 'a comment'

            # Post 2
            assert posts[1]['post_id'] == 2
            assert posts[1]['user_id'] == 2
            assert posts[1]['content'] == (
                'User 2 reposted a post from User 1. Repost content: This is '
                'a test post. ')
            assert posts[1]['num_likes'] == 1

            # Post 3
            assert posts[2]['post_id'] == 3
            assert posts[2]['user_id'] == 1
            assert posts[2]['content'] == (
                'User 1 quoted a post from User 1. Quote content: I like the '
                'post.. Original Content: This is a test post')
            assert posts[2]['num_likes'] == 0

            # Post 4
            assert posts[3]['post_id'] == 4
            assert posts[3]['user_id'] == 2
            assert posts[3]['content'] == (
                'User 2 quoted a post from User 1. Quote content: I quote to '
                'the reposted post.. Original Content: This is a test post')
            assert posts[3]['num_likes'] == 0

            # Post 5
            assert posts[4]['post_id'] == 5
            assert posts[4]['user_id'] == 1
            assert posts[4]['content'] == (
                'User 1 reposted a post from User 2. Repost content: This is '
                'a test post. ')
            assert posts[4]['num_likes'] == 0

            # Post 6
            assert posts[5]['post_id'] == 6
            assert posts[5]['user_id'] == 2
            assert posts[5]['content'] == (
                'User 2 quoted a post from User 1. Quote content: I quote to '
                'the quoted post.. Original Content: This is a test post')
            assert posts[5]['num_likes'] == 0


@pytest.fixture
def setup_platform():
    # Ensure test.db does not exist before testing
    if os.path.exists(test_db_filepath):
        os.remove(test_db_filepath)

    # Create the database and tables
    db_path = test_db_filepath

    mock_channel = MockChannel()
    instance = Platform(db_path=db_path,
                        channel=mock_channel,
                        refresh_rec_post_count=10,
                        max_rec_post_len=10)
    return instance


@pytest.mark.asyncio
async def test_create_repost_like_unlike_post(setup_platform):
    try:
        platform = setup_platform

        # Insert two users into the user table before testing begins
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
            (3, 3, "user3", 2, 4),
        )
        conn.commit()

        await platform.running()

        # Verify the correct insertion of data into the database
        conn = sqlite3.connect(test_db_filepath)
        cursor = conn.cursor()

    finally:
        # Cleanup
        conn.close()
        if os.path.exists(test_db_filepath):
            os.remove(test_db_filepath)
