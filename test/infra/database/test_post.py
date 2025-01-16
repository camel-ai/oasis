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
        self.messages = []

    async def receive_from(self):
        # The first call returns the command to create a post
        if self.call_count == 0:
            self.call_count += 1
            return ("id_", (1, "This is a test post", "create_post"))
        # The second call returns the command for a like operation
        elif self.call_count == 1:
            self.call_count += 1
            return ("id_", (1, 1, "like_post"))
        elif self.call_count == 2:
            self.call_count += 1
            return ("id_", (2, 1, "like_post"))
        elif self.call_count == 3:
            self.call_count += 1
            return ("id_", (2, 1, "unlike_post"))
        elif self.call_count == 4:
            self.call_count += 1
            return ("id_", (1, 1, "dislike_post"))
        elif self.call_count == 5:
            self.call_count += 1
            return ("id_", (2, 1, "dislike_post"))
        elif self.call_count == 6:
            self.call_count += 1
            return ("id_", (2, 1, "undo_dislike_post"))
        # The call returns the command for a repost operation
        elif self.call_count == 7:
            self.call_count += 1
            return ("id_", (2, 1, "repost"))
        elif self.call_count == 8:
            self.call_count += 1
            return ("id_", (2, 2, "like_post"))
        elif self.call_count == 9:
            self.call_count += 1
            return ("id_", (2, 1, "repost"))
        elif self.call_count == 10:
            self.call_count += 1
            return ("id_", (2, 3, "repost"))
        elif self.call_count == 11:
            self.call_count += 1
            return ("id_", (3, 2, "repost"))
        elif self.call_count == 12:
            self.call_count += 1
            return ("id_", (1, (1, 'I like the post.'), "quote_post"))
        elif self.call_count == 13:
            self.call_count += 1
            return ("id_", (2, (2, 'I quote to the reposted post.'),
                            "quote_post"))
        elif self.call_count == 14:
            self.call_count += 1
            return ("id_", (1, 4, "repost"))
        elif self.call_count == 15:
            self.call_count += 1
            return ("id_", (2, (4, 'I quote to the quoted post.'),
                            "quote_post"))
        # Returns the exit command
        else:
            return ("id_", (None, None, "exit"))

    async def send_to(self, message):
        # Store the message for subsequent assertions
        self.messages.append(message)
        if self.call_count == 1:
            # Assert the success message for creating a post
            assert message[2]["success"] is True
            assert "post_id" in message[2]
        elif self.call_count == 2:
            # Assert the success message for the like operation
            assert message[2]["success"] is True
            assert "like_id" in message[2]
        elif self.call_count == 3:
            assert message[2]["success"] is True
            assert "like_id" in message[2]
        elif self.call_count == 4:
            assert message[2]["success"] is True
            assert "like_id" in message[2]
        elif self.call_count == 5:
            # Assert the success message for the like operation
            assert message[2]["success"] is True
            assert "dislike_id" in message[2]
        elif self.call_count == 6:
            assert message[2]["success"] is True
            assert "dislike_id" in message[2]
        elif self.call_count == 7:
            assert message[2]["success"] is True
            assert "dislike_id" in message[2]
        elif self.call_count == 8:
            # Assert the success message for a repost
            assert message[2]["success"] is True
            assert "post_id" in message[2]
        elif self.call_count == 9:
            assert message[2]["success"] is True
            assert "like_id" in message[2]
        elif self.call_count == 10:
            # Assert the success message for a repost
            assert message[2]["success"] is False
            assert message[2]["error"] == "Repost record already exists."
        elif self.call_count == 11:
            # Assert the success message for a repost
            assert message[2]["success"] is False
            assert message[2]["error"] == "Post not found."
        elif self.call_count == 12:
            assert message[2]["success"] is True
            assert "post_id" in message[2]
        elif self.call_count == 13:
            # Assert the success message for a repost
            assert message[2]["success"] is True
            assert "post_id" in message[2]
        elif self.call_count == 14:
            # Assert the success message for a repost
            assert message[2]["success"] is True
            assert "post_id" in message[2]
        elif self.call_count == 15:
            assert message[2]["success"] is True
            assert "post_id" in message[2]
        elif self.call_count == 16:
            # Assert the success message for a repost
            assert message[2]["success"] is True
            assert "post_id" in message[2]


@pytest.fixture
def setup_platform():
    # Ensure test.db does not exist before testing
    if os.path.exists(test_db_filepath):
        os.remove(test_db_filepath)

    # Create the database and tables
    db_path = test_db_filepath

    mock_channel = MockChannel()
    instance = Platform(db_path, mock_channel)
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

        # Verify the post table (post) has the correct data inserted
        cursor.execute("SELECT * FROM post")
        posts = cursor.fetchall()
        assert len(posts) == 7  # One test post, one repost
        post = posts[0]
        assert post[1] == 1  # Assuming user ID is 1
        assert post[3] == "This is a test post"
        assert post[6] == 2  # num_likes
        assert post[7] == 1  # num_dislikes
        assert post[8] == 5  # num_shares

        repost = posts[1]
        assert repost[1] == 2  # Repost user ID is 2
        assert repost[2] == 1  # Original post ID is 1
        assert repost[3] == ''  # Reposted post is empty
        print('created_at:', repost[5])
        assert repost[5] is not None  # created_at
        assert repost[6] == 0  # num_likes

        repost_2 = posts[2]
        assert repost_2[1] == 3  # Repost user ID is 2
        assert repost_2[2] == 1  # Original post ID is 1
        assert repost_2[3] == ''  # Reposted post is empty
        assert repost[5] is not None  # created_at

        quote_post = posts[3]
        assert quote_post[1] == 1  # Repost user ID is
        assert quote_post[2] == 1  # Original post ID is 1
        assert quote_post[3] == "This is a test post"
        assert quote_post[4] == "I like the post."

        quote_post_2 = posts[4]
        assert quote_post_2[1] == 2  # Repost user ID is 2
        assert quote_post_2[2] == 1  # Original post ID is 1
        assert quote_post[3] == "This is a test post"

        repost_quote_post = posts[5]
        assert repost_quote_post[2] == 4  # Original post ID is 4
        assert quote_post[3] == "This is a test post"

        quote_post_4 = posts[6]
        assert quote_post_4[2] == 1  # Original post ID is 4
        assert quote_post[3] == "This is a test post"

        # Verify the like table has the correct data inserted
        cursor.execute("SELECT * FROM like")
        likes = cursor.fetchall()
        assert len(likes) == 2

        # Verify the dislike table has the correct data inserted
        cursor.execute("SELECT * FROM dislike")
        dislikes = cursor.fetchall()
        assert len(dislikes) == 1

        # Verify the trace table correctly recorded the create post and like
        cursor.execute("SELECT * FROM trace WHERE action='create_post'")
        assert cursor.fetchone() is not None, "Create post action not traced"

        cursor.execute("SELECT * FROM trace WHERE action='repost'")
        assert cursor.fetchone() is not None, "Repost action not traced"

        cursor.execute("SELECT * FROM trace WHERE action='like_post'")
        results = cursor.fetchall()
        assert results is not None, "Like post action not traced"
        assert len(results) == 3

        cursor.execute("SELECT * FROM trace WHERE action='unlike_post'")
        results = cursor.fetchall()
        assert results is not None, "Unlike post action not traced"
        assert results[0][0] == 2  # `user_id`
        assert results[0][-1] == '{"post_id": 1, "like_id": 2}'

        cursor.execute("SELECT * FROM trace WHERE action='dislike_post'")
        results = cursor.fetchall()
        assert results is not None, "Dislike post action not traced"
        assert len(results) == 2

        cursor.execute("SELECT * FROM trace WHERE action='undo_dislike_post'")
        results = cursor.fetchall()
        assert results is not None, "Undo dislike post action not traced"
        assert results[0][0] == 2  # `user_id`
        assert results[0][-1] == '{"post_id": 1, "dislike_id": 2}'

        cursor.execute("SELECT * FROM trace WHERE action='quote_post'")
        results = cursor.fetchall()
        assert results is not None, "Quote post action not traced"
        assert len(results) == 3

        # Verify the like table has the correct data for a like
        cursor.execute("SELECT * FROM like WHERE post_id=1 AND user_id=1")
        assert cursor.fetchone() is not None, "Like record not found"

        # Verify the dislike table has the correct data for a dislike
        cursor.execute("SELECT * FROM dislike WHERE post_id=1 AND user_id=1")
        assert cursor.fetchone() is not None, "Like record not found"

        # Verify the dislike table has the correct data for a dislike
        cursor.execute("SELECT * FROM trace WHERE user_id=3")
        result = cursor.fetchone()
        assert result[3] == '{"reposted_id": 2, "new_post_id": 3}'

    finally:
        # Cleanup
        conn.close()
        if os.path.exists(test_db_filepath):
            os.remove(test_db_filepath)
