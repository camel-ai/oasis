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
from oasis.testing.show_db import print_db_contents

parent_folder = osp.dirname(osp.abspath(__file__))
test_db_filepath = osp.join(parent_folder, "test.db")


class MockChannel:

    def __init__(self, user_actions):
        """
        user_actions: A list of tuples representing actions.
        Each tuple is in the format (user_id, message, action_type)
        """
        self.user_actions = user_actions
        self.messages = []
        self.action_index = 0  # Track the current action

    async def receive_from(self):
        if self.action_index < len(self.user_actions):
            action = self.user_actions[self.action_index]
            self.action_index += 1
            return ("id_", action)
        else:
            return ("id_", (None, None, "exit"))

    async def send_to(self, message):
        self.messages.append(message)


def generate_user_actions(n_users, posts_per_user):
    """
    Generate a list of user actions for n users with different posting
    behaviors. 1/3 of the users each sending m posts, 1/3 sending 1 post,
    and 1/3 not posting at all.
    """
    actions = []
    users_per_group = n_users // 3

    for user_id in range(1, n_users + 1):
        # Add sign up action for each user
        user_message = (
            "username" + str(user_id),
            "name" + str(user_id),
            "No descrption.",
        )
        actions.append((user_id, user_message, "sign_up"))

        if user_id <= users_per_group:
            # This group of users sends m posts each
            for post_num in range(1, posts_per_user + 1):
                actions.append((
                    user_id,
                    f"This is post {post_num} from User{user_id}",
                    "create_post",
                ))
        elif user_id <= 2 * users_per_group:
            # This group of users sends 1 post each
            actions.append(
                (user_id, f"This is post 1 from User{user_id}", "create_post"))
        # The last group does not send any posts

    return actions


@pytest.fixture
def setup_platform():
    if os.path.exists(test_db_filepath):
        os.remove(test_db_filepath)


@pytest.mark.asyncio
async def test_signup_and_create_post(setup_platform,
                                      n_users=30,
                                      posts_per_user=4):
    try:
        # To simplify the simulation, assume that n_users is a multiple of 3.
        assert n_users % 3 == 0, "n_users should be a multiple of 3."

        # Generate user actions based on n_users and posts_per_user
        user_actions = generate_user_actions(n_users, posts_per_user)

        mock_channel = MockChannel(user_actions)
        platform_instance = Platform(test_db_filepath, mock_channel)

        await platform_instance.running()

        conn = sqlite3.connect(test_db_filepath)
        cursor = conn.cursor()

        cursor.execute("SELECT * FROM user")
        users = cursor.fetchall()
        assert len(users) == n_users, ("The number of users in the database"
                                       "should match n_users.")

        cursor.execute("SELECT * FROM post")
        posts = cursor.fetchall()
        expected_posts = (n_users // 3) * posts_per_user + (n_users // 3)
        assert (len(posts) == expected_posts
                ), "The number of posts should match the expected value."

        conn.close()
        print_db_contents(test_db_filepath)

    finally:
        if os.path.exists(test_db_filepath):
            os.remove(test_db_filepath)
