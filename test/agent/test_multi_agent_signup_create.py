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
# File: ./test/infra/test_multi_agent_signup_create.py
import asyncio
import os
import os.path as osp
import random
import sqlite3

import pytest

from oasis.social_agent.agent import SocialAgent
from oasis.social_platform.channel import Channel
from oasis.social_platform.config import UserInfo
from oasis.social_platform.platform import Platform
from oasis.testing.show_db import print_db_contents

parent_folder = osp.dirname(osp.abspath(__file__))
test_db_filepath = osp.join(parent_folder, "test_multi.db")


@pytest.fixture
def setup_platform():
    if os.path.exists(test_db_filepath):
        os.remove(test_db_filepath)


@pytest.mark.asyncio
async def test_agents_posting(setup_platform):
    N = 5  # number of agents(users)
    M = 3  # Number of posts each user wants to send

    agents = []
    channel = Channel()
    infra = Platform(test_db_filepath, channel)
    task = asyncio.create_task(infra.running())

    # 创建并注册用户
    for i in range(N):
        real_name = "name" + str(i)
        description = "No description."
        # profile = {"some_key": "some_value"}
        profile = {
            "nodes": [],  # Relationships with other agents
            "edges": [],  # Relationship details
            "other_info": {
                "user_profile": "Nothing",
                "mbti": "INTJ",
                "activity_level": ["off_line"] * 24,
                "activity_level_frequency": [3] * 24,
                "active_threshold": [0.1] * 24,
            },
        }
        user_info = UserInfo(name=real_name,
                             description=description,
                             profile=profile)
        agent = SocialAgent(agent_id=i, user_info=user_info, channel=channel)
        await agent.env.action.sign_up(f"user{i}0101", f"User{i}", "A bio.")
        agents.append(agent)

    # create post
    for agent in agents:
        for _ in range(M):
            await agent.env.action.create_post(f"hello from {agent.agent_id}")
            await asyncio.sleep(random.uniform(0, 0.1))

    await channel.write_to_receive_queue((None, None, "exit"))
    await task

    # Verify if data was correctly inserted into the database
    conn = sqlite3.connect(test_db_filepath)
    cursor = conn.cursor()
    print_db_contents(test_db_filepath)
    # Verify if data was correctly inserted into the user table
    cursor.execute("SELECT * FROM user")
    users = cursor.fetchall()
    assert len(
        users) == N, "The number of users in the database" "should match n"

    # Verify if data was correctly inserted into the post table
    cursor.execute("SELECT * FROM post")
    posts = cursor.fetchall()
    assert len(
        posts) == M * N, "The number of posts should match the expected value."
    cursor.close()
    conn.close()
