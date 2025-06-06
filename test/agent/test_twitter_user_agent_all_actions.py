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
import asyncio
import os
import os.path as osp
import random

import pytest

from oasis.social_agent.agent import SocialAgent
from oasis.social_platform.channel import Channel
from oasis.social_platform.config import UserInfo
from oasis.social_platform.platform import Platform
from oasis.social_platform.typing import ActionType

parent_folder = osp.dirname(osp.abspath(__file__))
test_db_filepath = osp.join(parent_folder, "test_actions.db")


@pytest.fixture
def setup_twitter():
    if os.path.exists(test_db_filepath):
        os.remove(test_db_filepath)


@pytest.mark.asyncio
async def test_agents_actions(setup_twitter):
    agents = []
    channel = Channel()
    infra = Platform(test_db_filepath, channel)
    task = asyncio.create_task(infra.running())

    # create and sign up users
    for i in range(3):
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
        return_message = await agent.env.action.sign_up(
            f"user{i}0101", f"User{i}", "A bio.")
        assert return_message["success"] is True
        agents.append(agent)

    # create post
    for agent in agents:
        for _ in range(4):
            return_message = await agent.env.action.create_post(
                f"hello from {agent.agent_id}", )
            await asyncio.sleep(random.uniform(0, 0.1))
            assert return_message["success"] is True

    await channel.write_to_receive_queue(
        (None, None, ActionType.UPDATE_REC_TABLE))

    # Look at the posts returned by the recommendation system
    action_agent = agents[2]
    return_message = await action_agent.env.action.refresh()
    assert return_message["success"] is True
    await asyncio.sleep(random.uniform(0, 0.1))

    return_message = await action_agent.env.action.like_post(1)
    print(return_message)
    assert return_message["success"] is True
    await asyncio.sleep(random.uniform(0, 0.1))

    return_message = await action_agent.env.action.unlike_post(1)
    assert return_message["success"] is True
    await asyncio.sleep(random.uniform(0, 0.1))

    return_message = await action_agent.env.action.dislike_post(1)
    assert return_message["success"] is True
    await asyncio.sleep(random.uniform(0, 0.1))

    return_message = await action_agent.env.action.undo_dislike_post(1)
    assert return_message["success"] is True
    await asyncio.sleep(random.uniform(0, 0.1))

    return_message = await action_agent.env.action.search_posts("hello")
    assert return_message["success"] is True
    await asyncio.sleep(random.uniform(0, 0.1))

    return_message = await action_agent.env.action.search_user("2")
    assert return_message["success"] is True
    await asyncio.sleep(random.uniform(0, 0.1))

    return_message = await action_agent.env.action.follow(1)
    assert return_message["success"] is True
    await asyncio.sleep(random.uniform(0, 0.1))

    return_message = await action_agent.env.action.unfollow(1)
    assert return_message["success"] is True
    await asyncio.sleep(random.uniform(0, 0.1))

    return_message = await action_agent.env.action.mute(1)
    assert return_message["success"] is True
    await asyncio.sleep(random.uniform(0, 0.1))

    return_message = await action_agent.env.action.unmute(1)
    assert return_message["success"] is True
    await asyncio.sleep(random.uniform(0, 0.1))

    # Check the most popular post
    return_message = await action_agent.env.action.trend()
    assert return_message["success"] is True
    await asyncio.sleep(random.uniform(0, 0.1))

    # repost
    return_message = await action_agent.env.action.repost(1)
    assert return_message["success"] is True
    await asyncio.sleep(random.uniform(0, 0.1))

    return_message = await action_agent.env.action.quote_post(1, "Test quote")
    assert return_message["success"] is True
    await asyncio.sleep(random.uniform(0, 0.1))

    return_message = await action_agent.env.action.create_comment(
        1, "Test comment")
    assert return_message["success"] is True
    await asyncio.sleep(random.uniform(0, 0.1))

    # report once
    return_message = await action_agent.env.action.report_post(
        1, "Inappropriate content")
    assert return_message["success"] is True
    await asyncio.sleep(random.uniform(0, 0.1))

    # report twice
    other_agent = agents[1]
    return_message = await other_agent.env.action.report_post(
        1, "Spam content")
    assert return_message["success"] is True
    await asyncio.sleep(random.uniform(0, 0.1))

    return_message = await action_agent.env.action.like_comment(1)
    assert return_message["success"] is True
    await asyncio.sleep(random.uniform(0, 0.1))

    return_message = await action_agent.env.action.unlike_comment(1)
    assert return_message["success"] is True
    await asyncio.sleep(random.uniform(0, 0.1))

    return_message = await action_agent.env.action.dislike_comment(1)
    assert return_message["success"] is True
    await asyncio.sleep(random.uniform(0, 0.1))

    return_message = await action_agent.env.action.undo_dislike_comment(1)
    assert return_message["success"] is True
    await asyncio.sleep(random.uniform(0, 0.1))

    return_message = await action_agent.env.action.do_nothing()
    print(return_message)
    assert return_message["success"] is True
    await asyncio.sleep(random.uniform(0, 0.1))

    await infra.sign_up_product(1, "apple")
    return_message = await action_agent.env.action.purchase_product("apple", 1)
    assert return_message["success"] is True
    await asyncio.sleep(random.uniform(0, 0.1))

    return_message = await action_agent.env.action.create_group("AI Discussion"
                                                                )
    assert return_message["success"] is True
    await asyncio.sleep(random.uniform(0, 0.1))

    join_agent = agents[0]
    return_message = await join_agent.env.action.join_group(1)
    print(return_message)
    assert return_message["success"] is True
    await asyncio.sleep(random.uniform(0, 0.1))

    return_message = await action_agent.env.action.send_to_group(
        1, "Hello, everyone!")
    assert return_message["success"] is True
    await asyncio.sleep(random.uniform(0, 0.1))

    return_message = await action_agent.env.action.listen_from_group()
    assert return_message["success"] is True
    await asyncio.sleep(random.uniform(0, 0.1))

    return_message = await action_agent.env.action.leave_group(1)
    print(return_message)
    assert return_message["success"] is True
    await asyncio.sleep(random.uniform(0, 0.1))

    await channel.write_to_receive_queue((None, None, ActionType.EXIT))
    await task
