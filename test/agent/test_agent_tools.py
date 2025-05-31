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

import pytest
from camel.toolkits import MathToolkit

from oasis import ActionType
from oasis.social_agent.agent import SocialAgent
from oasis.social_platform.channel import Channel
from oasis.social_platform.config import UserInfo
from oasis.social_platform.platform import Platform

parent_folder = osp.dirname(osp.abspath(__file__))
test_db_filepath = osp.join(parent_folder, "test_multi.db")


@pytest.fixture
def setup_platform():
    if os.path.exists(test_db_filepath):
        os.remove(test_db_filepath)


@pytest.mark.asyncio
async def test_agents_posting(setup_platform):
    agents = []
    channel = Channel()
    infra = Platform(db_path=test_db_filepath,
                     channel=channel,
                     recsys_type='reddit')
    task = asyncio.create_task(infra.running())

    for i in range(2):
        real_name = "name" + str(i)
        description = "No description."
        # profile = {"some_key": "some_value"}
        profile = {
            "nodes": [],  # Relationships with other agents
            "edges": [],  # Relationship details
            "other_info": {
                "user_profile": "Please",
                "mbti": "INTJ",
                "activity_level": ["off_line"] * 24,
                "activity_level_frequency": [3] * 24,
                "active_threshold": [0.1] * 24,
            },
        }
        user_info = UserInfo(name=real_name,
                             description=description,
                             profile=profile)
        agent = SocialAgent(agent_id=0,
                            user_info=user_info,
                            channel=channel,
                            tools=MathToolkit().get_tools(),
                            available_actions=[ActionType.CREATE_POST],
                            single_iteration=False)
        await agent.env.action.sign_up(f"user{i}", f"User{i}", "A bio.")
        agents.append(agent)

    # create post
    await agents[0].env.action.create_post(
        "Can someone tell me the result of 2025*1119? "
        "Please use the multiply tool")

    await infra.update_rec_table()

    response = await agents[1].perform_action_by_llm()

    assert any(tool_call.tool_name == "multiply"
               for tool_call in response.info['tool_calls'])

    await channel.write_to_receive_queue((None, None, "exit"))
    await task
