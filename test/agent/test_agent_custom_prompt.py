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
import warnings

import pytest
from camel.prompts import TextPrompt

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
async def test_agents_profile():
    channel = Channel()
    user_info_template = TextPrompt('Your aim is: {aim} Your task is: {task}')

    profile = {
        "aim": "Persuade people to buy a product.",
        "task": "Using roleplay to tell some story about the product.",
    }
    user_info = UserInfo(name=None, description=None, profile=profile)
    agent = SocialAgent(agent_id=0,
                        user_info=user_info,
                        user_info_template=user_info_template,
                        channel=channel)
    assert agent.system_message.content == (
        'Your aim is: Persuade people to buy a product. Your task is: '
        'Using roleplay to tell some story about the product.')
    # test missing keys
    profile = {"aim": "Persuade people to buy a product."}
    user_info = UserInfo(name=None, description=None, profile=profile)
    with pytest.raises(ValueError) as excinfo:
        user_info.to_custom_system_message(user_info_template)
    assert "Missing required keys" in str(excinfo.value)
    # test extra keys
    profile = {
        "aim": "Persuade people to buy a product.",
        "task": "Using roleplay to tell some story about the product.",
        "extra_key": "extra_value"
    }
    user_info = UserInfo(name=None, description=None, profile=profile)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        user_info.to_custom_system_message(user_info_template)
        assert any("Extra keys not used" in str(warn.message) for warn in w)


@pytest.mark.asyncio
async def test_agents_posting(setup_platform):
    channel = Channel()
    infra = Platform(db_path=test_db_filepath,
                     channel=channel,
                     recsys_type='reddit')
    task = asyncio.create_task(infra.running())

    profile = {"name": "Maximiliano"}
    user_info_template = TextPrompt('Your name is {name}.')
    user_info = UserInfo(name=None, description=None, profile=profile)
    agent = SocialAgent(agent_id=0,
                        user_info=user_info,
                        user_info_template=user_info_template,
                        channel=channel)
    await agent.env.action.sign_up("user0", "User0", "A bio.")

    # create post
    await agent.env.action.create_post("Can you create a post about your name?"
                                       )

    await infra.update_rec_table()

    response = await agent.perform_action_by_llm()

    assert any(tool_call.tool_name == "create_post"
               and "Maximiliano" in tool_call.args.get("content", "")
               for tool_call in response.info['tool_calls'])
    await channel.write_to_receive_queue((None, None, "exit"))
    await task
