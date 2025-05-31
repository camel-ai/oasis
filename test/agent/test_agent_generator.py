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
# File: ./test/infra/test_agent_generator.py
import asyncio
import os
import os.path as osp

import pytest
from camel.models import ModelFactory
from camel.types import ModelPlatformType

from oasis.social_agent.agents_generator import (generate_agents,
                                                 generate_controllable_agents)
from oasis.social_platform.channel import Channel
from oasis.social_platform.platform import Platform

parent_folder = osp.dirname(osp.abspath(__file__))
test_db_filepath = osp.join(parent_folder, "test.db")
if osp.exists(test_db_filepath):
    os.remove(test_db_filepath)


async def running():
    agent_info_path = "./test/test_data/user_all_id_time.csv"
    twitter_channel = Channel()
    model = ModelFactory.create(
        model_platform=ModelPlatformType.OPENAI,
        model_type='gpt-4o-mini',
    )
    infra = Platform(test_db_filepath, twitter_channel)
    task = asyncio.create_task(infra.running())
    os.environ["SANDBOX_TIME"] = "0"
    agent_graph = await generate_agents(agent_info_path=agent_info_path,
                                        channel=twitter_channel,
                                        model=model,
                                        twitter=infra,
                                        start_time=0)
    await twitter_channel.write_to_receive_queue((None, None, "exit"))
    await task
    assert agent_graph.get_num_nodes() == 111


def test_agent_generator():
    asyncio.run(running())


@pytest.mark.skip(reason="Now controllable agent is not supported")
# @pytest.mark.asyncio
async def test_generate_controllable(monkeypatch):
    agent_info_path = "./test/test_data/user_all_id_time.csv"
    twitter_channel = Channel()
    if osp.exists(test_db_filepath):
        os.remove(test_db_filepath)
    infra = Platform(test_db_filepath, twitter_channel)
    task = asyncio.create_task(infra.running())
    inputs = iter(["Alice", "Ali", "a student"])
    monkeypatch.setattr("builtins.input", lambda _: next(inputs))
    agent_graph, agent_user_id_mapping = await generate_controllable_agents(
        twitter_channel, 1)
    agent_graph = await generate_agents(agent_info_path, twitter_channel,
                                        agent_graph, agent_user_id_mapping)
    await twitter_channel.write_to_receive_queue((None, None, "exit"))
    await task
    assert agent_graph.get_num_nodes() == 27
