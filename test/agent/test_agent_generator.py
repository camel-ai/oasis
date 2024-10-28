# File: ./test/infra/test_agent_generator.py
import asyncio
import os
import os.path as osp

import pytest
from camel.types import ModelType

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
    inferencer_channel = Channel()
    infra = Platform(test_db_filepath, twitter_channel)
    task = asyncio.create_task(infra.running())
    os.environ["SANDBOX_TIME"] = "0"
    agent_graph = await generate_agents(
        agent_info_path,
        twitter_channel,
        inferencer_channel,
        twitter=infra,
        start_time=0,
        num_agents=111,
        cfgs=[
            {
                "model_type": ModelType.LLAMA_3,
                "num": 100
            },
            {
                "model_type": ModelType.GPT_3_5_TURBO,
                "num": 11
            },
        ],
    )
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
    inferencer_channel = Channel()
    if osp.exists(test_db_filepath):
        os.remove(test_db_filepath)
    infra = Platform(test_db_filepath, twitter_channel)
    task = asyncio.create_task(infra.running())
    inputs = iter(["Alice", "Ali", "a student"])
    monkeypatch.setattr("builtins.input", lambda _: next(inputs))
    agent_graph, agent_user_id_mapping = await generate_controllable_agents(
        twitter_channel, 1)
    agent_graph = await generate_agents(
        agent_info_path,
        twitter_channel,
        inferencer_channel,
        agent_graph,
        agent_user_id_mapping,
    )
    await twitter_channel.write_to_receive_queue((None, None, "exit"))
    await task
    assert agent_graph.get_num_nodes() == 27
