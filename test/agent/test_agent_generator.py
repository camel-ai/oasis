# File: ./test/infra/test_agent_generator.py
import asyncio
import os
import os.path as osp

import pytest
from camel.types import ModelType

from social_simulation.social_agent.agents_generator import (
    generate_agents, generate_controllable_agents)
from social_simulation.social_platform.channel import Channel
from social_simulation.social_platform.platform import Platform

parent_folder = osp.dirname(osp.abspath(__file__))
test_db_filepath = osp.join(parent_folder, "test.db")
if osp.exists(test_db_filepath):
    os.remove(test_db_filepath)


async def running():
    agent_info_path = "./test/test_data/user_all_id_time.csv"
    channel = Channel()
    infra = Platform(test_db_filepath, channel)
    task = asyncio.create_task(infra.running())
    agent_graph = await generate_agents(
        agent_info_path,
        channel,
        num_agents=26,
        cfgs=[{
            "model_type": ModelType.LLAMA_3,
            "num": 20
        }, {
            "model_type": ModelType.GPT_3_5_TURBO,
            "num": 6
        }],
    )
    await channel.write_to_receive_queue((None, None, "exit"))
    await task
    assert agent_graph.get_num_nodes() == 26


def test_agent_generator():
    asyncio.run(running())


@pytest.mark.skip(reason="Now controllable agent is not supported")
@pytest.mark.asyncio
async def test_generate_controllable(monkeypatch):
    agent_info_path = "./test/test_data/user_all_id_time.csv"
    channel = Channel()
    if osp.exists(test_db_filepath):
        os.remove(test_db_filepath)
    infra = Platform(test_db_filepath, channel)
    task = asyncio.create_task(infra.running())
    inputs = iter(["Alice", "Ali", "a student"])
    monkeypatch.setattr('builtins.input', lambda _: next(inputs))
    agent_graph, agent_user_id_mapping = await generate_controllable_agents(
        channel, 1)
    agent_graph = await generate_agents(agent_info_path, channel, agent_graph,
                                        agent_user_id_mapping)
    await channel.write_to_receive_queue((None, None, "exit"))
    await task
    assert agent_graph.get_num_nodes() == 27
