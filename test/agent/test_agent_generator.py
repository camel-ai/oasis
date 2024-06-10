# File: ./test/infra/test_agent_generator.py
import asyncio
import os
import os.path as osp

from social_agent.agents_generator import (generate_agents,
                                           generate_controllable_agents)
from twitter.channel import Twitter_Channel
from twitter.twitter import Twitter

parent_folder = osp.dirname(osp.abspath(__file__))
test_db_filepath = osp.join(parent_folder, "mock_twitter.db")
if osp.exists(test_db_filepath):
    os.remove(test_db_filepath)


async def running():
    agent_info_path = "./test/test_data/user_all_id_time.csv"
    channel = Twitter_Channel()
    infra = Twitter(test_db_filepath, channel)
    task = asyncio.create_task(infra.running())
    agent_graph = await generate_agents(agent_info_path, channel)
    await channel.write_to_receive_queue((None, None, "exit"))
    await task
    print(agent_graph)


def test_agent_generator():
    asyncio.run(running())


async def controllable_running():
    agent_info_path = "./test/test_data/user_all_id_time.csv"
    channel = Twitter_Channel()
    if osp.exists(test_db_filepath):
        os.remove(test_db_filepath)
    infra = Twitter(test_db_filepath, channel)
    task = asyncio.create_task(infra.running())
    agent_graph, agent_user_id_mapping = await generate_controllable_agents(
        channel, 1)
    agent_graph = await generate_agents(
        agent_info_path, channel, agent_graph, agent_user_id_mapping)
    await channel.write_to_receive_queue((None, None, "exit"))
    await task


def test_generate_controllable():
    asyncio.run(controllable_running())
