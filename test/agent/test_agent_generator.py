# File: ./test/infra/test_agent_generator.py
import asyncio
import os
import os.path as osp

from social_agent.agents_generator import generate_agents
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
