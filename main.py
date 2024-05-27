import asyncio
import os
from datetime import datetime

from clock.clock import clock
from social_agent.agents_generator import generate_agents
from twitter.channel import Twitter_Channel
from twitter.twitter import Twitter
from twitter.typing import ActionType


async def running():
    test_db_filepath = "./data/mock_twitter.db"
    if os.path.exists(test_db_filepath):
        os.remove(test_db_filepath)

    # 将sandbox启动时间设为当前时刻
    start_time = datetime.now()
    # 将sandbox时间放大系数设为60，即系统运行1秒相当于现实世界60秒
    sandbox_clock = clock(K=60)

    channel = Twitter_Channel()

    # 如果不传入start_time或sandbox_clock，默认start_time为实例化Twitter的时间，sandbox_clock的K=60
    infra = Twitter(test_db_filepath, channel, sandbox_clock, start_time)

    task = asyncio.create_task(infra.running())

    agent_graph = await generate_agents("./data/user_all_id.csv", channel)

    for node_id, node_data in agent_graph.get_agents():
        agent = node_data['agent']
        await agent.perform_action_by_llm()

    await channel.write_to_receive_queue((None, None, ActionType.EXIT))
    await task


if __name__ == "__main__":
    asyncio.run(running())
