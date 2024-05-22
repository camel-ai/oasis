import asyncio

from social_agent.agents_generator import generate_agents
from twitter.channel import Twitter_Channel
from twitter.twitter import Twitter
from twitter.typing import ActionType
import os

async def running():

    os.environ["OPENAI_API_KEY"] = "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"
    test_db_filepath = "./data/mock_twitter.db"

    channel = Twitter_Channel()
    infra = Twitter(test_db_filepath, channel)
    task = asyncio.create_task(infra.running())

    agent_graph = await generate_agents("./data/user_all_id_time.csv", channel)

    for node_id, node_data in agent_graph.get_agents():
        agent = node_data['agent']
        await agent.perform_action_by_llm()

    await channel.write_to_receive_queue((None, None, ActionType.EXIT))
    await task


if __name__ == "__main__":
    asyncio.run(running())