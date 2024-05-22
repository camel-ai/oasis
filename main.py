import asyncio

from camel.types import ModelType

from social_agent.agents_generator import generate_agents
from twitter.channel import TwitterChannel
from twitter.twitter import Twitter
from twitter.typing import ActionType


async def running():
    test_db_filepath = "./data/mock_twitter.db"

    channel = TwitterChannel()
    infra = Twitter(test_db_filepath, channel)
    task = asyncio.create_task(infra.running())
    model_type = ModelType.GPT_3_5_TURBO

    agent_graph = await generate_agents(
        "./data/user_all_id.csv",
        channel,
        model_type,
    )

    for node_id, node_data in agent_graph.get_agents():
        agent = node_data['agent']
        await agent.perform_action_by_llm()

    await channel.write_to_receive_queue((None, None, ActionType.EXIT))
    await task


if __name__ == "__main__":
    asyncio.run(running())
