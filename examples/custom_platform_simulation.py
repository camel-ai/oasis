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

from camel.models import ModelFactory
from camel.types import ModelPlatformType, ModelType

import oasis
from oasis import (ActionType, EnvAction, Platform, SingleAction,
                   generate_twitter_agent_graph)
from oasis.social_platform.channel import Channel
from oasis.social_platform.typing import RecsysType


async def main():
    # Define the models for agents.
    models = ModelFactory.create(
        model_platform=ModelPlatformType.OPENAI,
        model_type=ModelType.GPT_4O_MINI,
    )

    # Define the available actions for the agents
    available_actions = [
        ActionType.CREATE_POST,
        ActionType.LIKE_POST,
        ActionType.REPOST,
        ActionType.FOLLOW,
        ActionType.DO_NOTHING,
    ]

    agent_graph = await generate_twitter_agent_graph(
        profile_path=("data/twitter_dataset/anonymous_topic_200_1h/"
                      "False_Business_0.csv"),
        model=models,
        available_actions=available_actions,
    )

    # Define the path to the database
    db_path = "./data/twitter_simulation.db"

    # Delete the old database
    if os.path.exists(db_path):
        os.remove(db_path)

    channel = Channel()
    customize_platform = Platform(
        db_path=db_path,
        channel=channel,
        recsys_type=RecsysType.TWHIN,
        allow_self_rating=False,  # Not allow agents to rate themselves
        show_score=True,  # Show the scores of posts for agents, instead of
        # only showing the number of likes and dislikes
    )

    # Make the environment
    env = oasis.make(
        agent_graph=agent_graph,
        platform=customize_platform,
    )

    # Run the environment
    await env.reset()

    action_1 = SingleAction(agent_id=0,
                            action=ActionType.CREATE_POST,
                            args={"content": "Earth is flat."})
    env_actions_1 = EnvAction(
        # Activate 5 agents with id 1, 3, 5, 7, 9
        activate_agents=[1, 3, 5, 7, 9],
        intervention=[action_1])

    action_2 = SingleAction(agent_id=1,
                            action=ActionType.CREATE_POST,
                            args={"content": "Earth is not flat."})
    env_actions_2 = EnvAction(activate_agents=[2, 4, 6, 8, 10],
                              intervention=[action_2])

    all_env_actions = [
        env_actions_1,
        env_actions_2,
    ]

    # Simulate 2 timesteps
    for i in range(2):
        env_actions = all_env_actions[i]
        # Perform the actions
        await env.step(env_actions)

    # Close the environment
    await env.close()


if __name__ == "__main__":
    asyncio.run(main())
