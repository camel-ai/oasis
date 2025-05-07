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
from oasis import (ActionType, AgentGraph, EnvAction, SingleAction,
                   SocialAgent, UserInfo)


async def main():
    # Define the model for the agents
    openai_model = ModelFactory.create(
        model_platform=ModelPlatformType.OPENAI,
        model_type=ModelType.GPT_4O_MINI,
    )
    qwen_model = ModelFactory.create(
        model_platform=ModelPlatformType.QWEN,
        model_type=ModelType.QWEN_TURBO,
    )

    # Define the available actions for the agents
    available_actions = [
        ActionType.LIKE_POST,
        ActionType.DISLIKE_POST,
        ActionType.CREATE_POST,
        ActionType.CREATE_COMMENT,
        ActionType.LIKE_COMMENT,
        ActionType.DISLIKE_COMMENT,
        ActionType.SEARCH_POSTS,
        ActionType.SEARCH_USER,
        ActionType.TREND,
        ActionType.REFRESH,
        ActionType.DO_NOTHING,
        ActionType.FOLLOW,
        ActionType.MUTE,
    ]

    agent_graph = AgentGraph()
    agent_1 = SocialAgent(
        agent_id=0,
        user_info=UserInfo(
            user_name="ali",
            name="Alice",
            description="A girl",
            profile=None,
            recsys_type="reddit",
        ),  # user_info can be customized
        agent_graph=agent_graph,  # the inital AgentGraph
        model=openai_model,  # BaseModelBackend
        available_actions=available_actions,  # List[ActionType]
    )
    agent_graph.add_agent(agent_1)

    agent_2 = SocialAgent(
        agent_id=1,
        user_info=UserInfo(
            user_name="bubble",
            name="Bob",
            description="A boy",
            profile=None,
            recsys_type="reddit",
        ),  # user_info can be customized
        agent_graph=agent_graph,  # the inital AgentGraph
        model=qwen_model,  # BaseModelBackend
        available_actions=available_actions,  # List[ActionType]
    )
    agent_graph.add_agent(agent_2)

    # Define the path to the database
    db_path = "./data/reddit_simulation.db"

    # Delete the old database
    if os.path.exists(db_path):
        os.remove(db_path)

    # Make the environment
    env = oasis.make(
        agent_graph=agent_graph,
        platform=oasis.DefaultPlatformType.REDDIT,
        database_path=db_path,
    )

    # Run the environment
    await env.reset()

    action_1 = SingleAction(agent_id=0,
                            action=ActionType.CREATE_POST,
                            args={"content": "Hello, world!"})
    action_2 = SingleAction(agent_id=0,
                            action=ActionType.CREATE_COMMENT,
                            args={
                                "post_id": "1",
                                "content": "Welcome to the OASIS World!"
                            })

    env_actions = EnvAction(activate_agents=[0, 1],
                            intervention=[action_1, action_2])

    # Perform the actions
    await env.step(env_actions)

    # Close the environment
    await env.close()


if __name__ == "__main__":
    asyncio.run(main())
