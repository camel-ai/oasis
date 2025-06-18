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
from oasis import (ActionType, AgentGraph, LLMAction, ManualAction,
                   SocialAgent, UserInfo)


async def main():
    # Define the model for the agents
    openai_model = ModelFactory.create(
        model_platform=ModelPlatformType.OPENAI,
        model_type=ModelType.GPT_4O_MINI,
    )

    # Define the available actions for the agents
    available_actions = [
        ActionType.LIKE_POST,
        ActionType.CREATE_POST,
        ActionType.CREATE_COMMENT,
        ActionType.FOLLOW,
    ]

    # initialize the agent graph
    agent_graph = AgentGraph()

    # initialize the agent alice and add it to the agent graph
    agent_alice = SocialAgent(
        agent_id=0,
        user_info=UserInfo(
            user_name="alice",
            name="Alice",
            description="A tech enthusiast and a fan of OASIS",
            profile=None,
            recsys_type="reddit",
        ),
        agent_graph=agent_graph,
        model=openai_model,
        available_actions=available_actions,
    )
    agent_graph.add_agent(agent_alice)

    # initialize the agent bob and add it to the agent graph
    agent_bob = SocialAgent(
        agent_id=1,
        user_info=UserInfo(
            user_name="bob",
            name="Bob",
            description=("A researcher of using OASIS to research "
                         "the social behavior of users"),
            profile=None,
            recsys_type="reddit",
        ),
        agent_graph=agent_graph,
        model=openai_model,
        available_actions=available_actions,
    )
    agent_graph.add_agent(agent_bob)

    # Define the path to the database
    db_path = "./reddit_simulation.db"
    os.environ["OASIS_DB_PATH"] = os.path.abspath(db_path)

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

    # Define a manual action for the agent alice to create a post
    action_hello = {
        env.agent_graph.get_agent(0): [
            ManualAction(action_type=ActionType.CREATE_POST,
                         action_args={"content": "Hello, OASIS World!"})
        ]
    }
    # Run the manual action
    await env.step(action_hello)

    # Define the LLM actions for all agents
    all_agents_llm_actions = {
        agent: LLMAction()
        for _, agent in env.agent_graph.get_agents()
    }
    # Run the LLM actions
    await env.step(all_agents_llm_actions)

    # Close the environment
    await env.close()


if __name__ == "__main__":
    asyncio.run(main())
