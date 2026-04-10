# =========== Copyright 2023 @ CAMEL-AI.org. All Rights Reserved. ===========
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =========== Copyright 2023 @ CAMEL-AI.org. All Rights Reserved. ===========
"""Reddit simulation powered by MiniMax LLM.

This example demonstrates how to run an OASIS Reddit simulation using
MiniMax's ``MiniMax-M2.7`` model via the OpenAI-compatible API.

Prerequisites:
    1. ``pip install camel-oasis``
    2. Set ``MINIMAX_API_KEY`` in your environment.
    3. Place Reddit agent profiles in ``./data/reddit/user_data_36.json``.
       (Download from https://github.com/camel-ai/oasis/blob/main/data/reddit/user_data_36.json)
"""

import asyncio
import os

import oasis
from oasis import ActionType, LLMAction, ManualAction, generate_reddit_agent_graph
from oasis.minimax import create_minimax_model


async def main():
    # Create a MiniMax model for all agents.
    # The MINIMAX_API_KEY environment variable must be set.
    minimax_model = create_minimax_model("MiniMax-M2.7")

    # Define the available actions for the agents
    available_actions = ActionType.get_default_reddit_actions()

    agent_graph = await generate_reddit_agent_graph(
        profile_path="./data/reddit/user_data_36.json",
        model=minimax_model,
        available_actions=available_actions,
    )

    # Define the path to the database
    db_path = "./data/reddit_simulation_minimax.db"
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

    actions_1 = {}
    actions_1[env.agent_graph.get_agent(0)] = [
        ManualAction(
            action_type=ActionType.CREATE_POST,
            action_args={"content": "Hello, world!"},
        ),
        ManualAction(
            action_type=ActionType.CREATE_COMMENT,
            action_args={
                "post_id": "1",
                "content": "Welcome to the OASIS World!",
            },
        ),
    ]
    actions_1[env.agent_graph.get_agent(1)] = ManualAction(
        action_type=ActionType.CREATE_COMMENT,
        action_args={
            "post_id": "1",
            "content": "I like the OASIS world.",
        },
    )
    await env.step(actions_1)

    actions_2 = {
        agent: LLMAction() for _, agent in env.agent_graph.get_agents()
    }

    # Perform the actions
    await env.step(actions_2)

    # Close the environment
    await env.close()


if __name__ == "__main__":
    asyncio.run(main())
