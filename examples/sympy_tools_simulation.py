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
from camel.toolkits import SymPyToolkit
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

    agent_graph = AgentGraph()
    agent_1 = SocialAgent(
        agent_id=0,
        user_info=UserInfo(
            user_name="ali",
            name="Alice",
            description="A girl",
            profile=None,
            recsys_type="reddit",
        ),
        agent_graph=agent_graph,
        model=openai_model,
        available_actions=available_actions,
    )
    agent_graph.add_agent(agent_1)

    agent_2 = SocialAgent(agent_id=1,
                          user_info=UserInfo(
                              user_name="bubble",
                              name="Bob",
                              description="A boy",
                              profile=None,
                              recsys_type="reddit",
                          ),
                          tools=SymPyToolkit().get_tools(),
                          agent_graph=agent_graph,
                          model=openai_model,
                          available_actions=available_actions,
                          single_iteration=False)
    agent_graph.add_agent(agent_2)

    # Define the path to the database
    db_path = "./data/reddit_simulation.db"
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

    actions_1 = {
        env.agent_graph.get_agent(0): [
            ManualAction(
                action_type=ActionType.CREATE_POST,
                action_args={
                    "content":
                    "I am doing my homework. Can any kind soul help me "
                    "simplify this expression: "
                    "(x**4 - 16)/(x**2 - 4) + sin(x)**2 + cos(x)**2 + "
                    "(x**3 + 6*x**2 + 12*x + 8)/(x + 2)"
                }),
            ManualAction(action_type=ActionType.CREATE_COMMENT,
                         action_args={
                             "post_id":
                             "1",
                             "content":
                             "I will give a big thumbs up to "
                             "anyone who helps me solve this!"
                         })
        ]
    }
    await env.step(actions_1)

    for _ in range(3):
        action = {
            agent: LLMAction()
            for _, agent in env.agent_graph.get_agents()
        }
        await env.step(action)

    # Close the environment
    await env.close()


if __name__ == "__main__":
    asyncio.run(main())
