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
import random

from camel.models import ModelFactory
from camel.types import ModelPlatformType, ModelType

import oasis
from oasis import ActionType, EnvAction, SingleAction


async def main():
    openai_model = ModelFactory.create(
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
        ActionType.QUOTE_POST,
    ]

    # Define the path to the database
    db_path = "./data/twitter_simulation.db"

    # Delete the old database
    if os.path.exists(db_path):
        os.remove(db_path)

    # Make the environment
    env = oasis.make(
        platform=oasis.DefaultPlatformType.TWITTER,
        database_path=db_path,
        agent_profile_path=("/Users/zzb/Documents/code/oasis_v2/oasis/tmp/random_network.csv"),
        agent_models=openai_model,
        available_actions=available_actions,
    )

    # Run the environment
    await env.reset()

    # inject misinformation
    action_1 = SingleAction(agent_id=0,
                            action=ActionType.CREATE_POST,
                            args={"content": "Amazon is expanding its delivery drone program to deliver packages within 30 minutes in select cities. This initiative aims to improve efficiency and reduce delivery times."})
    action_2 = SingleAction(agent_id=0,
                            action=ActionType.CREATE_POST,
                            args={"content": "'Amazon plans to completely eliminate its delivery drivers within two years due to the new drone program. #Automation #Future"})
    init_simulation = EnvAction(activate_agents=[0], intervention=[action_1, action_2])

    env_simulation_list = [init_simulation]
    for timestep in range(3):
        # Randomly select 10% of agents to activate
        total_agents = env.get_num_agents()
        num_agents_to_activate = max(1, int(total_agents * 0.01))  # Ensure at least 1 agent is activated
        agents_to_activate = random.sample(range(total_agents), num_agents_to_activate)
        
        # Create an environment action with the randomly selected agents
        random_action = EnvAction(activate_agents=agents_to_activate)
        env_simulation_list.append(random_action)

    # Simulate 3 timesteps
    for i in range(3):
        env_actions = env_simulation_list[i]
        # Perform the actions
        await env.step(env_actions)

    # Close the environment
    await env.close()


if __name__ == "__main__":
    asyncio.run(main())
