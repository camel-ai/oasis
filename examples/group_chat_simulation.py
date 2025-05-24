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
from camel.types import ModelPlatformType

import oasis
from oasis import ActionType, EnvAction, SingleAction


async def main():
    # NOTE: You need to deploy the vllm server first
    vllm_model_1 = ModelFactory.create(
        model_platform=ModelPlatformType.DEEPSEEK,
        model_type="deepseek-chat",
        url="https://api.deepseek.com/v1",
        api_key="ak",
    )
    vllm_model_2 = ModelFactory.create(
        model_platform=ModelPlatformType.DEEPSEEK,
        model_type="deepseek-chat",
        url="https://api.deepseek.com/v1",
        api_key="ak"
    )
    # Define the models for agents. Agents will select models based on
    # pre-defined scheduling strategies
    models = [vllm_model_1, vllm_model_2]

    # Define the available actions for the agents
    available_actions = [
        ActionType.JOIN_GROUP,
        ActionType.CREATE_GROUP,
        ActionType.SEND_TO_GROUP,
        ActionType.LEAVE_GROUP
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
        agent_profile_path=(
            "data/twitter_dataset/anonymous_topic_200_1h/False_Business_0.csv"
        ),
        agent_models=models,
        available_actions=available_actions,
    )

    # Run the environment
    await env.reset()

    group_result = await env.platform.create_group(1, "AI Discussion Group")
    group_id = group_result["group_id"]

    action_1 = SingleAction(
        agent_id=0, action=ActionType.JOIN_GROUP, args={"group_id": group_id}
    )
    env_actions_1 = EnvAction(
        # Activate 5 agents with id 1, 3, 5, 7, 9
        activate_agents=[1, 3, 5, 7, 9],
        intervention=[action_1])

    action_2 = SingleAction(agent_id=1,
                            action=ActionType.SEND_TO_GROUP,
                            args={"group_id":  group_id, "message": "DeepSeek is amazing!"})
    env_actions_2 = EnvAction(activate_agents=[1, 3, 5, 7, 9],
                              intervention=[action_2])

    empty_action = EnvAction()  # Means activate all agents and no intervention

    all_env_actions = [
        env_actions_1,
        env_actions_2,
        empty_action,
    ]

    # Simulate 3 timesteps
    for i in range(2):
        env_actions = all_env_actions[i]
        # Perform the actions
        await env.step(env_actions)

    # Close the environment
    await env.close()


if __name__ == "__main__":
    asyncio.run(main())
