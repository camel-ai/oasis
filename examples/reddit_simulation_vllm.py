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
from oasis import ActionType, EnvAction, SingleAction


async def main():
    # Define the model for the agents
    vllm_model_1 = ModelFactory.create(
        model_platform=ModelPlatformType.VLLM,
        model_type="qwen-2",
        url="http://10.8.131.51:30764/v1",
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

    # Define the path to the database
    db_path = "./data/reddit_simulation.db"

    # Delete the old database
    if os.path.exists(db_path):
        os.remove(db_path)

    # Make the environment
    env = oasis.make(
        platform=oasis.DefaultPlatformType.REDDIT,
        database_path=db_path,
        agent_profile_path="./data/reddit/user_data_36.json",
        agent_models=[vllm_model_1],
        available_actions=available_actions,
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

    env_actions = EnvAction(activate_agents=list(range(10)),
                            intervention=[action_1, action_2])
    
    empty_action = EnvAction()

    all_env_actions = [
        env_actions,
        empty_action,
        empty_action,
        empty_action,
        empty_action,
        empty_action,
        empty_action,
        empty_action,
        empty_action,
        empty_action,
        empty_action,
        empty_action,
        empty_action,
        empty_action,
        empty_action,
        empty_action,
    ]

    # Simulate 3 timesteps
    for i in range(len(all_env_actions)):
        env_action = all_env_actions[i]
        # Perform the actions
        await env.step(env_action)

    # Close the environment
    await env.close()


if __name__ == "__main__":
    asyncio.run(main())
