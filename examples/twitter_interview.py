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
import asyncio
import os
import sqlite3
import json

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
        ActionType.INTERVIEW,  # Add the INTERVIEW action type
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
        agent_profile_path=("data/twitter_dataset/anonymous_topic_200_1h/"
                            "False_Business_0.csv"),
        agent_models=openai_model,
        available_actions=available_actions,
    )

    # Run the environment
    await env.reset()

    # First timestep: Agent 0 creates a post
    action_1 = SingleAction(agent_id=0,
                            action=ActionType.CREATE_POST,
                            args={"content": "Earth is flat."})
    env_actions_1 = EnvAction(
        # Activate 5 agents with id 1, 3, 5, 7, 9
        activate_agents=[1, 3, 5, 7, 9],
        intervention=[action_1])

    # Second timestep: Agent 1 creates a post, and we interview Agent 0
    action_2 = SingleAction(agent_id=1,
                           action=ActionType.CREATE_POST,
                           args={"content": "Earth is not flat."})
    
    # Create an interview action to ask Agent 0 about their views
    interview_action = SingleAction(
        agent_id=0,
        action=ActionType.INTERVIEW,
        args={"prompt": "What do you think about the shape of the Earth? Please explain your reasoning."}
    )
    
    env_actions_2 = EnvAction(
        activate_agents=[2, 4, 6, 8, 10],
        intervention=[action_2, interview_action]
    )

    # Third timestep: Interview multiple agents
    interview_actions = [
        SingleAction(
            agent_id=1,
            action=ActionType.INTERVIEW,
            args={"prompt": "Why do you believe the Earth is not flat?"}
        ),
        SingleAction(
            agent_id=2,
            action=ActionType.INTERVIEW,
            args={"prompt": "What are your thoughts on the debate about Earth's shape?"}
        ),
    ]
    
    env_actions_3 = EnvAction(
        activate_agents=[3, 5, 7, 9],
        intervention=interview_actions
    )

    all_env_actions = [
        env_actions_1,
        env_actions_2,
        env_actions_3,
    ]

    # Simulate 3 timesteps
    for i in range(3):
        print(f"\n=== Timestep {i+1} ===")
        env_actions = all_env_actions[i]
        # Perform the actions
        await env.step(env_actions)

    # Close the environment
    await env.close()
    
    # visualize the interview results
    print("\n=== Interview Results ===")
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("""
        SELECT user_id, info, created_at
        FROM trace
        WHERE action = ?
    """, (ActionType.INTERVIEW.value,))
    
    for user_id, info_json, timestamp in cursor.fetchall():
        info = json.loads(info_json)
        print(f"\nAgent {user_id} (Timestep {timestamp}):")
        print(f"Prompt: {info.get('prompt', 'N/A')}")
        print(f"Interview ID: {info.get('interview_id', 'N/A')}")
        print(f"Response: {info.get('response', 'N/A')}")
    
    conn.close()


if __name__ == "__main__":
    asyncio.run(main())
