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
import json
import os
import sqlite3

from camel.models import ModelFactory
from camel.types import ModelPlatformType, ModelType

import oasis
from oasis import (ActionType, LLMAction, ManualAction,
                   generate_twitter_agent_graph)


async def main():
    openai_model = ModelFactory.create(
        model_platform=ModelPlatformType.OPENAI,
        model_type=ModelType.GPT_4O_MINI,
    )

    # Define the available actions for the agents
    # Note: INTERVIEW is NOT included here to
    # prevent LLM from automatically selecting it
    # INTERVIEW can still be used manually via ManualAction
    available_actions = [
        ActionType.CREATE_POST,
        ActionType.LIKE_POST,
        ActionType.REPOST,
        ActionType.FOLLOW,
        ActionType.DO_NOTHING,
        ActionType.QUOTE_POST,
    ]

    agent_graph = await generate_twitter_agent_graph(
        profile_path=("data/twitter_dataset/anonymous_topic_200_1h/"
                      "False_Business_0.csv"),
        model=openai_model,
        available_actions=available_actions,
    )

    # Define the path to the database
    db_path = "./data/twitter_simulation.db"
    os.environ["OASIS_DB_PATH"] = os.path.abspath(db_path)

    # Delete the old database
    if os.path.exists(db_path):
        os.remove(db_path)

    # Make the environment
    env = oasis.make(
        agent_graph=agent_graph,
        platform=oasis.DefaultPlatformType.TWITTER,
        database_path=db_path,
    )

    # Run the environment
    await env.reset()

    # First timestep: Agent 0 creates a post
    actions_1 = {}
    actions_1[env.agent_graph.get_agent(0)] = ManualAction(
        action_type=ActionType.CREATE_POST,
        action_args={"content": "Earth is flat."})
    await env.step(actions_1)

    # Second timestep: Let some agents respond with LLM actions
    actions_2 = {
        agent: LLMAction()
        # Activate 5 agents with id 1, 3, 5, 7, 9
        for _, agent in env.agent_graph.get_agents([1, 3, 5, 7, 9])
    }
    await env.step(actions_2)

    # Third timestep: Agent 1 creates a post, and we interview Agent 0
    actions_3 = {}
    actions_3[env.agent_graph.get_agent(1)] = ManualAction(
        action_type=ActionType.CREATE_POST,
        action_args={"content": "Earth is not flat."})

    # Create an interview action to ask Agent 0 about their views
    # ActionType.INTERVIEW is a external action,
    # which can not be exceuted by agents themselves
    actions_3[env.agent_graph.get_agent(0)] = ManualAction(
        action_type=ActionType.INTERVIEW,
        action_args={
            "prompt": "What do you think about the shape of the Earth?"
        })

    await env.step(actions_3)

    # Fourth timestep: Let some other agents respond
    # Activate 5 agents with id 2, 4, 6, 8, 10
    actions_4 = {
        agent: LLMAction()
        for _, agent in env.agent_graph.get_agents([2, 4, 6, 8, 10])
    }
    await env.step(actions_4)

    # Fifth timestep: Interview multiple agents
    actions_5 = {}
    actions_5[env.agent_graph.get_agent(1)] = ManualAction(
        action_type=ActionType.INTERVIEW,
        action_args={"prompt": "Why do you believe the Earth is not flat?"})

    actions_5[env.agent_graph.get_agent(2)] = ManualAction(
        action_type=ActionType.INTERVIEW,
        action_args={
            "prompt":
            "What are your thoughts on the debate about Earth's shape?"
        })

    await env.step(actions_5)

    # Sixth timestep: Final LLM actions for remaining agents
    actions_6 = {
        agent: LLMAction()
        for _, agent in env.agent_graph.get_agents([3, 5, 7, 9])
    }
    await env.step(actions_6)

    # Close the environment
    await env.close()

    # visualize the interview results
    print("\n=== Interview Results ===")
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    # Here we query all interview records from the database
    # We use ActionType.INTERVIEW.value as the query condition
    # to get all interview records
    # Each record contains user ID, interview information
    # (in JSON format), and creation timestamp
    cursor.execute(
        """
        SELECT user_id, info, created_at
        FROM trace
        WHERE action = ?
    """, (ActionType.INTERVIEW.value, ))

    # This query retrieves all interview records from the trace table
    # - user_id: the ID of the agent who was interviewed
    # - info: JSON string containing interview details (prompt, response, etc.)
    # - created_at: timestamp when the interview was conducted
    # We'll parse this data below to display the interview results
    for user_id, info_json, timestamp in cursor.fetchall():
        info = json.loads(info_json)
        print(f"\nAgent {user_id} (Timestep {timestamp}):")
        print(f"Prompt: {info.get('prompt', 'N/A')}")
        print(f"Interview ID: {info.get('interview_id', 'N/A')}")
        print(f"Response: {info.get('response', 'N/A')}")

    conn.close()


if __name__ == "__main__":
    asyncio.run(main())
