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
"""
Fixed Twitter Interview Example

This example demonstrates the correct way to implement interviews in OASIS:
- INTERVIEW is NOT included in available_actions to prevent LLM auto-selection
- Interviews are conducted manually using ManualAction
- LLM agents can only perform regular social media actions
"""
import asyncio
import os
import sqlite3
import json

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
    # IMPORTANT: INTERVIEW is NOT included here to prevent LLM from automatically selecting it
    # This ensures that interviews are only conducted manually by researchers/developers
    available_actions = [
        ActionType.CREATE_POST,
        ActionType.LIKE_POST,
        ActionType.REPOST,
        ActionType.FOLLOW,
        ActionType.DO_NOTHING,
        ActionType.QUOTE_POST,
        # ActionType.INTERVIEW,  # Deliberately excluded - interviews are manual only
    ]

    agent_graph = await generate_twitter_agent_graph(
        profile_path=("data/twitter_dataset/anonymous_topic_200_1h/"
                      "False_Business_0.csv"),
        model=openai_model,
        available_actions=available_actions,
    )

    # Define the path to the database
    db_path = "./data/twitter_simulation_fixed.db"

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
    print("=== Timestep 1: Initial post ===")
    actions_1 = {}
    actions_1[env.agent_graph.get_agent(0)] = ManualAction(
        action_type=ActionType.CREATE_POST,
        action_args={"content": "Earth is flat."})
    await env.step(actions_1)

    # Second timestep: Let some agents respond with LLM actions
    # Note: Since INTERVIEW is not in available_actions, LLM agents cannot choose it
    print("=== Timestep 2: LLM agents respond (no interview option) ===")
    actions_2 = {
        agent: LLMAction()
        # Activate 5 agents with id 1, 3, 5, 7, 9
        for _, agent in env.agent_graph.get_agents([1, 3, 5, 7, 9])
    }
    await env.step(actions_2)

    # Third timestep: Agent 1 creates a post, and we manually interview Agent 0
    print("=== Timestep 3: Manual interview ===")
    actions_3 = {}
    actions_3[env.agent_graph.get_agent(1)] = ManualAction(
        action_type=ActionType.CREATE_POST,
        action_args={"content": "Earth is not flat."})
    
    # Manual interview - this is controlled by the researcher, not the LLM
    actions_3[env.agent_graph.get_agent(0)] = ManualAction(
        action_type=ActionType.INTERVIEW,
        action_args={"prompt": "What do you think about the shape of the Earth? Please explain your reasoning."})
    
    await env.step(actions_3)

    # Fourth timestep: Let some other agents respond with LLM actions
    print("=== Timestep 4: More LLM responses ===")
    actions_4 = {
        agent: LLMAction()
        for _, agent in env.agent_graph.get_agents([2, 4, 6, 8, 10])
    }
    await env.step(actions_4)

    # Fifth timestep: Interview multiple agents manually
    print("=== Timestep 5: Multiple manual interviews ===")
    actions_5 = {}
    actions_5[env.agent_graph.get_agent(1)] = ManualAction(
        action_type=ActionType.INTERVIEW,
        action_args={"prompt": "Why do you believe the Earth is not flat?"})
    
    actions_5[env.agent_graph.get_agent(2)] = ManualAction(
        action_type=ActionType.INTERVIEW,
        action_args={"prompt": "What are your thoughts on the debate about Earth's shape?"})
    
    await env.step(actions_5)

    # Sixth timestep: Final LLM actions for remaining agents
    print("=== Timestep 6: Final LLM responses ===")
    actions_6 = {
        agent: LLMAction()
        for _, agent in env.agent_graph.get_agents([3, 5, 7, 9])
    }
    await env.step(actions_6)

    # Close the environment
    await env.close()
    
    # Visualize the interview results
    print("\n=== Interview Results ===")
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Query all interview records from the database
    cursor.execute("""
        SELECT user_id, info, created_at
        FROM trace
        WHERE action = ?
        ORDER BY created_at
    """, (ActionType.INTERVIEW.value,))
    
    interview_count = 0
    for user_id, info_json, timestamp in cursor.fetchall():
        info = json.loads(info_json)
        interview_count += 1
        print(f"\nInterview #{interview_count} - Agent {user_id} (Timestep {timestamp}):")
        print(f"Question: {info.get('prompt', 'N/A')}")
        print(f"Response: {info.get('response', 'N/A')}")
        print(f"Interview ID: {info.get('interview_id', 'N/A')}")
    
    if interview_count == 0:
        print("No interviews found in the database.")
    else:
        print(f"\nTotal interviews conducted: {interview_count}")
    
    # Also check what actions LLM agents actually performed
    print("\n=== LLM Agent Actions (should not include interviews) ===")
    cursor.execute("""
        SELECT DISTINCT action, COUNT(*) as count
        FROM trace
        WHERE action != ?
        GROUP BY action
        ORDER BY count DESC
    """, (ActionType.INTERVIEW.value,))
    
    for action, count in cursor.fetchall():
        print(f"{action}: {count} times")
    
    conn.close()
    
    print("\n=== Summary ===")
    print("✅ INTERVIEW was not included in available_actions")
    print("✅ LLM agents could not automatically select interview actions")
    print("✅ Interviews were conducted manually using ManualAction")
    print("✅ This prevents unwanted agent-to-agent interviews")


if __name__ == "__main__":
    asyncio.run(main()) 