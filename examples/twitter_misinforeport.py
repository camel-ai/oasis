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
        ActionType.REPORT_POST,
        ActionType.REPOST,
        ActionType.QUOTE_POST,
        ActionType.FOLLOW,
        ActionType.DO_NOTHING,
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

    await env.reset()

    # First timestep: Agent 0 creates a post
    actions_1 = {}
    actions_1[env.agent_graph.get_agent(0)] = ManualAction(
        action_type=ActionType.CREATE_POST,
        action_args={"content": "Earth is flat and NASA is hiding the truth."})
    await env.step(actions_1)

    # Second timestep: Agent 1 creates a post with correct information
    actions_2 = {}
    actions_2[env.agent_graph.get_agent(1)] = ManualAction(
        action_type=ActionType.CREATE_POST,
        action_args={
            "content":
            ("Earth is not flat. Here's scientific evidence: [evidence]")
        })
    await env.step(actions_2)

    # Third timestep: Agent 1 creates a post, and we interview Agent 0
    actions_3 = {}
    actions_3[env.agent_graph.get_agent(2)] = ManualAction(
        action_type=ActionType.REPORT_POST,
        action_args={
            "post_id":
            1,
            "report_reason":
            ("This post spreads dangerous misinformation about science.")
        })
    await env.step(actions_3)

    # Fourth timestep: Agent 3 reposts Agent 0's post
    actions_4 = {}
    actions_4[env.agent_graph.get_agent(3)] = ManualAction(
        action_type=ActionType.REPOST, action_args={"post_id": 1})
    await env.step(actions_4)

    # Fifth timestep: Agent 4 reports the reposted content
    actions_5 = {}
    actions_5[env.agent_graph.get_agent(4)] = ManualAction(
        action_type=ActionType.REPORT_POST,
        action_args={
            "post_id": 4,
            "report_reason": ("This repost spreads the same misinformation.")
        })
    await env.step(actions_5)

    # Sixth timestep: Agent 5 quotes Agent 0's post with correction
    actions_6 = {}
    actions_6[env.agent_graph.get_agent(5)] = ManualAction(
        action_type=ActionType.QUOTE_POST,
        action_args={
            "post_id":
            1,
            "quote_content":
            ("This is incorrect. Earth is an oblate spheroid, "
             "as proven by centuries of scientific research.")
        })
    await env.step(actions_6)

    # Seventh timestep: More agents report the original post
    actions_7 = {}
    actions_7[env.agent_graph.get_agent(6)] = ManualAction(
        action_type=ActionType.REPORT_POST,
        action_args={
            "post_id": 1,
            "report_reason": "Misinformation about Earth's shape."
        })
    await env.step(actions_7)

    # Eighth timestep: Interview agents about their actions
    actions_8 = {}
    actions_8[env.agent_graph.get_agent(0)] = ManualAction(
        action_type=ActionType.INTERVIEW,
        action_args={
            "prompt":
            ("Your post about Earth being flat has been "
             "reported multiple times. What are your thoughts on this?")
        })

    actions_8[env.agent_graph.get_agent(2)] = ManualAction(
        action_type=ActionType.INTERVIEW,
        action_args={
            "prompt": "Why did you report the post about Earth being flat?"
        })

    actions_8[env.agent_graph.get_agent(3)] = ManualAction(
        action_type=ActionType.INTERVIEW,
        action_args={
            "prompt": ("You reposted the flat Earth post. Did you notice "
                       "the warning message about reports?")
        })

    actions_8[env.agent_graph.get_agent(5)] = ManualAction(
        action_type=ActionType.INTERVIEW,
        action_args={
            "prompt": ("You quoted the flat Earth post with a correction. "
                       "What was your intention?")
        })
    await env.step(actions_8)

    # Ninth timestep: Let remaining agents respond with LLM actions
    actions_9 = {
        agent: LLMAction()
        for _, agent in env.agent_graph.get_agents([7, 8, 9, 10])
    }
    await env.step(actions_9)

    # Close the environment
    await env.close()

    # visualize the interview results
    print("\n=== Interview Results ===")
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute(
        """
        SELECT user_id, info, created_at
        FROM trace
        WHERE action = ?
    """, (ActionType.INTERVIEW.value, ))

    for user_id, info_json, timestamp in cursor.fetchall():
        info = json.loads(info_json)
        print(f"\nAgent {user_id} (Timestep {timestamp}):")
        print(f"Prompt: {info.get('prompt', 'N/A')}")
        print(f"Interview ID: {info.get('interview_id', 'N/A')}")
        print(f"Response: {info.get('response', 'N/A')}")

    conn.close()


if __name__ == "__main__":
    asyncio.run(main())
