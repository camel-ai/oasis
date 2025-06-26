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
from oasis import (ActionType, LLMAction, ManualAction,
                   generate_twitter_agent_graph)


async def main():

    vllm_model = ModelFactory.create(
        model_platform=ModelPlatformType.VLLM,
        model_type="qwen-2",
        url="http://10.8.131.51:30973/v1",
    )

    # Define the available actions for the agents
    # Note: INTERVIEW is NOT included here to
    # prevent LLM from automatically selecting it
    # INTERVIEW can still be used manually via ManualAction
    available_actions = [
        ActionType.CREATE_POST,
        ActionType.LIKE_POST,
        ActionType.REPOST,
        ActionType.QUOTE_POST,
        ActionType.FOLLOW,
        ActionType.DO_NOTHING,
        ActionType.CREATE_COMMENT,
    ]

    # Generate agent graph with post filtering enabled
    # This prevents agents from seeing too many posts and exceeding context limits
    agent_graph = await generate_twitter_agent_graph(
        profile_path=("data/twitter_dataset/anonymous_topic_200_1h/"
                      "False_Business_0.csv"),
        model=vllm_model,
        available_actions=available_actions,
        # Post filtering configuration
        enable_post_filtering=True,  # Enable post filtering
        max_posts_in_memory=3,  # Limit each agent to see max 12 posts
        post_filter_strategy=
        "mixed",  # Use mixed strategy (recency + popularity)
        # Comment filtering configuration
        enable_comment_filtering=True,  # Enable comment filtering
        max_comments_per_post=3,  # Limit to 3 comments per post
        comment_filter_strategy="mixed",  # Use mixed strategy for comments too
    )

    # Print filtering configuration info
    print("=== Post & Comment Filtering Configuration ===")
    print("✓ Post filtering enabled")
    print("✓ Max posts per agent: 3")
    print("✓ Post filtering strategy: mixed (recency + popularity)")
    print("✓ Comment filtering enabled")
    print("✓ Max comments per post: 3")
    print("✓ Comment filtering strategy: mixed (recency + popularity)")
    print("✓ This prevents context overflow during long simulations")
    print("===============================================\n")

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

    # Initial posts to seed the discussion
    # These will be filtered along with other posts during agent interactions
    actions_1 = {
        env.agent_graph.get_agent(0): [
            ManualAction(
                action_type=ActionType.CREATE_POST,
                action_args={
                    "content":
                    "Amazon is expanding its delivery drone program to deliver "
                    "packages within 30 minutes in select cities. This initiative "
                    "aims to improve efficiency and reduce delivery times."
                }),
            ManualAction(
                action_type=ActionType.CREATE_POST,
                action_args={
                    "content":
                    "Amazon plans to completely eliminate its delivery drivers "
                    "within two years due to the new drone program. "
                    "#Automation #Future"
                })
        ]
    }

    print("Creating initial posts...")
    await env.step(actions_1)

    # Run for 20 timesteps
    # With post filtering, agents will only see the most relevant posts
    # based on the configured strategy (mixed: recency + popularity)
    print("Running simulation for 20 timesteps...")
    print("Note: Agents will see filtered posts to prevent context overflow\n")

    for timestep in range(20):
        print(f"Timestep {timestep + 1}/20")
        actions = {
            agent: LLMAction()
            for _, agent in env.agent_graph.get_agents()
        }
        await env.step(actions)

    print("\n=== Simulation Complete ===")
    print("Post filtering helped prevent context overflow")
    print("Agents only saw the most relevant posts during the simulation")

    # Close the environment
    await env.close()


if __name__ == "__main__":
    asyncio.run(main())
