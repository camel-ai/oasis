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
#!/usr/bin/env python3
"""
Twitter Misinformation Simulation with Configurable Post Filtering

This example demonstrates how to run a Twitter misinformation simulation
with different post filtering configurations to prevent context overflow.
"""

import asyncio
import os

from camel.models import ModelFactory
from camel.types import ModelPlatformType

import oasis
from oasis import (ActionType, LLMAction, ManualAction,
                   generate_twitter_agent_graph)

# Post filtering configurations for different scenarios
FILTERING_CONFIGS = {
    "conservative": {
        "enable_post_filtering":
        True,
        "max_posts_in_memory":
        8,
        "post_filter_strategy":
        "recency",
        "enable_comment_filtering":
        True,
        "max_comments_per_post":
        3,
        "comment_filter_strategy":
        "recency",
        "description":
        "Conservative filtering - 8 recent posts, 3 recent comments per post"
    },
    "balanced": {
        "enable_post_filtering":
        True,
        "max_posts_in_memory":
        12,
        "post_filter_strategy":
        "mixed",
        "enable_comment_filtering":
        True,
        "max_comments_per_post":
        5,
        "comment_filter_strategy":
        "mixed",
        "description":
        "Balanced filtering - 12 posts and 5 comments using mixed strategy"
    },
    "engagement_focused": {
        "enable_post_filtering":
        True,
        "max_posts_in_memory":
        15,
        "post_filter_strategy":
        "popularity",
        "enable_comment_filtering":
        True,
        "max_comments_per_post":
        8,
        "comment_filter_strategy":
        "popularity",
        "description":
        "Engagement focused - 15 popular posts, 8 popular comments per post"
    },
    "no_filtering": {
        "enable_post_filtering":
        False,
        "max_posts_in_memory":
        10,  # Ignored when filtering is disabled
        "post_filter_strategy":
        "recency",  # Ignored when filtering is disabled
        "enable_comment_filtering":
        False,
        "max_comments_per_post":
        5,  # Ignored when filtering is disabled
        "comment_filter_strategy":
        "recency",  # Ignored when filtering is disabled
        "description":
        "No filtering - agents see all posts and comments (may cause context overflow)"
    }
}


async def run_simulation_with_config(config_name: str):
    """Run the simulation with a specific filtering configuration"""

    config = FILTERING_CONFIGS[config_name]
    print(f"\n=== Running Simulation: {config_name.upper()} ===")
    print(f"Description: {config['description']}")
    print(f"Post filtering enabled: {config['enable_post_filtering']}")
    if config['enable_post_filtering']:
        print(f"Max posts: {config['max_posts_in_memory']}")
        print(f"Post strategy: {config['post_filter_strategy']}")
    print(f"Comment filtering enabled: {config['enable_comment_filtering']}")
    if config['enable_comment_filtering']:
        print(f"Max comments per post: {config['max_comments_per_post']}")
        print(f"Comment strategy: {config['comment_filter_strategy']}")
    print("=" * 50)

    # Create the model
    vllm_model = ModelFactory.create(
        model_platform=ModelPlatformType.VLLM,
        model_type="qwen-2",
        url="http://10.8.131.51:30973/v1",
    )

    # Define available actions
    available_actions = [
        ActionType.CREATE_POST,
        ActionType.LIKE_POST,
        ActionType.REPOST,
        ActionType.QUOTE_POST,
        ActionType.FOLLOW,
        ActionType.DO_NOTHING,
        ActionType.CREATE_COMMENT,
    ]

    # Generate agent graph with the specified filtering configuration
    agent_graph = await generate_twitter_agent_graph(
        profile_path=("data/twitter_dataset/anonymous_topic_200_1h/"
                      "False_Business_0.csv"),
        model=vllm_model,
        available_actions=available_actions,
        **config  # Unpack the filtering configuration
    )

    # Database setup
    db_path = f"./data/twitter_simulation_{config_name}.db"
    os.environ["OASIS_DB_PATH"] = os.path.abspath(db_path)

    if os.path.exists(db_path):
        os.remove(db_path)

    # Create environment
    env = oasis.make(
        agent_graph=agent_graph,
        platform=oasis.DefaultPlatformType.TWITTER,
        database_path=db_path,
    )

    await env.reset()

    # Create initial posts
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

    await env.step(actions_1)

    # Run simulation
    num_timesteps = 10  # Reduced for demonstration
    for timestep in range(num_timesteps):
        print(f"  Timestep {timestep + 1}/{num_timesteps}")
        actions = {
            agent: LLMAction()
            for _, agent in env.agent_graph.get_agents()
        }
        await env.step(actions)

    await env.close()
    print(f"✓ Simulation completed: {config_name}")
    return db_path


async def compare_filtering_strategies():
    """Compare different filtering strategies"""

    print("Twitter Misinformation Simulation with Post Filtering")
    print(
        "This example shows how different filtering configurations affect agent behavior"
    )
    print("\nAvailable configurations:")

    for name, config in FILTERING_CONFIGS.items():
        print(f"  {name}: {config['description']}")

    # You can choose which configurations to run
    configs_to_run = ["balanced", "conservative"]  # Add more as needed

    results = {}
    for config_name in configs_to_run:
        try:
            db_path = await run_simulation_with_config(config_name)
            results[config_name] = {"status": "success", "db_path": db_path}
        except Exception as e:
            results[config_name] = {"status": "error", "error": str(e)}
            print(f"❌ Error in {config_name}: {e}")

    # Summary
    print("\n" + "=" * 60)
    print("SIMULATION RESULTS SUMMARY")
    print("=" * 60)

    for config_name, result in results.items():
        status_icon = "✓" if result["status"] == "success" else "❌"
        print(f"{status_icon} {config_name.upper()}: {result['status']}")
        if result["status"] == "success":
            print(f"    Database: {result['db_path']}")

    print("\nPost & Comment filtering benefits:")
    print("1. Prevents context overflow in long-running simulations")
    print("2. Improves agent response time by reducing input tokens")
    print("3. Ensures agents focus on most relevant content")
    print("4. Filters both posts and comments to manage memory usage")
    print("5. Configurable strategies for different experimental needs")
    print("6. Supports recency, popularity, and mixed filtering strategies")


async def main():
    """Main function to run the demonstration"""
    await compare_filtering_strategies()


if __name__ == "__main__":
    asyncio.run(main())
