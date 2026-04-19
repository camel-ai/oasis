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
"""TikTok simulation example.

Demonstrates how to create a TikTok simulation with:
- Short video uploads and feed consumption
- Traffic pool recommendation algorithm
- Configurable algorithm parameters

Usage:
    export OPENAI_API_KEY="your-api-key"
    python examples/tiktok_simulation.py
"""

import asyncio
import os
import tempfile

from camel.models import ModelFactory
from camel.types import ModelPlatformType, ModelType

import oasis
from oasis import (ActionType, AgentGraph, LLMAction, ManualAction,
                   SocialAgent, UserInfo)


async def main():
    # Create the LLM model for agent decision-making
    model = ModelFactory.create(
        model_platform=ModelPlatformType.OPENAI,
        model_type=ModelType.GPT_4O_MINI,
    )

    # Define TikTok-specific available actions
    available_actions = ActionType.get_default_tiktok_actions()

    # Initialize the agent graph with TikTok users
    agent_graph = AgentGraph()

    # Creator agent: produces short videos
    creator = SocialAgent(
        agent_id=0,
        user_info=UserInfo(
            user_name="dance_queen",
            name="Lily",
            description=(
                "A popular dance creator on TikTok. "
                "[interests: dance, music, trending]"
            ),
            profile={
                "other_info": {
                    "user_profile": "Professional dancer sharing daily "
                                    "routines and challenges",
                    "role": "creator",
                }
            },
            recsys_type="tiktok",
        ),
        agent_graph=agent_graph,
        model=model,
        available_actions=available_actions,
    )
    agent_graph.add_agent(creator)

    # Viewer agent: consumes content
    viewer = SocialAgent(
        agent_id=1,
        user_info=UserInfo(
            user_name="music_fan",
            name="Tom",
            description=(
                "A college student who loves watching dance and music videos. "
                "[interests: dance, comedy, food]"
            ),
            profile={
                "other_info": {
                    "user_profile": "Casual TikTok user, scrolls during "
                                    "commute and before bed",
                    "role": "viewer",
                }
            },
            recsys_type="tiktok",
        ),
        agent_graph=agent_graph,
        model=model,
        available_actions=available_actions,
    )
    agent_graph.add_agent(viewer)

    # Set up the database
    db_path = os.path.join(tempfile.gettempdir(), "tiktok_example.db")
    if os.path.exists(db_path):
        os.remove(db_path)

    # Create the TikTok environment with custom algorithm parameters
    env = oasis.make(
        agent_graph=agent_graph,
        platform=oasis.DefaultPlatformType.TIKTOK,
        database_path=db_path,
        semaphore=10,
        # Configurable traffic pool algorithm parameters
        tiktok_recsys_params={
            "score_weights": {
                "completion_rate": 0.35,
                "like_rate": 0.15,
                "comment_rate": 0.15,
                "share_rate": 0.20,
                "negative_rate": 0.15,
            },
            "promote_percentile": 0.20,
            "demote_percentile": 0.70,
            "max_pool_level": 7,
            "rec_mix": {
                "interest": 0.70,
                "following": 0.15,
                "explore": 0.10,
            },
            "decay_half_life": 72,
        },
    )

    # Reset the environment (signs up all agents)
    await env.reset()
    print("TikTok environment initialized with 2 agents\n")

    # Step 1: Creator uploads a video
    print("Step 1: Creator uploads a dance video...")
    await env.step({
        creator: ManualAction(
            action_type=ActionType.UPLOAD_VIDEO,
            action_args={
                "content": "Amazing dance challenge! Try this with your "
                           "friends #dance #challenge #trending",
                "duration_seconds": 15,
                "category": "dance",
                "topic_tags": '["dance", "challenge", "trending"]',
            },
        )
    })
    print("  Video uploaded!\n")

    # Step 2: Viewer consumes content via LLM decision
    print("Step 2: Viewer browses TikTok feed (LLM decides action)...")
    await env.step({
        viewer: LLMAction()
    })
    print("  Viewer performed an action based on LLM reasoning\n")

    # Step 3: Manual watch action with completion tracking
    print("Step 3: Viewer watches the video (85% completion)...")
    await env.step({
        viewer: ManualAction(
            action_type=ActionType.WATCH_VIDEO,
            action_args={
                "post_id": 1,
                "watch_ratio": 0.85,
            },
        )
    })
    print("  Watch recorded with 85% completion rate\n")

    # Step 4: Both agents act via LLM
    print("Step 4: Both agents perform LLM-driven actions...")
    await env.step({
        creator: LLMAction(),
        viewer: LLMAction(),
    })
    print("  Both agents acted\n")

    # Print simulation results
    print("=" * 50)
    print("Simulation Results")
    print("=" * 50)
    print(f"Database: {db_path}")
    print("\nCheck the trace table for all recorded actions:")
    print(f"  sqlite3 {db_path} \"SELECT action, info FROM trace\"")
    print("\nCheck the video table for traffic pool data:")
    print(f"  sqlite3 {db_path} \"SELECT * FROM video\"")

    # Close the environment
    await env.close()
    print("\nEnvironment closed. Simulation complete!")

    # Clean up
    if os.path.exists(db_path):
        os.remove(db_path)


if __name__ == "__main__":
    asyncio.run(main())
