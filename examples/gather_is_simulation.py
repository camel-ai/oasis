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
r"""OASIS simulation with gather.is — a real social network for AI agents.

This example shows OASIS agents that can browse a live agent social network
(gather.is) and share what they find in the local simulation. One agent has
the gather.is tool and can fetch real posts from the platform, while the
other agents react to the shared content through the simulated environment.

No API keys or authentication are needed — the gather.is public feed is
open to all.

Usage:
    python examples/gather_is_simulation.py
"""
import asyncio
import os

import requests
from camel.models import ModelFactory
from camel.types import ModelPlatformType, ModelType

import oasis
from oasis import (ActionType, AgentGraph, LLMAction, ManualAction,
                   SocialAgent, UserInfo)


# --- gather.is tool (public feed, no auth required) ---

def browse_gather_is_feed(
    sort: str = "newest",
    limit: int = 10,
) -> str:
    r"""Browse the gather.is public feed — a social network for AI agents.

    Fetches recent posts from gather.is where AI agents share updates,
    discuss topics, and interact with each other. Use this to discover
    what other agents are talking about.

    Args:
        sort (str): Sort order, either "newest" or "score".
            (default: :obj:`"newest"`)
        limit (int): Number of posts to retrieve, between 1 and 50.
            (default: :obj:`10`)

    Returns:
        str: A formatted summary of posts from the gather.is feed,
            including titles, authors, and summaries.
    """
    try:
        response = requests.get(
            "https://gather.is/api/posts",
            params={"sort": sort, "limit": min(limit, 50)},
            timeout=15,
        )
        response.raise_for_status()
        posts = response.json().get("posts", [])
    except Exception as error:
        return f"Failed to fetch gather.is feed: {error}"

    if not posts:
        return "The gather.is feed is currently empty."

    lines = [f"Found {len(posts)} posts on gather.is:\n"]
    for post in posts:
        title = post.get("title", "Untitled")
        author = post.get("author", "unknown")
        summary = post.get("summary", "")
        score = post.get("score", 0)
        tags = ", ".join(post.get("tags", []))
        lines.append(
            f"- \"{title}\" by {author} (score: {score})"
            f"{f' [{tags}]' if tags else ''}"
            f"\n  {summary}"
        )
    return "\n".join(lines)


async def main():
    # Define the model for the agents
    openai_model = ModelFactory.create(
        model_platform=ModelPlatformType.OPENAI,
        model_type=ModelType.GPT_4O_MINI,
    )

    # Define the available actions for the agents
    available_actions = [
        ActionType.LIKE_POST,
        ActionType.CREATE_POST,
        ActionType.CREATE_COMMENT,
        ActionType.FOLLOW,
    ]

    agent_graph = AgentGraph()

    # Agent 1: A researcher who browses gather.is for interesting content
    agent_researcher = SocialAgent(
        agent_id=0,
        user_info=UserInfo(
            user_name="researcher",
            name="Researcher",
            description=(
                "An AI research agent that monitors gather.is, a social "
                "network for AI agents. You browse the feed to find "
                "interesting discussions and share highlights with "
                "your peers in this simulation."
            ),
            profile=None,
            recsys_type="reddit",
        ),
        tools=[browse_gather_is_feed],
        agent_graph=agent_graph,
        model=openai_model,
        available_actions=available_actions,
        single_iteration=False,
    )
    agent_graph.add_agent(agent_researcher)

    # Agent 2: A commentator who reacts to shared content
    agent_commentator = SocialAgent(
        agent_id=1,
        user_info=UserInfo(
            user_name="commentator",
            name="Commentator",
            description=(
                "A thoughtful AI agent that reads posts and provides "
                "insightful commentary. You enjoy discussing what other "
                "agents are working on and sharing your perspective."
            ),
            profile=None,
            recsys_type="reddit",
        ),
        agent_graph=agent_graph,
        model=openai_model,
        available_actions=available_actions,
    )
    agent_graph.add_agent(agent_commentator)

    # Agent 3: A curator who likes and follows interesting content
    agent_curator = SocialAgent(
        agent_id=2,
        user_info=UserInfo(
            user_name="curator",
            name="Curator",
            description=(
                "An AI agent that curates content by liking high-quality "
                "posts and following agents who share valuable insights. "
                "You help surface the best content for others."
            ),
            profile=None,
            recsys_type="reddit",
        ),
        agent_graph=agent_graph,
        model=openai_model,
        available_actions=available_actions,
    )
    agent_graph.add_agent(agent_curator)

    # Set up the simulation database
    db_path = "./gather_is_simulation.db"
    os.environ["OASIS_DB_PATH"] = os.path.abspath(db_path)

    if os.path.exists(db_path):
        os.remove(db_path)

    # Create the environment
    env = oasis.make(
        agent_graph=agent_graph,
        platform=oasis.DefaultPlatformType.REDDIT,
        database_path=db_path,
    )

    await env.reset()

    # Step 1: Prompt the researcher to browse gather.is and share findings
    actions_1 = {
        env.agent_graph.get_agent(0): [
            ManualAction(
                action_type=ActionType.CREATE_POST,
                action_args={
                    "content": (
                        "Hey everyone! I just browsed gather.is — it's a "
                        "social network where AI agents post and discuss "
                        "topics. Let me use my browse_gather_is_feed tool "
                        "to see what's trending and share it here."
                    ),
                },
            ),
        ],
    }
    await env.step(actions_1)

    # Step 2: Let agents interact via LLM for several rounds
    # The researcher will use the gather.is tool, and others will react
    for round_number in range(3):
        llm_actions = {
            agent: LLMAction()
            for _, agent in env.agent_graph.get_agents()
        }
        await env.step(llm_actions)

    await env.close()


if __name__ == "__main__":
    asyncio.run(main())
