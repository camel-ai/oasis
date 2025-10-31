#!/usr/bin/env python3
from __future__ import annotations

import argparse
import asyncio
import os
from pathlib import Path

from camel.models import ModelFactory
from camel.types import ModelPlatformType, ModelType

import oasis
from oasis import (ActionType, LLMAction, ManualAction,
                   generate_twitter_agent_graph)


async def run(db_path: Path, personas_csv: Path, steps: int, seed_post: bool) -> None:
    openai_model = ModelFactory.create(
        model_platform=ModelPlatformType.OPENAI,
        model_type=ModelType.GPT_4O_MINI,
    )

    available_actions = ActionType.get_default_twitter_actions()

    agent_graph = await generate_twitter_agent_graph(
        profile_path=str(personas_csv), model=openai_model, available_actions=available_actions
    )

    os.environ["OASIS_DB_PATH"] = os.path.abspath(str(db_path))
    db_path.parent.mkdir(parents=True, exist_ok=True)
    if db_path.exists():
        os.remove(db_path)

    env = oasis.make(
        agent_graph=agent_graph,
        platform=oasis.DefaultPlatformType.TWITTER,
        database_path=str(db_path),
    )

    await env.reset()

    # Optional seed posts to kick off content
    if seed_post and agent_graph.get_num_nodes() > 0:
        actions = {}
        actions[env.agent_graph.get_agent(0)] = ManualAction(
            action_type=ActionType.CREATE_POST,
            action_args={"content": "Starting the simulation <LBL:SUPPORTIVE>"},
        )
        await env.step(actions)

    # Run LLM-driven steps
    for _ in range(steps):
        actions = {agent: LLMAction() for _, agent in env.agent_graph.get_agents()}
        await env.step(actions)

    await env.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run MVP simulation to produce posts")
    parser.add_argument("--db", type=str, default="./data/mvp/oasis_mvp.db")
    parser.add_argument("--personas", type=str, default="./data/personas_mvp.csv")
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--seed-post", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    asyncio.run(
        run(db_path=Path(args.db), personas_csv=Path(args.personas), steps=args.steps, seed_post=args.seed_post)
    )


if __name__ == "__main__":
    main()


