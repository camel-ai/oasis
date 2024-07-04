from __future__ import annotations

import argparse
import asyncio
import json
import os
from datetime import datetime

from colorama import Back
from yaml import safe_load

from social_simulation.clock.clock import Clock
from social_simulation.social_agent.agents_generator import (
    gen_control_agents_with_data, generate_reddit_agents)
from social_simulation.social_platform.channel import Channel
from social_simulation.social_platform.platform import Platform
from social_simulation.social_platform.typing import ActionType
from social_simulation.testing.show_db import print_db_contents

parser = argparse.ArgumentParser(description="Arguments for script.")
parser.add_argument(
    "--config_path",
    type=str,
    help="Path to the YAML config file.",
    required=False,
    default="",
)

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
DEFAULT_DB_PATH = os.path.join(DATA_DIR, "mock_reddit.db")
DEFAULT_USER_PATH = os.path.join(DATA_DIR, "reddit",
                                 "filter_user_results.json")
DEFAULT_PAIR_PATH = os.path.join(DATA_DIR, "reddit", "RS-RC-pairs.json")

ROUND_POST_NUM = 20


async def running(
    db_path: str | None = DEFAULT_DB_PATH,
    user_path: str | None = DEFAULT_USER_PATH,
    pair_path: str | None = DEFAULT_PAIR_PATH,
    num_timesteps: int = 3,
    clock_factor: int = 60,
    recsys_type: str = "reddit",
    controllable_user: bool = True,
    allow_self_rating: bool = False,
    show_score: bool = True,
    rec_update_time: int = 40,
) -> None:
    db_path = DEFAULT_DB_PATH if db_path is None else db_path
    user_path = DEFAULT_USER_PATH if user_path is None else user_path
    pair_path = DEFAULT_PAIR_PATH if pair_path is None else pair_path
    if os.path.exists(db_path):
        os.remove(db_path)

    start_time = datetime.now()
    clock = Clock(k=clock_factor)
    channel = Channel()
    infra = Platform(
        db_path,
        channel,
        clock,
        start_time,
        allow_self_rating=allow_self_rating,
        show_score=show_score,
        recsys_type=recsys_type,
        rec_update_time=rec_update_time,
    )
    task = asyncio.create_task(infra.running())
    if not controllable_user:
        raise ValueError("Uncontrollable user is not supported")
    else:
        agent_graph, agent_user_id_mapping = \
            await gen_control_agents_with_data(
                channel,
                2,
            )
        agent_graph = await generate_reddit_agents(
            user_path,
            channel,
            agent_graph,
            agent_user_id_mapping,
        )
    with open(pair_path, "r") as f:
        pairs = json.load(f)

    for timestep in range(num_timesteps):
        print(Back.GREEN + f"timestep:{timestep}" + Back.RESET)
        post_agent = agent_graph.get_agent(0)
        rate_agent = agent_graph.get_agent(1)

        for i in range(ROUND_POST_NUM):
            rs_rc_index = i + timestep * ROUND_POST_NUM
            if rs_rc_index >= len(pairs):
                break
            else:
                response = await post_agent.perform_action_by_data(
                    'create_post', content=pairs[rs_rc_index]["submission"])
                post_id = response['post_id']
                response = await post_agent.perform_action_by_data(
                    'create_comment',
                    post_id=post_id,
                    content=pairs[rs_rc_index]["comment"])
                comment_id = response['comment_id']

                if pairs[rs_rc_index]["group"] == 'up':
                    await rate_agent.perform_action_by_data(
                        'like_comment', comment_id)
                elif pairs[rs_rc_index]["group"] == 'down':
                    await rate_agent.perform_action_by_data(
                        'dislike_comment', comment_id)
                elif pairs[rs_rc_index]["group"] == 'control':
                    pass
                else:
                    raise ValueError("Unsupported value of 'group'")

        for _, node_data in agent_graph.get_agents():
            agent = node_data['agent']
            if agent.user_info.is_controllable is False:
                await agent.perform_action_by_llm()

    await channel.write_to_receive_queue((None, None, ActionType.EXIT))
    await task

    print_db_contents(db_path)


if __name__ == "__main__":
    args = parser.parse_args()
    if os.path.exists(args.config_path):
        with open(args.config_path, "r") as f:
            cfg = safe_load(f)
        data_params = cfg.get("data")
        simulation_params = cfg.get("simulation")
        asyncio.run(running(**data_params, **simulation_params))
    else:
        asyncio.run(running())
