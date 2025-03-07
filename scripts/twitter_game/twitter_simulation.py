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
# flake8: noqa: E402
from __future__ import annotations

import argparse
import asyncio
import logging
import os
import random
import sys
from datetime import datetime
from typing import Any

from colorama import Back

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from oasis.clock.clock import Clock
from oasis.social_agent.agents_generator import (gen_control_agents_with_data,
                                                 generate_reddit_agents)
from oasis.social_platform.channel import Channel
from oasis.social_platform.platform import Platform
from oasis.social_platform.typing import ActionType

logging.basicConfig(level=logging.CRITICAL)
# social_log = logging.getLogger(name="social")
# social_log.setLevel("DEBUG")
# now = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
# file_handler = logging.FileHandler(f"./log/social-{str(now)}.log",
#                                    encoding="utf-8")
# file_handler.setLevel("DEBUG")
# file_handler.setFormatter(
#     logging.Formatter("%(levelname)s - %(asctime)s - %(name)s - %(message)s"))
# social_log.addHandler(file_handler)
# stream_handler = logging.StreamHandler()
# stream_handler.setLevel("DEBUG")
# stream_handler.setFormatter(
#     logging.Formatter("%(levelname)s - %(asctime)s - %(name)s - %(message)s"))
# social_log.addHandler(stream_handler)

parser = argparse.ArgumentParser(description="Arguments for script.")
parser.add_argument(
    "--config_path",
    type=str,
    help="Path to the YAML config file.",
    required=False,
    default="",
)

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "data")
DEFAULT_DB_PATH = os.path.join(DATA_DIR, "mock_reddit.db")
DEFAULT_USER_PATH = os.path.join(DATA_DIR, "reddit",
                                 "filter_user_results.json")
DEFAULT_PAIR_PATH = os.path.join(DATA_DIR, "reddit", "RS-RC-pairs.json")

ROUND_POST_NUM = 20


async def running(
    db_path: str | None = DEFAULT_DB_PATH,
    user_path: str | None = DEFAULT_USER_PATH,
    num_timesteps: int = 3,
    recsys_type: str = "twrec-bert",
    controllable_user: bool = True,
    activate_prob: float = 0.2,
    inference_configs: dict[str, Any] | None = None,
    refresh_rec_post_count: int = 5,
    user_number: int = 1,
    action_space_file_path: str = None,
    content_id: int = 0,
) -> None:
    os.environ["SANDBOX_TIME"] = str(0)
    db_path = DEFAULT_DB_PATH if db_path is None else db_path
    user_path = DEFAULT_USER_PATH if user_path is None else user_path
    if os.path.exists(db_path):
        os.remove(db_path)

    start_time = datetime(2024, 8, 6, 8, 0)
    clock = Clock(k=60)
    twitter_channel = Channel()
    inference_channel = Channel()

    infra = Platform(
        db_path,
        twitter_channel,
        clock,
        start_time,
        recsys_type=recsys_type,
        refresh_rec_post_count=refresh_rec_post_count,
        content_id=content_id,
    )

    twitter_task = asyncio.create_task(infra.running())

    if inference_configs["model_type"][:3] == "gpt":
        is_openai_model = True
    if not controllable_user:
        raise ValueError("Uncontrollable user is not supported")
    else:
        agent_graph, id_mapping = await gen_control_agents_with_data(
            twitter_channel,
            user_number,
        )
        agent_graph = await generate_reddit_agents(
            agent_info_path=user_path,
            twitter_channel=twitter_channel,
            inference_channel=inference_channel,
            agent_graph=agent_graph,
            agent_user_id_mapping=id_mapping,
            follow_post_agent=False,
            mute_post_agent=False,
            action_space_prompt=None,
            model_type=inference_configs["model_type"],
            is_openai_model=is_openai_model,
        )

    for timestep in range(1, num_timesteps + 1):
        os.environ["SANDBOX_TIME"] = str(timestep * 3)
        print(Back.GREEN + f"timestep:{timestep}" + Back.RESET)
        # social_log.info(f"timestep:{timestep + 1}.")

        player_agent = agent_graph.get_agent(0)
        await player_agent.perform_action_by_hci()

        await infra.update_rec_table()
        # social_log.info("update rec table.")
        tasks = []
        for _, agent in agent_graph.get_agents():
            if agent.user_info.is_controllable is False:
                if random.random() < activate_prob:
                    tasks.append(agent.perform_action_by_llm())
        random.shuffle(tasks)
        await asyncio.gather(*tasks)

    await twitter_channel.write_to_receive_queue((None, None, ActionType.EXIT))

    await twitter_task

    # social_log.info("Simulation finish!")
    log_info("Simulation finish!")


def log_info(message: str) -> None:
    print(f"INFO - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - {message}")


def parse_args():
    parser = argparse.ArgumentParser(description="Simulation parameters")
    parser.add_argument("--db_path", type=str, default='twitter.db')
    parser.add_argument("--user_path",
                        type=str,
                        default=f'{DATA_DIR}/reddit/user_data_36.json')
    parser.add_argument("--content_id", type=int)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    os.environ["SANDBOX_TIME"] = str(0)

    inference_configs = {
        "model_type": "gpt-4o",
        "is_openai_model": True,
    }

    asyncio.run(
        running(db_path=args.db_path,
                user_path=args.user_path,
                num_timesteps=3,
                recsys_type="twhin-bert",
                inference_configs=inference_configs,
                content_id=args.content_id))
