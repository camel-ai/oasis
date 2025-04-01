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
import json
import logging
import os
import random
import re
import sys
from datetime import datetime
from typing import Any

from camel.agents import ChatAgent
from camel.models import ModelFactory
from camel.types import ModelPlatformType, ModelType
from colorama import Back
from pydantic import BaseModel

from oasis.clock.clock import Clock
from oasis.social_agent.agents_generator import (gen_control_agents_with_data,
                                                 generate_reddit_agents)
from oasis.social_platform.channel import Channel
from oasis.social_platform.platform import Platform
from oasis.social_platform.typing import ActionType
from scripts.base.listen import redis, redis_publish

# os.environ["HTTP_PROXY"] = "http://127.0.0.1:7890"
# os.environ["HTTPS_PROXY"] = "http://127.0.0.1:7890"
sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

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

DATA_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "data")
DEFAULT_DB_PATH = os.path.join(DATA_DIR, "mock_reddit.db")
DEFAULT_USER_PATH = os.path.join(DATA_DIR, "reddit",
                                 "filter_user_results.json")
DEFAULT_PAIR_PATH = os.path.join(DATA_DIR, "reddit", "RS-RC-pairs.json")

ROUND_POST_NUM = 20


async def running(
    db_path: str | None = DEFAULT_DB_PATH,
    user_path: str | None = DEFAULT_USER_PATH,
    num_timesteps: int = 5,
    recsys_type: str = "twrec-bert",
    controllable_user: bool = True,
    activate_prob_normal: float = 0.2,
    activate_prob_celebrity: float = 0.1,
    inference_configs: dict[str, Any] | None = None,
    refresh_rec_post_count: int = 5,
    user_number: int = 1,
    action_space_file_path: str = None,
    content: str = "",
    content_id: int = 0,
) -> None:
    current_timestep = str(0)
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
        current_timestep=current_timestep,
        use_openai_embedding=True,
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

    async def trigger(timestep: int, predict_content: str):
        # os.environ["SANDBOX_TIME"] = str(timestep * 3)
        activate_prob_celebrity = 0.1
        activate_prob_normal = 0.2
        infra.current_timestep = str(timestep * 3)
        print(Back.GREEN + f"timestep:{timestep}" + Back.RESET)
        # social_log.info(f"timestep:{timestep + 1}.")

        player_agent = agent_graph.get_agent(0)
        await player_agent.perform_action_by_hci(
            predict_content, 0 if predict_content != "" else 6)
        if timestep == 1:
            # First try rule-based detection for English/Chinese
            global language_type
            language_type = detect_language(predict_content)

            print(f"language_type: {language_type}")
            for _, agent in agent_graph.get_agents():
                if language_type == "chinese":
                    agent.model_backend = ModelFactory.create(
                        model_platform=ModelPlatformType.QWEN,
                        model_type=ModelType.QWEN_PLUS,
                    )
                    agent.model_backend.model_config_dict['temperature'] = 1.0

        await infra.update_rec_table()
        # social_log.info("update rec table.")
        tasks = []
        round_num = 1
        # 爆款概率
        if random.random() < 0.05:
            round_num = 3
        if random.random() < 0.1:
            activate_prob_celebrity = 0.05
            activate_prob_normal = 0.05
        for i in range(round_num):
            for _, agent in agent_graph.get_agents():
                if agent.agent_id == 33:
                    assert agent.user_info.profile['other_info'][
                        'realname'] == "Liu Qiangdong"
                if agent.user_info.is_controllable is False:
                    if agent.agent_id <= 33:  # celebrity
                        if random.random() < activate_prob_celebrity:
                            agent.language_type = language_type
                            tasks.append(agent.perform_action_by_llm())
                    else:  # normal
                        if random.random() < activate_prob_normal:
                            agent.language_type = language_type
                            tasks.append(agent.perform_action_by_llm())
        random.shuffle(tasks)
        await asyncio.gather(*tasks)
        redis_publish(content_id, {"action": "predict_end", "step": timestep})

    step = 1
    await trigger(step, content)

    channel_name = f"predict_new_{content_id}"
    pubsub = redis.pubsub()
    pubsub.subscribe(channel_name)

    for message in pubsub.listen():
        print(message)
        if message["type"] != "message":
            continue
        step += 1
        await trigger(step, json.loads(message["data"]))

    # num_timesteps = 1
    # for timestep in range(1, num_timesteps + 1):
    #     os.environ["SANDBOX_TIME"] = str(timestep * 3)
    #     print(Back.GREEN + f"timestep:{timestep}" + Back.RESET)
    #     # social_log.info(f"timestep:{timestep + 1}.")

    #     player_agent = agent_graph.get_agent(0)
    #     await player_agent.perform_action_by_hci(content)

    #     await infra.update_rec_table()
    #     # social_log.info("update rec table.")
    #     tasks = []
    #     for _, agent in agent_graph.get_agents():
    #         if agent.user_info.is_controllable is False:
    #             if random.random() < activate_prob:
    #                 tasks.append(agent.perform_action_by_llm())
    #     random.shuffle(tasks)
    #     await asyncio.gather(*tasks)

    await twitter_channel.write_to_receive_queue((None, None, ActionType.EXIT))

    await twitter_task

    # social_log.info("Simulation finish!")
    log_info("Simulation finish!")


def log_info(message: str) -> None:
    print(f"INFO - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - {message}")


def detect_language(text: str) -> str:
    """
    Detect if text is English, Chinese, or other using rule-based methods.
    
    Args:
        text: The text to analyze
        
    Returns:
        'english', 'chinese', or 'other'
    """
    if not text or len(text.strip()) == 0:
        return "english"  # Default to English for empty text
    
    # Remove URLs, mentions, hashtags, and emojis to focus on the actual text
    cleaned_text = re.sub(r'https?://\S+|@\w+|#\w+|[\U00010000-\U0010ffff]', '', text)
    cleaned_text = cleaned_text.strip()
    
    if not cleaned_text:
        return "english"  # Default to English if only URLs/mentions/hashtags
    
    # Count Chinese characters
    chinese_chars = len(re.findall(r'[\u4e00-\u9fff]', cleaned_text))
    
    # Count total characters (excluding whitespace)
    total_chars = len(re.sub(r'\s', '', cleaned_text))
    if total_chars == 0:
        return "english"
    
    # Calculate Chinese character ratio
    chinese_ratio = chinese_chars / total_chars if total_chars > 0 else 0
    
    # If more than 40% characters are Chinese, consider it Chinese
    if chinese_ratio > 0.4:
        return "chinese"
    
    # If text only contains ASCII characters (0-127), consider it English
    if all(ord(c) < 128 for c in cleaned_text):
        return "english"
    
    # For any other case, return "other" to trigger OpenAI detection
    return "other"


def parse_args():
    parser = argparse.ArgumentParser(description="Simulation parameters")
    parser.add_argument("--db_path", type=str, default="twitter.db")
    parser.add_argument("--content", type=str)
    parser.add_argument("--content_id", type=int)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    current_timestep = str(0)

    inference_configs = {
        "model_type": "gpt-4o-mini",
        "is_openai_model": True,
    }

    user_profile_root_path = './data/game/'
    all_user_profile_path = ["mixed_agents.json"]
    user_profile_path = user_profile_root_path + random.choice(
        all_user_profile_path)

    asyncio.run(
        running(db_path=args.db_path,
                user_path=user_profile_path,
                num_timesteps=100,
                recsys_type="twhin-bert",
                inference_configs=inference_configs,
                content=args.content,
                content_id=args.content_id))
