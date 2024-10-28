from __future__ import annotations

import argparse
import asyncio
import logging
import os
import random
from datetime import datetime
from typing import Any

import pandas as pd
from colorama import Back
from yaml import safe_load

from social_simulation.clock.clock import Clock
from social_simulation.social_agent.agents_generator import generate_agents
from social_simulation.social_platform.channel import Channel
from social_simulation.social_platform.config import Neo4jConfig
from social_simulation.social_platform.platform import Platform
from social_simulation.social_platform.typing import ActionType

logger = logging.getLogger("twitter_simulation")
logger.setLevel("DEBUG")
file_handler = logging.FileHandler("twitter_simulation.log")
file_handler.setLevel("DEBUG")
file_handler.setFormatter(
    logging.Formatter("%(levelname)s - %(asctime)s - %(name)s - %(message)s"))
logger.addHandler(file_handler)
stream_handler = logging.StreamHandler()
stream_handler.setLevel("DEBUG")
stream_handler.setFormatter(
    logging.Formatter("%(levelname)s - %(asctime)s - %(name)s - %(message)s"))
logger.addHandler(stream_handler)

parser = argparse.ArgumentParser(description="Arguments for script.")
parser.add_argument(
    "--config_path",
    type=str,
    help="Path to the YAML config file.",
    required=False,
    default="",
)

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
DEFAULT_DB_PATH = ":memory:"
DEFAULT_CSV_PATH = os.path.join(DATA_DIR, "user_all_id_time.csv")


async def running(
    db_path: str | None = DEFAULT_DB_PATH,
    neo4j_username: str | None = None,
    neo4j_password: str | None = None,
    neo4j_uri: str | None = None,
    csv_path: str | None = DEFAULT_CSV_PATH,
    num_timesteps: int = 3,
    clock_factor: int = 60,
    recsys_type: str = "twitter",
    model_configs: dict[str, Any] | None = None,
) -> None:
    db_path = DEFAULT_DB_PATH if db_path is None else db_path
    csv_path = DEFAULT_CSV_PATH if csv_path is None else csv_path
    neo4j_config = Neo4jConfig(
        uri=neo4j_uri,
        username=neo4j_username,
        password=neo4j_password,
    )
    if not neo4j_config.is_valid():
        neo4j_config = None
        logger.warning("Neo4j config is not valid. "
                       "Using igraph backend for social graph.")
    if os.path.exists(db_path):
        os.remove(db_path)

    start_time = datetime.now()
    logger.info(f"Start time: {start_time}")
    clock = Clock(k=clock_factor)
    channel = Channel()
    infra = Platform(
        db_path,
        channel,
        clock,
        start_time,
        recsys_type=recsys_type,
    )
    task = asyncio.create_task(infra.running())
    model_configs = model_configs or {}
    agent_graph = await generate_agents(
        agent_info_path=csv_path,
        channel=channel,
        **model_configs,
    )
    # agent_graph.visualize("initial_social_graph.png")

    # 从label_clean_v7中把开始时间读出来
    # TODO this part is not good. We do not have data/label_clean_v7.csv
    try:
        all_topic_df = pd.read_csv("data/label_clean_v7.csv")
        if "False" in csv_path or "True" in csv_path:
            if "-" not in csv_path:
                topic_name = csv_path.split("/")[-1].split(".")[0]
            else:
                topic_name = csv_path.split("/")[-1].split(".")[0].split(
                    "-")[0]
            start_time = (
                all_topic_df[all_topic_df["topic_name"] ==
                             topic_name]["start_time"].item().split(" ")[1])
            start_hour = int(start_time.split(":")[0]) + float(
                int(start_time.split(":")[1]) / 60)
    except Exception:
        print("No real-world data, let start_hour be 13")
        start_hour = 13

    for timestep in range(1, num_timesteps + 1):
        os.environ["SANDBOX_TIME"] = str(timestep * 3)
        logger.info(f"timestep:{timestep}")
        print(Back.GREEN + f"timestep:{timestep}" + Back.RESET)
        await infra.update_rec_table()
        # 0.2 * timestep here means 12 minutes
        simulation_time_hour = start_hour + 0.05 * timestep
        for _, agent in agent_graph.get_agents():
            if agent.user_info.is_controllable is False:
                agent_ac_prob = random.random()
                threshold = agent.user_info.profile["other_info"][
                    "active_threshold"][int(simulation_time_hour % 24)]
                if agent_ac_prob < threshold:
                    await agent.perform_action_by_llm()
            else:
                await agent.perform_action_by_hci()
        # agent_graph.visualize(f"timestep_{timestep}_social_graph.png")

    await channel.write_to_receive_queue((None, None, ActionType.EXIT))
    await task


if __name__ == "__main__":
    args = parser.parse_args()
    os.environ["SANDBOX_TIME"] = str(0)
    if os.path.exists(args.config_path):
        with open(args.config_path, "r") as f:
            cfg = safe_load(f)
        data_params = cfg.get("data")
        simulation_params = cfg.get("simulation")
        model_configs = cfg.get("model")

        asyncio.run(
            running(
                **data_params,
                **simulation_params,
                model_configs=model_configs,
            ))
    else:
        asyncio.run(running())
    logger.info("Simulation finished.")
