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
from __future__ import annotations

import ast
import asyncio
import json
import random
from typing import Any, List

import numpy as np
import pandas as pd
import tqdm
from camel.memories import MemoryRecord
from camel.messages import BaseMessage
from camel.models import BaseModelBackend
from camel.types import ModelType, OpenAIBackendRole

from oasis.social_agent import AgentGraph, SocialAgent
from oasis.social_platform import Channel, Platform
from oasis.social_platform.config import Neo4jConfig, UserInfo
from oasis.social_platform.typing import ActionType


async def generate_agents(
    agent_info_path: str,
    twitter_channel: Channel,
    model: BaseModelBackend,
    start_time,
    recsys_type: str = "twitter",
    twitter: Platform = None,
    num_agents: int = 26,
    available_actions: list[ActionType] = None,
    model_random_seed: int = 42,
    cfgs: list[Any] | None = None,
    neo4j_config: Neo4jConfig | None = None,
) -> AgentGraph:
    """TODO: need update the description of args and check
    Generate and return a dictionary of agents from the agent
    information CSV file. Each agent is added to the database and
    their respective profiles are updated.

    Args:
        agent_info_path (str): The file path to the agent information CSV file.
        channel (Channel): Information channel.
        num_agents (int): Number of agents.
        action_space_prompt (str): determine the action space of agents.
        model_random_seed (int): Random seed to randomly assign model to
            each agent. (default: 42)
        cfgs (list, optional): List of configuration. (default: `None`)
        neo4j_config (Neo4jConfig, optional): Neo4j graph database
            configuration. (default: `None`)

    Returns:
        dict: A dictionary of agent IDs mapped to their respective agent
            class instances.
    """
    random.seed(model_random_seed)
    model_types = []
    model_temperatures = []
    model_config_dict = {}
    for _, cfg in enumerate(cfgs):
        model_type = cfg["model_type"]
        model_config_dict[model_type] = cfg
        model_types.extend([model_type] * cfg["num"])
        temperature = cfg.get("temperature", 0.0)
        model_temperatures.extend([temperature] * cfg["num"])
    random.shuffle(model_types)
    assert len(model_types) == num_agents
    agent_info = pd.read_csv(agent_info_path)
    # agent_info = agent_info[:10000]
    assert len(model_types) == len(agent_info), (
        f"Mismatch between the number of agents "
        f"and the number of models, with {len(agent_info)} "
        f"agents and {len(model_types)} models.")

    mbti_types = ["INTJ", "ENTP", "INFJ", "ENFP"]

    freq = list(agent_info["activity_level_frequency"])
    all_freq = np.array([ast.literal_eval(fre) for fre in freq])
    normalized_prob = all_freq / np.max(all_freq)
    # Make sure probability is not too small
    normalized_prob[normalized_prob < 0.6] += 0.1
    normalized_prob = np.round(normalized_prob, 2)
    prob_list: list[float] = normalized_prob.tolist()

    agent_graph = (AgentGraph() if neo4j_config is None else AgentGraph(
        backend="neo4j",
        neo4j_config=neo4j_config,
    ))

    # agent_graph = []
    sign_up_list = []
    follow_list = []
    user_update1 = []
    user_update2 = []
    post_list = []

    for agent_id in range(len(agent_info)):
        profile = {
            "nodes": [],
            "edges": [],
            "other_info": {},
        }
        profile["other_info"]["user_profile"] = agent_info["user_char"][
            agent_id]
        profile["other_info"]["mbti"] = random.choice(mbti_types)
        profile["other_info"]["activity_level_frequency"] = ast.literal_eval(
            agent_info["activity_level_frequency"][agent_id])
        profile["other_info"]["active_threshold"] = prob_list[agent_id]

        user_info = UserInfo(
            name=agent_info["username"][agent_id],
            description=agent_info["description"][agent_id],
            profile=profile,
            recsys_type=recsys_type,
        )

        agent = SocialAgent(
            agent_id=agent_id,
            user_info=user_info,
            twitter_channel=twitter_channel,
            model=model,
            agent_graph=agent_graph,
            available_actions=available_actions,
        )

        agent_graph.add_agent(agent)
        num_followings = 0
        num_followers = 0
        # print('agent_info["following_count"]', agent_info["following_count"])
        if not agent_info["following_count"].empty:
            num_followings = int(agent_info["following_count"][agent_id])
        if not agent_info["followers_count"].empty:
            num_followers = int(agent_info["followers_count"][agent_id])

        sign_up_list.append((
            agent_id,
            agent_id,
            agent_info["username"][agent_id],
            agent_info["name"][agent_id],
            agent_info["description"][agent_id],
            start_time,
            num_followings,
            num_followers,
        ))

        following_id_list = ast.literal_eval(
            agent_info["following_agentid_list"][agent_id])
        if not isinstance(following_id_list, int):
            if len(following_id_list) != 0:
                for follow_id in following_id_list:
                    follow_list.append((agent_id, follow_id, start_time))
                    user_update1.append((agent_id, ))
                    user_update2.append((follow_id, ))
                    agent_graph.add_edge(agent_id, follow_id)

        previous_posts = ast.literal_eval(
            agent_info["previous_tweets"][agent_id])
        if len(previous_posts) != 0:
            for post in previous_posts:
                post_list.append((agent_id, post, start_time, 0, 0))

    # generate_log.info('agent gegenerate finished.')

    user_insert_query = (
        "INSERT INTO user (user_id, agent_id, user_name, name, bio, "
        "created_at, num_followings, num_followers) VALUES "
        "(?, ?, ?, ?, ?, ?, ?, ?)")
    twitter.pl_utils._execute_many_db_command(user_insert_query,
                                              sign_up_list,
                                              commit=True)

    follow_insert_query = (
        "INSERT INTO follow (follower_id, followee_id, created_at) "
        "VALUES (?, ?, ?)")
    twitter.pl_utils._execute_many_db_command(follow_insert_query,
                                              follow_list,
                                              commit=True)

    if not (agent_info["following_count"].empty
            and agent_info["followers_count"].empty):
        user_update_query1 = (
            "UPDATE user SET num_followings = num_followings + 1 "
            "WHERE user_id = ?")
        twitter.pl_utils._execute_many_db_command(user_update_query1,
                                                  user_update1,
                                                  commit=True)

        user_update_query2 = (
            "UPDATE user SET num_followers = num_followers + 1 "
            "WHERE user_id = ?")
        twitter.pl_utils._execute_many_db_command(user_update_query2,
                                                  user_update2,
                                                  commit=True)

    # generate_log.info('twitter followee update finished.')

    post_insert_query = (
        "INSERT INTO post (user_id, content, created_at, num_likes, "
        "num_dislikes) VALUES (?, ?, ?, ?, ?)")
    twitter.pl_utils._execute_many_db_command(post_insert_query,
                                              post_list,
                                              commit=True)

    # generate_log.info('twitter creat post finished.')

    return agent_graph


async def generate_agents_100w(
    agent_info_path: str,
    twitter_channel: Channel,
    start_time,
    model: BaseModelBackend,
    recsys_type: str = "twitter",
    twitter: Platform = None,
    num_agents: int = 26,
    available_actions: list[ActionType] = None,
    model_random_seed: int = 42,
    cfgs: list[Any] | None = None,
) -> List:
    """ TODO: need update the description of args.
    Generate and return a dictionary of agents from the agent
    information CSV file. Each agent is added to the database and
    their respective profiles are updated.

    Args:
        agent_info_path (str): The file path to the agent information CSV file.
        channel (Channel): Information channel.
        num_agents (int): Number of agents.
        action_space_prompt (str): determine the action space of agents.
        model_random_seed (int): Random seed to randomly assign model to
            each agent. (default: 42)

    Returns:
        dict: A dictionary of agent IDs mapped to their respective agent
            class instances.
    """
    random.seed(model_random_seed)
    model_types = []
    model_temperatures = []
    model_config_dict = {}
    for _, cfg in enumerate(cfgs):
        model_type = ModelType(cfg["model_type"])
        model_config_dict[model_type] = cfg
        model_types.extend([model_type] * cfg["num"])
        temperature = cfg.get("temperature", 0.0)
        model_temperatures.extend([temperature] * cfg["num"])
    random.shuffle(model_types)
    assert len(model_types) == num_agents
    agent_info = pd.read_csv(agent_info_path)
    # agent_info = agent_info[:10000]
    assert len(model_types) == len(agent_info), (
        f"Mismatch between the number of agents "
        f"and the number of models, with {len(agent_info)} "
        f"agents and {len(model_types)} models.")

    mbti_types = ["INTJ", "ENTP", "INFJ", "ENFP"]

    freq = list(agent_info["activity_level_frequency"])
    all_freq = np.array([ast.literal_eval(fre) for fre in freq])
    normalized_prob = all_freq / np.max(all_freq)
    # Make sure probability is not too small
    normalized_prob[normalized_prob < 0.6] += 0.1
    normalized_prob = np.round(normalized_prob, 2)
    prob_list: list[float] = normalized_prob.tolist()

    # TODO when setting 100w agents, the agentgraph class is too slow.
    # I use the list.
    agent_graph = []
    # agent_graph = (AgentGraph() if neo4j_config is None else AgentGraph(
    #     backend="neo4j",
    #     neo4j_config=neo4j_config,
    # ))

    # agent_graph = []
    sign_up_list = []
    follow_list = []
    user_update1 = []
    user_update2 = []
    post_list = []

    # precompute to speed up agent generation in one million scale
    _ = agent_info["following_agentid_list"].apply(ast.literal_eval)
    previous_tweets_lists = agent_info["previous_tweets"].apply(
        ast.literal_eval)
    activity_level_frequencies = agent_info["activity_level_frequency"].apply(
        ast.literal_eval)
    previous_tweets_lists = agent_info['previous_tweets'].apply(
        ast.literal_eval)
    following_id_lists = agent_info["following_agentid_list"].apply(
        ast.literal_eval)

    for agent_id in tqdm.tqdm(range(len(agent_info))):
        profile = {
            "nodes": [],
            "edges": [],
            "other_info": {},
        }
        profile["other_info"]["user_profile"] = agent_info["user_char"][
            agent_id]
        profile["other_info"]["mbti"] = random.choice(mbti_types)
        profile['other_info'][
            'activity_level_frequency'] = activity_level_frequencies[agent_id]
        profile["other_info"]["active_threshold"] = prob_list[agent_id]
        # TODO if you simulate one million agents, use active threshold below.
        # profile['other_info']['active_threshold'] = [0.01] * 24

        user_info = UserInfo(
            name=agent_info["username"][agent_id],
            description=agent_info["description"][agent_id],
            profile=profile,
            recsys_type=recsys_type,
        )

        model_type: ModelType = model_types[agent_id]

        agent = SocialAgent(
            agent_id=agent_id,
            user_info=user_info,
            twitter_channel=twitter_channel,
            model=model,
            agent_graph=agent_graph,
            available_actions=available_actions,
        )

        agent_graph.append(agent)
        num_followings = 0
        num_followers = 0
        # print('agent_info["following_count"]', agent_info["following_count"])

        # TODO some data does not cotain this key.
        if 'following_count' not in agent_info.columns:
            agent_info['following_count'] = 0
        if 'followers_count' not in agent_info.columns:
            agent_info['followers_count'] = 0

        if not agent_info["following_count"].empty:
            num_followings = agent_info["following_count"][agent_id]
        if not agent_info["followers_count"].empty:
            num_followers = agent_info["followers_count"][agent_id]

        sign_up_list.append((
            agent_id,
            agent_id,
            agent_info["username"][agent_id],
            agent_info["name"][agent_id],
            agent_info["description"][agent_id],
            start_time,
            num_followings,
            num_followers,
        ))

        following_id_list = following_id_lists[agent_id]

        # TODO If we simulate 1 million agents, we can not use agent_graph
        # class. It is not scalble.
        if not isinstance(following_id_list, int):
            if len(following_id_list) != 0:
                for follow_id in following_id_list:
                    follow_list.append((agent_id, follow_id, start_time))
                    user_update1.append((agent_id, ))
                    user_update2.append((follow_id, ))
                    # agent_graph.add_edge(agent_id, follow_id)

        previous_posts = previous_tweets_lists[agent_id]
        if len(previous_posts) != 0:
            for post in previous_posts:
                post_list.append((agent_id, post, start_time, 0, 0))

    # generate_log.info('agent gegenerate finished.')

    user_insert_query = (
        "INSERT INTO user (user_id, agent_id, user_name, name, bio, "
        "created_at, num_followings, num_followers) VALUES "
        "(?, ?, ?, ?, ?, ?, ?, ?)")
    twitter.pl_utils._execute_many_db_command(user_insert_query,
                                              sign_up_list,
                                              commit=True)

    follow_insert_query = (
        "INSERT INTO follow (follower_id, followee_id, created_at) "
        "VALUES (?, ?, ?)")
    twitter.pl_utils._execute_many_db_command(follow_insert_query,
                                              follow_list,
                                              commit=True)

    if not (agent_info["following_count"].empty
            and agent_info["followers_count"].empty):
        user_update_query1 = (
            "UPDATE user SET num_followings = num_followings + 1 "
            "WHERE user_id = ?")
        twitter.pl_utils._execute_many_db_command(user_update_query1,
                                                  user_update1,
                                                  commit=True)

        user_update_query2 = (
            "UPDATE user SET num_followers = num_followers + 1 "
            "WHERE user_id = ?")
        twitter.pl_utils._execute_many_db_command(user_update_query2,
                                                  user_update2,
                                                  commit=True)

    # generate_log.info('twitter followee update finished.')

    post_insert_query = (
        "INSERT INTO post (user_id, content, created_at, num_likes, "
        "num_dislikes) VALUES (?, ?, ?, ?, ?)")
    twitter.pl_utils._execute_many_db_command(post_insert_query,
                                              post_list,
                                              commit=True)

    # generate_log.info('twitter creat post finished.')

    return agent_graph


async def generate_controllable_agents(
    channel: Channel,
    control_user_num: int,
) -> tuple[AgentGraph, dict]:
    agent_graph = AgentGraph()
    agent_user_id_mapping = {}
    for i in range(control_user_num):
        user_info = UserInfo(
            is_controllable=True,
            profile={"other_info": {
                "user_profile": "None"
            }},
            recsys_type="reddit",
        )
        # controllable的agent_id全都在llm agent的agent_id的前面
        agent = SocialAgent(i, user_info, channel, agent_graph=agent_graph)
        # Add agent to the agent graph
        agent_graph.add_agent(agent)

        username = input(f"Please input username for agent {i}: ")
        name = input(f"Please input name for agent {i}: ")
        bio = input(f"Please input bio for agent {i}: ")

        response = await agent.env.action.sign_up(username, name, bio)
        user_id = response["user_id"]
        agent_user_id_mapping[i] = user_id

    for i in range(control_user_num):
        for j in range(control_user_num):
            agent = agent_graph.get_agent(i)
            # controllable agent互相也全部关注
            if i != j:
                user_id = agent_user_id_mapping[j]
                await agent.env.action.follow(user_id)
                agent_graph.add_edge(i, j)
    return agent_graph, agent_user_id_mapping


async def gen_control_agents_with_data(
    channel: Channel,
    control_user_num: int,
) -> tuple[AgentGraph, dict]:
    agent_graph = AgentGraph()
    agent_user_id_mapping = {}
    for i in range(control_user_num):
        user_info = UserInfo(
            is_controllable=True,
            profile={
                "other_info": {
                    "user_profile": "None",
                    "gender": "None",
                    "mbti": "None",
                    "country": "None",
                    "age": "None",
                }
            },
            recsys_type="reddit",
        )
        # controllable的agent_id全都在llm agent的agent_id的前面
        agent = SocialAgent(i, user_info, channel, agent_graph=agent_graph)
        # Add agent to the agent graph
        agent_graph.add_agent(agent)
        user_name = "momo"
        name = "momo"
        bio = "None."
        response = await agent.env.action.sign_up(user_name, name, bio)
        user_id = response["user_id"]
        agent_user_id_mapping[i] = user_id

    return agent_graph, agent_user_id_mapping


async def generate_reddit_agents(
    agent_info_path: str,
    twitter_channel: Channel,
    agent_graph: AgentGraph | None = AgentGraph,
    agent_user_id_mapping: dict[int, int] | None = None,
    follow_post_agent: bool = False,
    mute_post_agent: bool = False,
    model: BaseModelBackend = None,
    available_actions: list[ActionType] = None,
) -> AgentGraph:
    if agent_user_id_mapping is None:
        agent_user_id_mapping = {}
    if agent_graph is None:
        agent_graph = AgentGraph()

    control_user_num = agent_graph.get_num_nodes()

    with open(agent_info_path, "r") as file:
        agent_info = json.load(file)

    async def process_agent(i):
        # Instantiate an agent
        profile = {
            "nodes": [],  # Relationships with other agents
            "edges": [],  # Relationship details
            "other_info": {},
        }
        # Update agent profile with additional information
        profile["other_info"]["user_profile"] = agent_info[i]["persona"]
        profile["other_info"]["mbti"] = agent_info[i]["mbti"]
        profile["other_info"]["gender"] = agent_info[i]["gender"]
        profile["other_info"]["age"] = agent_info[i]["age"]
        profile["other_info"]["country"] = agent_info[i]["country"]

        user_info = UserInfo(
            name=agent_info[i]["username"],
            description=agent_info[i]["bio"],
            profile=profile,
            recsys_type="reddit",
        )

        agent = SocialAgent(
            agent_id=i + control_user_num,
            user_info=user_info,
            twitter_channel=twitter_channel,
            agent_graph=agent_graph,
            model=model,
            available_actions=available_actions,
        )

        # Add agent to the agent graph
        agent_graph.add_agent(agent)

        # Sign up agent and add their information to the database
        # print(f"Signing up agent {agent_info['username'][i]}...")
        response = await agent.env.action.sign_up(agent_info[i]["username"],
                                                  agent_info[i]["realname"],
                                                  agent_info[i]["bio"])
        user_id = response["user_id"]
        agent_user_id_mapping[i + control_user_num] = user_id

        if follow_post_agent:
            await agent.env.action.follow(1)
            content = """
{
    "reason": "He is my friend, and I would like to follow him "
              "on social media.",
    "functions": [
        {
            "name": "follow",
            "arguments": {
                "user_id": 1
            }
        }
    ]
}
"""

            agent_msg = BaseMessage.make_assistant_message(
                role_name="Assistant", content=content)
            agent.memory.write_record(
                MemoryRecord(agent_msg, OpenAIBackendRole.ASSISTANT))
        elif mute_post_agent:
            await agent.env.action.mute(1)
            content = """
{
    "reason": "He is my enemy, and I would like to mute him on social media.",
    "functions": [{
        "name": "mute",
        "arguments": {
            "user_id": 1
        }
}
"""
            agent_msg = BaseMessage.make_assistant_message(
                role_name="Assistant", content=content)
            agent.memory.write_record(
                MemoryRecord(agent_msg, OpenAIBackendRole.ASSISTANT))

    tasks = [process_agent(i) for i in range(len(agent_info))]
    await asyncio.gather(*tasks)

    return agent_graph
