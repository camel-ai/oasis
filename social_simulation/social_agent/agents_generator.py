from __future__ import annotations

import ast
import asyncio
import json
import random
from typing import Any

import numpy as np
import pandas as pd
from camel.types.enums import ModelType

from social_simulation.social_agent import AgentGraph, SocialAgent
from social_simulation.social_platform import Channel
from social_simulation.social_platform.config import UserInfo


async def generate_agents(
    agent_info_path: str,
    twitter_channel: Channel,
    inference_channel: Channel,
    num_agents: int = 26,
    model_random_seed: int = 42,
    cfgs: list[Any] | None = None,
) -> AgentGraph:
    """Generate and return a dictionary of agents from the agent
    information CSV file. Each agent is added to the database and
    their respective profiles are updated.

    Args:
        agent_info_path (str): The file path to the agent information CSV file.
        channel (Channel): Information channel.
        num_agents (int): Number of agents.
        model_random_seed (int): Random seed to randomly assign model to
            each agent. (default: 42)
        cfgs (list, optional): List of configuration. (default: `None`)

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
    assert len(model_types) == len(agent_info), \
        (f"Mismatch between the number of agents "
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

    agent_graph = AgentGraph()

    async def setup_agent(agent_id: int):
        profile = {
            'nodes': [],
            'edges': [],
            'other_info': {},
        }
        profile['other_info']['user_profile'] = agent_info['user_char'][
            agent_id]
        profile['other_info']['mbti'] = random.choice(mbti_types)
        profile['other_info']['activity_level_frequency'] = ast.literal_eval(
            agent_info["activity_level_frequency"][agent_id])
        profile['other_info']['active_threshold'] = prob_list[agent_id]

        user_info = UserInfo(
            name=agent_info['username'][agent_id],
            description=agent_info['description'][agent_id],
            profile=profile,
        )

        model_type: ModelType = model_types[agent_id]

        agent = SocialAgent(
            agent_id=agent_id,
            user_info=user_info,
            twitter_channel=twitter_channel,
            inference_channel=inference_channel,
            model_type=model_type,
            agent_graph=agent_graph,
        )

        agent_graph.add_agent(agent)

        await agent.env.action.sign_up(
            agent_info["username"][agent_id],
            agent_info["name"][agent_id],
            agent_info["description"][agent_id],
        )

        # following_id_list = ast.literal_eval(
        #         agent_info["following_agentid_list"][agent_id])
        following_id = random.randint(0, len(agent_info))
        await agent.env.action.follow(following_id + 1)
        agent_graph.add_edge(agent_id, following_id)

        previous_posts = ast.literal_eval(
                agent_info['previous_tweets'][agent_id])
        if len(previous_posts) != 0:
            post_tasks = [
                agent.env.action.create_post(post) for post in previous_posts
            ]
            await asyncio.gather(*post_tasks)

    tasks = [setup_agent(i) for i in range(len(agent_info))]
    await asyncio.gather(*tasks)

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
            profile={'other_info': {
                'user_profile': 'None'
            }},
        )
        # controllable的agent_id全都在llm agent的agent_id的前面
        agent = SocialAgent(i, user_info, channel, agent_graph=agent_graph)
        # Add agent to the agent graph
        agent_graph.add_agent(agent)

        username = input(f"Please input username for agent {i}: ")
        name = input(f"Please input name for agent {i}: ")
        bio = input(f"Please input bio for agent {i}: ")

        response = await agent.env.action.sign_up(username, name, bio)
        user_id = response['user_id']
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
            profile={'other_info': {
                'user_profile': 'None'
            }},
        )
        # controllable的agent_id全都在llm agent的agent_id的前面
        agent = SocialAgent(i, user_info, channel, agent_graph=agent_graph)
        # Add agent to the agent graph
        agent_graph.add_agent(agent)

        response = await agent.env.action.sign_up(
            user_name='momo',
            name='momo',
            bio='None.',
        )
        user_id = response['user_id']
        agent_user_id_mapping[i] = user_id

    return agent_graph, agent_user_id_mapping


async def generate_reddit_agents(
    agent_info_path: str,
    twitter_channel: Channel,
    inference_channel: Channel,
    agent_graph: AgentGraph | None = AgentGraph,
    agent_user_id_mapping: dict[int, int]
    | None = None,
) -> AgentGraph:
    if agent_user_id_mapping is None:
        agent_user_id_mapping = {}
    if agent_graph is None:
        agent_graph = AgentGraph()

    control_user_num = agent_graph.get_num_nodes()

    with open(agent_info_path, 'r') as file:
        agent_info = json.load(file)

    for i in range(len(agent_info)):
        # Instantiate an agent
        profile = {
            'nodes': [],  # Relationships with other agents
            'edges': [],  # Relationship details
            'other_info': {},
        }

        # Update agent profile with additional information
        profile['other_info']['user_profile'] = agent_info[i]['bio']

        user_info = UserInfo(name=agent_info[i]['nickname'],
                             description=agent_info[i]['description'],
                             profile=profile)

        # controllable的agent_id全都在llm agent的agent_id的前面
        agent = SocialAgent(i + control_user_num, user_info, twitter_channel, inference_channel, agent_graph=agent_graph)

        # Add agent to the agent graph
        agent_graph.add_agent(agent)

        # Sign up agent and add their information to the database
        # print(f"Signing up agent {agent_info['username'][i]}...")
        response = await agent.env.action.sign_up(
            agent_info[i]['nickname'],
            agent_info[i]['nickname'],
            agent_info[i]['description'],
        )
        user_id = response['user_id']
        agent_user_id_mapping[i + control_user_num] = user_id

    return agent_graph
