from __future__ import annotations

import ast
import asyncio
import json
import random

from camel.types.enums import ModelType
import numpy as np
import pandas as pd

from social_simulation.social_agent.agent import SocialAgent
from social_simulation.social_agent.agent_graph import AgentGraph
from social_simulation.social_platform.channel import Channel
from social_simulation.social_platform.config import UserInfo


model_types = [
    ModelType.LLAMA_3,
    ModelType.INTERNLM,
    ModelType.QWEN
]


async def generate_agents(
    agent_info_path: str, 
    channel: Channel, 
    model_configs: dict = None
    ) -> AgentGraph:
    """Generates and returns a dictionary of agents from the agent
    information CSV file. Each agent is added to the database and
    their respective profiles are updated.

    Args:
        agent_info_path (str): The file path to the agent information CSV file.
        channel (Channel): Information channel.
        model_configs(dict): Configuration file for backend model of each agent.

    Returns:
        dict: A dictionary of agent IDs mapped to their respective agent
            class instances.
    """

    cfgs = model_configs.get("cfgs")
    model_indices = []
    cfgs = model_configs.get("cfgs")
    for i, cfg in enumerate(cfgs):
        model_indices.extend([i] * cfg['num'])
    random.shuffle(model_indices)

    mbti_types = ["INTJ", "ENTP", "INFJ", "ENFP"]
    agent_info = pd.read_csv(agent_info_path)

    assert len(model_indices) == len(agent_info), \
        f"Mismatch between the number of agents and the number of models, with {len(agent_info)} agents and {len(model_indices)} models."

    freq = list(agent_info["activity_level_frequency"])
    all_freq = np.array([ast.literal_eval(fre) for fre in temp])
    normalized_prob =  all_freq / np.max(all_freq)
    normalized_prob[normalized_prob < 0.6] += 0.1
    normalized_prob = np.round(normalized_prob, 2)
    prob_list = normalized_prob.tolist()

    agent_graph = AgentGraph()

    async def process_agent(i):
        profile = {
            'nodes': [],  
            'edges': [],  
            'other_info': {},
        }
        profile['other_info']['user_profile'] = agent_info['user_char'][i]
        profile['other_info']['mbti'] = random.choice(mbti_types)
        profile['other_info']['activity_level_frequency'] = ast.literal_eval(
            agent_info["activity_level_frequency"][i])
        profile['other_info']['active_threshold'] = prob_list[i]

        user_info = UserInfo(name=agent_info['username'][i],
                             description=agent_info['description'][i],
                             profile=profile)

        model_config = cfgs[model_indices[i]]
        agent = SocialAgent(i, 
                            user_info, 
                            channel, 
                            model_config['model_path'], 
                            model_config['server_url'], 
                            model_config['stop_tokens'],
                            model_types[model_indices[i]]
                                 )

        await agent_graph.add_agent(agent)

        await agent.env.action.sign_up(agent_info['username'][i],
                                       agent_info['name'][i],
                                       agent_info['description'][i])

        if agent_info['following_agentid_list'][i] != "0":
            following_id_list = ast.literal_eval(
                agent_info['following_agentid_list'][i])
            follow_tasks = [
                agent.env.action.follow(_agent_id + 1)
                for _agent_id in following_id_list
            ]
            await asyncio.gather(*follow_tasks)
            for _agent_id in following_id_list:
                await agent_graph.add_edge(i, _agent_id)

        if len(agent_info['previous_tweets']) != 0:
            previous_tweets = ast.literal_eval(
                agent_info['previous_tweets'][i])

            tweet_tasks = [
                agent.env.action.create_tweet(tweet)
                for tweet in previous_tweets
            ]
            await asyncio.gather(*tweet_tasks)

    tasks = [process_agent(i) for i in range(len(agent_info))]
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
        agent = SocialAgent(i, user_info, channel)
        # Add agent to the agent graph
        await agent_graph.add_agent(agent)

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
                await agent_graph.add_edge(i, j)
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
        agent = SocialAgent(i, user_info, channel)
        # Add agent to the agent graph
        await agent_graph.add_agent(agent)

        response = await agent.env.action.sign_up(user_name='momo',
                                                  name='momo',
                                                  bio='None.')
        user_id = response['user_id']
        agent_user_id_mapping[i] = user_id

    return agent_graph, agent_user_id_mapping


async def generate_reddit_agents(
    agent_info_path: str,
    channel: Channel,
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
        agent = SocialAgent(i + control_user_num, user_info, channel)

        # Add agent to the agent graph
        await agent_graph.add_agent(agent)

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
