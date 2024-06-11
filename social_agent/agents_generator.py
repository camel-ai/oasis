import ast
import random
from typing import Dict

import pandas as pd

from social_agent.agent_graph import AgentGraph
from twitter.config import UserInfo

from .agent import TwitterUserAgent


async def generate_agents(
        agent_info_path, twitter_channel, agent_graph=AgentGraph(),
        agent_user_id_mapping: Dict[int, int] = {}):
    """Generates and returns a dictionary of agents from the agent
    information CSV file. Each agent is added to the database and
    their respective profiles are updated.

    Args:
        agent_info_path (str): The file path to the agent information CSV file.

    Returns:
        dict: A dictionary of agent IDs mapped to their respective agent
            class instances.
    """
    # 目前已有的都是controllable agent
    # agent_graph.get_acount() == len(controllable_user_id)
    control_user_num = agent_graph.get_acount()

    mbti_types = ["INTJ", "ENTP", "INFJ", "ENFP"]
    # activities = ["High", "Medium", "Low"]
    agent_info = pd.read_csv(agent_info_path)

    # active state to active prob dict
    threshold_dict = {"off_line": 0.1, "busy": 0.3, "normal": 0.6, "active": 1}

    for i in range(len(agent_info)):
        # Instantiate an agent
        profile = {
            'nodes': [],  # Relationships with other agents
            'edges': [],  # Relationship details
            'other_info': {},
        }
        # Update agent profile with additional information
        profile['other_info']['user_profile'] = agent_info['user_char'][i]
        # Randomly assign an MBTI type (temporary, subject to change)
        profile['other_info']['mbti'] = random.choice(mbti_types)
        # Randomly assign an activity level (temporary, subject to change)
        profile['other_info']['activity_level'] = ast.literal_eval(
            agent_info["activity_level"][i])
        profile['other_info']['activity_level_frequency'] = ast.literal_eval(
            agent_info["activity_level_frequency"][i])
        profile['other_info']['active_threshold'] = [
            threshold_dict[ac_lv]
            for ac_lv in profile['other_info']['activity_level']
        ]

        user_info = UserInfo(name=agent_info['username'][i],
                             description=agent_info['description'][i],
                             profile=profile)

        # controllable的agent_id全都在llm agent的agent_id的前面
        agent = TwitterUserAgent(
            i+control_user_num, user_info, twitter_channel)

        # Add agent to the agent graph
        await agent_graph.add_agent(agent)

        # Sign up agent and add their information to the database
        # print(f"Signing up agent {agent_info['username'][i]}...")
        response = await agent.env.twitter_action.action_sign_up(
            agent_info['username'][i], agent_info['name'][i],
            agent_info['description'][i])
        user_id = response['user_id']
        agent_user_id_mapping[i+control_user_num] = user_id

    for i in range(len(agent_info)):
        agent = agent_graph.get_agent(i)
        # Add user relationships if any
        if agent_info['following_agentid_list'][i] != "0":
            following_id_list = ast.literal_eval(
                agent_info['following_agentid_list'][i])
            for _agent_id in following_id_list:
                # 这里action_follow接受的是user_id，不是agent id，所以会出现关注错误的问题
                # 由于二者只差一个1，所以加个1就可以了
                user_id = agent_user_id_mapping[_agent_id+control_user_num]
                await agent.env.twitter_action.action_follow(user_id)
                await agent_graph.add_edge(
                    agent.agent_id, _agent_id+control_user_num)
            for _control_agent_id in range(control_user_num):
                user_id = agent_user_id_mapping[_control_agent_id]
                await agent.env.twitter_action.action_follow(user_id)
                await agent_graph.add_edge(
                    agent.agent_id, _control_agent_id)

        if len(agent_info['previous_tweets']) != 0:
            previous_tweets = ast.literal_eval(
                agent_info['previous_tweets'][i])
            for tweet in previous_tweets:
                await agent.env.twitter_action.action_create_tweet(tweet)
    print('agent_user_id_mapping:', agent_user_id_mapping)
    return agent_graph


async def generate_controllable_agents(twitter_channel, control_user_num: int):
    agent_graph = AgentGraph()
    agent_user_id_mapping = {}
    for i in range(control_user_num):
        # peofile实际上用不到，写成'None'防止报错
        user_info = UserInfo(
            is_controllable=True,
            profile={'other_info': {'user_profile': 'None'}})
        # controllable的agent_id全都在llm agent的agent_id的前面
        agent = TwitterUserAgent(i, user_info, twitter_channel)
        # Add agent to the agent graph
        await agent_graph.add_agent(agent)

        username = input(f"Please input username for agent {i}: ")
        name = input(f"Please input name for agent {i}: ")
        bio = input(f"Please input bio for agent {i}: ")

        response = await agent.env.twitter_action.action_sign_up(
            username, name, bio)
        user_id = response['user_id']
        agent_user_id_mapping[i] = user_id

    for i in range(control_user_num):
        for j in range(control_user_num):
            agent = agent_graph.get_agent(i)
            # controllable agent互相也全部关注
            if i != j:
                user_id = agent_user_id_mapping[j]
                await agent.env.twitter_action.action_follow(user_id)
                await agent_graph.add_edge(i, j)
    return agent_graph, agent_user_id_mapping
