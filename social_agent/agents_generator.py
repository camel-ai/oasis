from __future__ import annotations

import ast
import random

import pandas as pd
from camel.configs import FunctionCallingConfig
from camel.types import ModelType

from social_agent.agent_action import ActionSpace
from social_agent.agent_graph import AgentGraph
from twitter.channel import TwitterChannel
from twitter.config import UserInfo

from .agent import TwitterUserAgent


async def generate_agents(
    agent_info_path: str,
    channel: TwitterChannel,
    model_type: ModelType = ModelType.GPT_3_5_TURBO,
):
    """Generates and returns a dictionary of agents from the agent
    information CSV file. Each agent is added to the database and
    their respective profiles are updated.

    Args:
        agent_info_path (str): The file path to the agent information CSV file.
        channel (TwitterChannel): Twitter channel.
        model_type (ModelType): Model type to use.

    Returns:
        dict: A dictionary of agent IDs mapped to their respective agent
            class instances.
    """
    mbti_types = ["INTJ", "ENTP", "INFJ", "ENFP"]
    activities = ["High", "Medium", "Low"]
    agent_info = pd.read_csv(agent_info_path)

    agent_graph = AgentGraph()
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
        profile['other_info']['activity_level'] = random.choice(activities)

        user_info = UserInfo(name=agent_info['username'][i],
                             description=agent_info['description'][i],
                             profile=profile)

        action_space = ActionSpace(agent_id=i, channel=channel)
        model_config = FunctionCallingConfig.from_openai_function_list(
            function_list=action_space.get_openai_function_list(),
            kwargs=dict(temperature=0.0),
        )
        agent = TwitterUserAgent(
            agent_id=i,
            user_info=user_info,
            action_space=action_space,
            channel=channel,
            model_type=model_type,
            model_config=model_config,
        )

        # Add agent to the agent graph
        await agent_graph.add_agent(agent)

        # Sign up agent and add their information to the database
        # print(f"Signing up agent {agent_info['username'][i]}...")
        await agent.action_space.action_sign_up(agent_info['username'][i],
                                                agent_info['name'][i],
                                                agent_info['description'][i])

        # Add user relationships if any
        if agent_info['following_agentid_list'][i] != "0":
            following_id_list = ast.literal_eval(
                agent_info['following_agentid_list'][i])
            for _agent_id in following_id_list:
                await agent.action_space.action_follow(_agent_id)
                await agent_graph.add_edge(i, _agent_id)

        if len(agent_info['previous_tweets']) != 0:
            previous_tweets = ast.literal_eval(
                agent_info['previous_tweets'][i])
            for tweet in previous_tweets:
                await agent.action_space.action_create_tweet(tweet)

    return agent_graph
