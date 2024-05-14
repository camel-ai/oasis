
import ast
import random
import networkx as nx
import pandas as pd

from .twitterUserAgent import TwitterUserAgent


async def generate_agents(agent_info_path, twitter_channel):
    """
    Generates and returns a dictionary of agents from the agent information CSV file.
    Each agent is added to the database and their respective profiles are updated.

    Args:
        agent_info_path (str): The file path to the agent information CSV file.

    Returns:
        dict: A dictionary of agent IDs mapped to their respective agent class instances.
    """
    mbti_types = ["INTJ", "ENTP", "INFJ", "ENFP"]
    activities = ["High", "Medium", "Low"] 
    agent_info = pd.read_csv(agent_info_path)
    agents_dict = {}
    for i in range(len(agent_info)):
        # Instantiate an agent
        profile = {'nodes': [],  # Relationships with other agents
        'edges': [],  # Relationship details
        'other_info': {},
        }
        # Update agent profile with additional information
        profile['other_info']['user_profile'] = agent_info['user_char'][i]
        # Randomly assign an MBTI type (temporary, subject to change)
        profile['other_info']['mbti'] = random.choice(mbti_types)
        # Randomly assign an activity level (temporary, subject to change)
        profile['other_info']['activity_level'] = random.choice(activities)
            
        agent = TwitterUserAgent(
            i, 
            agent_info['username'][i], 
            agent_info['description'][i],
            profile,
            twitter_channel
        )
            
        # Sign up agent and add their information to the database
        print(f"Signing up agent {agent_info['username'][i]}...")
        await agent.action_sign_up(
            agent_info['username'][i], 
            agent_info['name'][i], 
            agent_info['description'][i],
        )
            
        # Add user relationships if any
        if agent_info['following_agentid_list'][i] != "0":
            following_id_list = ast.literal_eval(agent_info['following_agentid_list'][i])
            for _agent_id in following_id_list:
                await agent.action_follow(_agent_id)

        if len(agent_info['previous_tweets']) != 0:
            previous_tweets = ast.literal_eval(agent_info['previous_tweets'][i])
            for tweet in previous_tweets:
                await agent.action_create_tweet(tweet)

        agents_dict[i] = agent

    return agents_dict
