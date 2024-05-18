# File: ./test/test_infra/test_agent_generator.py
import ast
import random
import asyncio
import unittest
from unittest.mock import patch, MagicMock
import networkx as nx
import pandas as pd
from social_agent.agents_generator import generate_agents
from twitter.channel import Twitter_Channel
from twitter.twitter import Twitter


async def running():
    test_db_filepath = "./db/test.db"
    agent_info_path = "./test/test_data/user_all_id.csv"
    channel = Twitter_Channel()
    infra = Twitter(test_db_filepath, channel)
    task = asyncio.create_task(infra.running())
    agent_graph = await generate_agents(agent_info_path, channel)
    print(agent_graph)

    


def test_agent_generator():
    my_agent_graph = asyncio.run(running())
