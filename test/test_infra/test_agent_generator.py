# File: ./test/test_infra/test_agent_generator.py
import ast
import random
import unittest
from unittest.mock import patch, MagicMock
import networkx as nx
import pandas as pd
from social_agent.agents_generator import generate_agents



agent_info_path = "./test/test_data/user_all_id.csv"
agent_dict = generate_agents(agent_info_path, channel)
print(agent_dict)

