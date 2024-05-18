# File: ./test/test_infra/test_agent_generator.py
import ast
import random
import unittest
from unittest.mock import patch, MagicMock
import networkx as nx
import pandas as pd
from SandboxTwitterModule.infra.agents_generator import AgentsGenerator



agent_info_path = "../../data/user_all_id.csv"
agent_generator = AgentsGenerator(agent_info_path)
agent_dict = agent_generator.generate_agents()
print(agent_dict)