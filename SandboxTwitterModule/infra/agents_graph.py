# File: SandboxTwitterModule/infra/agents_graph.py
import asyncio
import networkx as nx
import random
from SandboxTwitterModule.infra.Agent.twitterUserAgent import TwitterUserAgent
from SandboxTwitterModule.core.decorators import function_call_logger

class homoAgentsGraph:
    def __init__(self, channel):
        self.channel = channel
        self.graph = nx.DiGraph()
        self.initialize_agents()

    @function_call_logger
    def initialize_agents(self):
        agent_count = self.get_agent_count()
        for agent_id in range(agent_count):
            name = self.id_to_name(agent_id)
            agent = TwitterUserAgent(agent_id, name, self.channel)
            self.graph.add_node(agent_id, agent=agent)
            # Optionally add random connections (e.g., follow relationships) here

    @function_call_logger
    def get_agent_count(self):
        # Simplified to return a fixed number for illustration
        return 10  # Assume there are 100 agents

    @function_call_logger
    def simulate_agent_activities(self, num_activities=10):
        print("Simulating agent activities...")
        activities = ['create_tweet', 'sign_up']
        for _ in range(num_activities):
            for node_id, node_data in self.graph.nodes(data=True):
                agent = node_data['agent']
                action = random.choice(activities)
                if action == 'create_tweet':
                    content = f"Hello from agent {node_id}"
                    asyncio.create_task(agent.action_create_tweet(content))
                elif action == 'sign_up':
                    user_name = f"user_{node_id}"
                    name = f"Agent {node_id}"
                    bio = "This is a bio."
                    asyncio.create_task(agent.action_sign_up(user_name, name, bio))

    @function_call_logger
    def id_to_name(self, agent_id):
        name = ''
        while len(name) < 3:
            remainder = agent_id % 26
            name = chr(97 + remainder) + name
            agent_id //= 26
        return name
