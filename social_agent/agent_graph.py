from __future__ import annotations

import networkx as nx

from social_agent.agent import TwitterUserAgent


class AgentGraph:

    def __init__(self):
        self.graph = nx.DiGraph()

    async def add_agent(self, agent: TwitterUserAgent):
        self.graph.add_node(agent.agent_id, agent=agent)

    async def add_edge(self, agent1_id: int, agent2_id: int):
        self.graph.add_edge(agent1_id, agent2_id)

    async def remove_agent(self, agent: TwitterUserAgent):
        self.graph.remove_node(agent.agent_id)

    def get_agent(self, agent_id: int) -> TwitterUserAgent:
        return self.graph.nodes[agent_id]['agent']

    def get_agents(self):
        return self.graph.nodes.data()

    def get_edges(self):
        return self.graph.edges.data()

    @property
    def num_agents(self):
        return self.graph.number_of_nodes()
