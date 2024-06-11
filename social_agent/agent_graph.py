import networkx as nx


class AgentGraph:

    def __init__(self):
        self.graph = nx.DiGraph()

    async def add_agent(self, agent):
        self.graph.add_node(agent.agent_id, agent=agent)

    async def add_edge(self, agent1_id, agent2_id):
        self.graph.add_edge(agent1_id, agent2_id)

    async def remove_agent(self, agent):
        self.graph.remove_node(agent.agent_id)

    def get_agent(self, agent_id):
        return self.graph.nodes[agent_id]['agent']

    def get_agents(self):
        return self.graph.nodes.data()

    def get_edges(self):
        return self.graph.edges.data()

    def get_incoming_nodes(self):
        return self.graph.in_edges()

    def get_outgoing_nodes(self):
        return self.graph.out_edges()

    def get_num_nodes(self):
        return self.graph.number_of_nodes()
