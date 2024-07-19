from __future__ import annotations

import igraph as ig

from social_simulation.social_agent.agent import SocialAgent


class AgentGraph:

    def __init__(self):
        self.graph = ig.Graph(directed=True)
        self.agent_mappings: dict[int, SocialAgent] = {}

    def add_agent(self, agent: SocialAgent):
        self.graph.add_vertex(agent.agent_id)
        self.agent_mappings[agent.agent_id] = agent

    def add_edge(self, agent_id_0: int, agent_id_1: int):
        self.graph.add_edge(agent_id_0, agent_id_1)

    def remove_agent(self, agent: SocialAgent):
        self.graph.delete_vertices(agent.agent_id)
        del self.agent_mappings[agent.agent_id]

    def remove_edge(self, agent_id_0: int, agent_id_1: int):
        if self.graph.are_connected(agent_id_0, agent_id_1):
            self.graph.delete_edges([(agent_id_0, agent_id_1)])

    def get_agent(self, agent_id: int) -> SocialAgent:
        return self.agent_mappings[agent_id]

    def get_agents(self) -> list[tuple[int, SocialAgent]]:
        return [(node.index, self.agent_mappings[node.index])
                for node in self.graph.vs]

    def get_edges(self) -> list[tuple[int, int]]:
        return [(edge.source, edge.target) for edge in self.graph.es]

    def get_num_nodes(self) -> int:
        return self.graph.vcount()

    def get_num_edges(self) -> int:
        return self.graph.ecount()

    def visualize(
        self,
        path: str,
        vertex_size: int = 20,
        edge_arrow_size: float = 0.5,
        with_labels: bool = True,
        vertex_color: str = "#f74f1b",
        vertex_frame_width: int = 2,
        width: int = 1000,
        height: int = 1000,
    ):
        layout = self.graph.layout("auto")
        if with_labels:
            labels = [node_id for node_id, _ in self.get_agents()]
        else:
            labels = None
        ig.plot(
            self.graph,
            target=path,
            layout=layout,
            vertex_label=labels,
            vertex_size=vertex_size,
            vertex_color=vertex_color,
            edge_arrow_size=edge_arrow_size,
            vertex_frame_width=vertex_frame_width,
            bbox=(width, height),
        )
