# =========== Copyright 2023 @ CAMEL-AI.org. All Rights Reserved. ===========
# Licensed under the Apache License, Version 2.0 (the “License”);
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an “AS IS” BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =========== Copyright 2023 @ CAMEL-AI.org. All Rights Reserved. ===========
from __future__ import annotations

from typing import Any, Literal

import igraph as ig
from neo4j import GraphDatabase

from oasis.social_agent.agent import SocialAgent
from oasis.social_platform.config import Neo4jConfig


class Neo4jHandler:
    r"""
    This class provides methods to create, delete, and manage 
    agents (nodes) and their relationships (edges) in a Neo4j 
    graph database.
    """

    def __init__(self, nei4j_config: Neo4jConfig):
        r"""Initialize the Neo4jHandler with the given configuration.

        Args:
            nei4j_config (Neo4jConfig): Configuration for Neo4j connection.
        """
        self.driver = GraphDatabase.driver(
            nei4j_config.uri,
            auth=(nei4j_config.username, nei4j_config.password),
        )
        self.driver.verify_connectivity()

    def close(self):
        r"""Close the Neo4j driver connection.
        """
        self.driver.close()

    def create_agent(self, agent_id: int):
        r"""
        Create a new agent node in the Neo4j graph.

        Args:
            agent_id (int): The ID of the agent.
        """
        with self.driver.session() as session:
            session.write_transaction(self._create_and_return_agent, agent_id)

    def delete_agent(self, agent_id: int):
        r"""
        Delete an agent node and all its relationships from the graph.

        Args:
            agent_id (int): ID of the agent to be deleted.
        """
        with self.driver.session() as session:
            session.write_transaction(
                self._delete_agent_and_relationships,
                agent_id,
            )

    def get_number_of_nodes(self) -> int:
        r"""
        Get the total number of agent nodes in the graph.

        Returns:
            int: Count of nodes in the graph.
        """
        with self.driver.session() as session:
            return session.read_transaction(self._get_number_of_nodes)

    def get_number_of_edges(self) -> int:
        r"""
        Get the total number of relationships (edges) in the graph.

        Returns:
            int: Count of edges in the graph.
        """
        with self.driver.session() as session:
            return session.read_transaction(self._get_number_of_edges)

    def add_edge(self, src_agent_id: int, dst_agent_id: int):
        r"""
        Add a directed FOLLOW relationship between two agents.

        Args:
            src_agent_id (int): Source agent ID.
            dst_agent_id (int): Destination agent ID.
        """
        with self.driver.session() as session:
            session.write_transaction(
                self._add_and_return_edge,
                src_agent_id,
                dst_agent_id,
            )

    def remove_edge(self, src_agent_id: int, dst_agent_id: int):
        r"""
        Remove a FOLLOW relationship between two agents.

        Args:
            src_agent_id (int): Source agent ID.
            dst_agent_id (int): Destination agent ID.
        """
        with self.driver.session() as session:
            session.write_transaction(
                self._remove_and_return_edge,
                src_agent_id,
                dst_agent_id,
            )

    def get_all_nodes(self) -> list[int]:
        r"""
        Get a list of all agent IDs in the graph.

        Returns:
            list[int]: List of agent IDs.
        """
        with self.driver.session() as session:
            return session.read_transaction(self._get_all_nodes)

    def get_all_edges(self) -> list[tuple[int, int]]:
        r"""
        Get all FOLLOW relationships in the graph.

        Returns:
            list[tuple[int, int]]: List of (source, destination) 
            agent ID pairs.
        """
        with self.driver.session() as session:
            return session.read_transaction(self._get_all_edges)

    def reset_graph(self):
        r"""Clear the entire graph by deleting all nodes and relationships.
        """
        with self.driver.session() as session:
            session.write_transaction(self._reset_graph)

    @staticmethod
    def _create_and_return_agent(tx: Any, agent_id: int):
        query = """
        CREATE (a:Agent {id: $agent_id})
        RETURN a
        """
        result = tx.run(query, agent_id=agent_id)
        return result.single()

    @staticmethod
    def _delete_agent_and_relationships(tx: Any, agent_id: int):
        query = """
        MATCH (a:Agent {id: $agent_id})
        DETACH DELETE a
        RETURN count(a) AS deleted
        """
        result = tx.run(query, agent_id=agent_id)
        return result.single()

    @staticmethod
    def _add_and_return_edge(tx: Any, src_agent_id: int, dst_agent_id: int):
        query = """
        MATCH (a:Agent {id: $src_agent_id}), (b:Agent {id: $dst_agent_id})
        CREATE (a)-[r:FOLLOW]->(b)
        RETURN r
        """
        result = tx.run(query,
                        src_agent_id=src_agent_id,
                        dst_agent_id=dst_agent_id)
        return result.single()

    @staticmethod
    def _remove_and_return_edge(tx: Any, src_agent_id: int, dst_agent_id: int):
        query = """
        MATCH (a:Agent {id: $src_agent_id})
        MATCH (b:Agent {id: $dst_agent_id})
        MATCH (a)-[r:FOLLOW]->(b)
        DELETE r
        RETURN count(r) AS deleted
        """
        result = tx.run(query,
                        src_agent_id=src_agent_id,
                        dst_agent_id=dst_agent_id)
        return result.single()

    @staticmethod
    def _get_number_of_nodes(tx: Any) -> int:
        query = """
        MATCH (n)
        RETURN count(n) AS num_nodes
        """
        result = tx.run(query)
        return result.single()["num_nodes"]

    @staticmethod
    def _get_number_of_edges(tx: Any) -> int:
        query = """
        MATCH ()-[r]->()
        RETURN count(r) AS num_edges
        """
        result = tx.run(query)
        return result.single()["num_edges"]

    @staticmethod
    def _get_all_nodes(tx: Any) -> list[int]:
        query = """
        MATCH (a:Agent)
        RETURN a.id AS agent_id
        """
        result = tx.run(query)
        return [record["agent_id"] for record in result]

    @staticmethod
    def _get_all_edges(tx: Any) -> list[tuple[int, int]]:
        query = """
        MATCH (a:Agent)-[r:FOLLOW]->(b:Agent)
        RETURN a.id AS src_agent_id, b.id AS dst_agent_id
        """
        result = tx.run(query)
        return [(record["src_agent_id"], record["dst_agent_id"])
                for record in result]

    @staticmethod
    def _reset_graph(tx: Any):
        query = """
        MATCH (n)
        DETACH DELETE n
        """
        tx.run(query)


class AgentGraph:
    r"""AgentGraph class to manage the social graph of agents."""

    def __init__(
        self,
        backend: Literal["igraph", "neo4j"] = "igraph",
        neo4j_config: Neo4jConfig | None = None,
    ):
        r"""
        Initialize the agent graph.

        Args:
        backend (Literal["igraph", "neo4j"]): The graph backend to use. 
        Default to "igraph".
        neo4j_config (Neo4jConfig | None): Configuration for Neo4j backend. 
        Required if backend is "neo4j".
        """
        self.backend = backend
        if self.backend == "igraph":
            self.graph = ig.Graph(directed=True)
        else:
            assert neo4j_config is not None
            assert neo4j_config.is_valid()
            self.graph = Neo4jHandler(neo4j_config)
        self.agent_mappings: dict[int, SocialAgent] = {}

    def reset(self):
        r"""Reset the graph by removing all nodes and edges.
        """
        if self.backend == "igraph":
            self.graph = ig.Graph(directed=True)
        else:
            self.graph.reset_graph()
        self.agent_mappings: dict[int, SocialAgent] = {}

    def add_agent(self, agent: SocialAgent):
        r"""
        Add a new social agent to the graph.

        Args:
            agent (SocialAgent): The agent to be added to the graph.
        """
        if self.backend == "igraph":
            self.graph.add_vertex(agent.social_agent_id)
        else:
            self.graph.create_agent(agent.social_agent_id)
        self.agent_mappings[agent.social_agent_id] = agent

    def add_edge(self, agent_id_0: int, agent_id_1: int):
        r"""
        Add a directed edge between two agents.

        Args:
            agent_id_0 (int): Source agent ID.
            agent_id_1 (int): Destination agent ID.
        """
        try:
            self.graph.add_edge(agent_id_0, agent_id_1)
        except Exception:
            pass

    def remove_agent(self, agent: SocialAgent):
        r"""
        Remove a social agent from the graph.

        Args:
            agent (SocialAgent): The agent to be deleted from the graph.
        """
        if self.backend == "igraph":
            self.graph.delete_vertices(agent.social_agent_id)
        else:
            self.graph.delete_agent(agent.social_agent_id)
        del self.agent_mappings[agent.social_agent_id]

    def remove_edge(self, agent_id_0: int, agent_id_1: int):
        r"""
        Remove a directed edge between two agents.

        Args:
            agent_id_0 (int): Source agent ID.
            agent_id_1 (int): Destination agent ID.
        """
        if self.backend == "igraph":
            if self.graph.are_connected(agent_id_0, agent_id_1):
                self.graph.delete_edges([(agent_id_0, agent_id_1)])
        else:
            self.graph.remove_edge(agent_id_0, agent_id_1)

    def get_agent(self, agent_id: int) -> SocialAgent:
        r"""
        Get a social agent by its ID.

        Args:
            agent_id (int): The ID of the agent to retrive.

        Returns:
            SocialAgent: The requested social agent.
        """
        return self.agent_mappings[agent_id]

    def get_agents(
            self,
            agent_ids: list[int] = None) -> list[tuple[int, SocialAgent]]:
        r"""
        Get specific agents by their IDs.

        Args:
            agent_ids (list[int], optional): List of agent IDs to retrieve.
                If None, returns all agents.

        Returns:
            list[tuple[int, SocialAgent]]: List of (agent_id, agent) tuples.
        """
        if agent_ids:
            return [(agent_id, self.get_agent(agent_id))
                    for agent_id in agent_ids]
        if self.backend == "igraph":
            return [(node.index, self.agent_mappings[node.index])
                    for node in self.graph.vs]
        else:
            return [(agent_id, self.agent_mappings[agent_id])
                    for agent_id in self.graph.get_all_nodes()]

    def get_edges(self) -> list[tuple[int, int]]:
        r"""
        Get all edges in the graph.

        Returns:
            list[tuple[int, int]]: List of (source, destination) 
            agent ID pairs.
        """
        if self.backend == "igraph":
            return [(edge.source, edge.target) for edge in self.graph.es]
        else:
            return self.graph.get_all_edges()

    def get_num_nodes(self) -> int:
        r"""
        Get the number of nodes in the graph.

        Returns:
            int: Number of nodes.
        """
        if self.backend == "igraph":
            return self.graph.vcount()
        else:
            return self.graph.get_number_of_nodes()

    def get_num_edges(self) -> int:
        r"""
        Get the number of edges in the graph.

        Returns:
            int: Number of edges.
        """
        if self.backend == "igraph":
            return self.graph.ecount()
        else:
            return self.graph.get_number_of_edges()

    def close(self) -> None:
        r"""Close the graph connection. 
        """
        if self.backend == "neo4j":
            self.graph.close()

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
        r"""
        Visualize the graph and save it to a file only when 
        the backend is igraph.

        Args:
            path (str): Path to save the visualization.
            vertex_size (int, optional): Size of vertices.
            edge_arrow_size (float, optional): Size of edge arrows.
            with_labels (bool, optional): Whether to show vertex labels. 
            vertex_color (str, optional): Color of vertices. 
            vertex_frame_width (int, optional): Width of vertex frames. 
            width (int, optional): Width of the image. 
            height (int, optional): Height of the image.
        """
        if self.backend == "neo4j":
            raise ValueError("Neo4j backend does not support visualization.")
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
