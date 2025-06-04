import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
import networkx as nx

class TransportNode:
    """Represents a station or node in the transportation network"""
    
    def __init__(self, node_id: str, x: float, y: float, name: Optional[str] = None, attributes: Dict = None):
        """
        Initialize a transport node
        
        Args:
            node_id: Unique identifier for the node
            x: x-coordinate of the node
            y: y-coordinate of the node
            name: Human-readable name of the node (station name)
            attributes: Additional attributes for the node
        """
        self.node_id = node_id
        self.x = x
        self.y = y
        self.name = name if name else node_id
        self.attributes = attributes if attributes else {}
    
    def __repr__(self):
        return f"TransportNode(id={self.node_id}, name={self.name}, pos=({self.x}, {self.y}))"


class TransportLink:
    """Represents a route or link between nodes in the transportation network"""
    
    def __init__(self, link_id: str, from_node: str, to_node: str, 
                 length: float, capacity: float, free_speed: float,
                 attributes: Dict = None):
        """
        Initialize a transport link
        
        Args:
            link_id: Unique identifier for the link
            from_node: ID of the origin node
            to_node: ID of the destination node
            length: Length of the link in meters
            capacity: Capacity of the link (vehicles/hour)
            free_speed: Free flow speed on the link (m/s)
            attributes: Additional attributes for the link
        """
        self.link_id = link_id
        self.from_node = from_node
        self.to_node = to_node
        self.length = length
        self.capacity = capacity
        self.free_speed = free_speed
        self.attributes = attributes if attributes else {}
        
        # Historical weights by time period
        self.weights_history = {}
        
    def add_weight_data(self, time_period: str, weight: float):
        """Add historical weight data for a specific time period"""
        self.weights_history[time_period] = weight
    
    def get_weight(self, time_period: str) -> float:
        """Get weight for a specific time period"""
        return self.weights_history.get(time_period, 1.0)
    
    def __repr__(self):
        return f"TransportLink(id={self.link_id}, {self.from_node}->{self.to_node}, len={self.length}m)"


class TransportNetwork:
    """Represents a complete transportation network with nodes and links"""
    
    def __init__(self, name: str = "Transport Network"):
        """
        Initialize an empty transport network
        
        Args:
            name: Name of the network
        """
        self.name = name
        self.nodes = {}  # node_id -> TransportNode
        self.links = {}  # link_id -> TransportLink
        self.time_periods = set()
    
    def add_node(self, node: TransportNode):
        """Add a node to the network"""
        self.nodes[node.node_id] = node
    
    def add_link(self, link: TransportLink):
        """Add a link to the network"""
        self.links[link.link_id] = link
    
    def get_node(self, node_id: str) -> TransportNode:
        """Get a node by its ID"""
        return self.nodes.get(node_id)
    
    def get_link(self, link_id: str) -> TransportLink:
        """Get a link by its ID"""
        return self.links.get(link_id)
    
    def add_weight_data(self, link_id: str, time_period: str, weight: float):
        """Add weight data for a specific link and time period"""
        if link_id in self.links:
            self.links[link_id].add_weight_data(time_period, weight)
            self.time_periods.add(time_period)
    
    def to_networkx(self, time_period: Optional[str] = None) -> nx.DiGraph:
        """
        Convert the transport network to a NetworkX directed graph
        
        Args:
            time_period: Optional time period to use for edge weights
        
        Returns:
            NetworkX DiGraph representation of the network
        """
        G = nx.DiGraph(name=self.name)
        
        # Add nodes with positions
        for node_id, node in self.nodes.items():
            G.add_node(node_id, pos=(node.x, node.y), name=node.name, **node.attributes)
        
        # Add edges with weights
        for link_id, link in self.links.items():
            weight = link.get_weight(time_period) if time_period else 1.0
            G.add_edge(
                link.from_node, 
                link.to_node, 
                id=link_id,
                length=link.length,
                capacity=link.capacity,
                free_speed=link.free_speed,
                weight=weight,
                **link.attributes
            )
        
        return G
    
    def visualize(self, time_period: Optional[str] = None, figsize=(12, 10), 
                  node_size=100, node_color='skyblue', edge_width_factor=1.0):
        """
        Visualize the network with optional weights for a specific time period
        
        Args:
            time_period: Optional time period to use for edge weights
            figsize: Figure size
            node_size: Size of nodes in the visualization
            node_color: Color of nodes
            edge_width_factor: Factor to multiply edge widths by
        """
        G = self.to_networkx(time_period)
        pos = nx.get_node_attributes(G, 'pos')
        
        plt.figure(figsize=figsize)
        
        # Draw nodes
        nx.draw_networkx_nodes(G, pos, node_size=node_size, node_color=node_color)
        
        # Draw edges with width proportional to weight
        if time_period:
            weights = [G[u][v]['weight'] for u, v in G.edges()]
            max_weight = max(weights) if weights else 1.0
            normalized_weights = [w/max_weight * edge_width_factor for w in weights]
            nx.draw_networkx_edges(G, pos, width=normalized_weights, 
                                  edge_color='gray', alpha=0.7, arrows=True)
        else:
            nx.draw_networkx_edges(G, pos, width=1.0, edge_color='gray', alpha=0.7, arrows=True)
        
        # Draw labels
        nx.draw_networkx_labels(G, pos, font_size=8)
        
        title = f"{self.name}"
        if time_period:
            title += f" - Time Period: {time_period}"
        plt.title(title)
        plt.axis('off')
        plt.tight_layout()
        
        return plt
    
    def __repr__(self):
        return f"TransportNetwork(name={self.name}, nodes={len(self.nodes)}, links={len(self.links)})" 