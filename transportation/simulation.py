import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Callable
import matplotlib.pyplot as plt
from .network import TransportNetwork, TransportNode, TransportLink
import copy
import random

class TransportSimulation:
    """Simulates transportation network evolution over time"""
    
    def __init__(self, network: TransportNetwork, seed: int = 42):
        """
        Initialize a transport simulation
        
        Args:
            network: The base transport network to simulate
            seed: Random seed for reproducibility
        """
        self.base_network = network
        self.simulated_networks = {}  # time_step -> TransportNetwork
        self.random = random.Random(seed)
        
        # Store the original network as the starting point
        self.simulated_networks[0] = copy.deepcopy(network)
    
    def simulate_time_evolution(self, 
                               num_steps: int, 
                               weight_evolution_model: Callable[[str, float, int], float],
                               time_periods: Optional[List[str]] = None):
        """
        Simulate the evolution of the network over time
        
        Args:
            num_steps: Number of time steps to simulate
            weight_evolution_model: Function that takes (link_id, current_weight, time_step) 
                                   and returns the new weight
            time_periods: List of time periods to simulate (if None, use all time periods in base network)
        """
        if time_periods is None:
            time_periods = list(self.base_network.time_periods)
        
        # Start with the base network
        current_network = self.simulated_networks[0]
        
        # Simulate each time step
        for step in range(1, num_steps + 1):
            # Create a new network for this time step
            new_network = copy.deepcopy(current_network)
            
            # Update weights for each link and time period
            for link_id, link in new_network.links.items():
                for period in time_periods:
                    if period in link.weights_history:
                        current_weight = link.weights_history[period]
                        new_weight = weight_evolution_model(link_id, current_weight, step)
                        link.weights_history[period] = new_weight
            
            # Store the network for this time step
            self.simulated_networks[step] = new_network
            current_network = new_network
    
    def get_network_at_time(self, time_step: int) -> TransportNetwork:
        """Get the simulated network at a specific time step"""
        if time_step in self.simulated_networks:
            return self.simulated_networks[time_step]
        else:
            raise ValueError(f"No simulation data for time step {time_step}")
    
    def visualize_evolution(self, time_steps: List[int], time_period: str, 
                           figsize=(18, 12), rows=1):
        """
        Visualize the network evolution over multiple time steps
        
        Args:
            time_steps: List of time steps to visualize
            time_period: Time period to visualize
            figsize: Figure size
            rows: Number of rows in the subplot grid
        """
        cols = int(np.ceil(len(time_steps) / rows))
        fig = plt.figure(figsize=figsize)
        
        for i, step in enumerate(time_steps):
            network = self.get_network_at_time(step)
            ax = fig.add_subplot(rows, cols, i+1)
            
            G = network.to_networkx(time_period)
            pos = nx.get_node_attributes(G, 'pos')
            
            # Draw nodes
            nx.draw_networkx_nodes(G, pos, node_size=80, node_color='skyblue', ax=ax)
            
            # Draw edges with width proportional to weight
            weights = [G[u][v]['weight'] for u, v in G.edges()]
            max_weight = max(weights) if weights else 1.0
            normalized_weights = [w/max_weight * 3 for w in weights]
            nx.draw_networkx_edges(G, pos, width=normalized_weights, 
                                  edge_color='gray', alpha=0.7, arrows=True, ax=ax)
            
            # Draw labels
            nx.draw_networkx_labels(G, pos, font_size=7, ax=ax)
            
            ax.set_title(f"Time Step: {step}, Period: {time_period}")
            ax.axis('off')
        
        plt.tight_layout()
        return plt
    
    def get_weight_evolution_data(self, link_id: str, time_period: str) -> Dict[int, float]:
        """
        Get the evolution of weights for a specific link and time period
        
        Returns:
            Dictionary mapping time steps to weights
        """
        result = {}
        for step, network in self.simulated_networks.items():
            link = network.get_link(link_id)
            if link and time_period in link.weights_history:
                result[step] = link.weights_history[time_period]
        return result
    
    def plot_weight_evolution(self, link_ids: List[str], time_period: str, figsize=(10, 6)):
        """
        Plot the evolution of weights for specific links over time
        
        Args:
            link_ids: List of link IDs to plot
            time_period: Time period to plot
            figsize: Figure size
        """
        plt.figure(figsize=figsize)
        
        for link_id in link_ids:
            data = self.get_weight_evolution_data(link_id, time_period)
            if data:
                steps = sorted(data.keys())
                weights = [data[step] for step in steps]
                plt.plot(steps, weights, marker='o', label=f"Link {link_id}")
        
        plt.xlabel("Time Step")
        plt.ylabel("Weight")
        plt.title(f"Weight Evolution for Time Period: {time_period}")
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        return plt 