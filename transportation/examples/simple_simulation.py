import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
import networkx as nx

# Add the parent directory to the path so we can import the modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from oasis.transportation.network import TransportNetwork, TransportNode, TransportLink
from oasis.transportation.simulation import TransportSimulation
from oasis.transportation.io import read_network_from_csv, write_network_to_csv

def create_sample_network():
    """Create a sample transport network for demonstration"""
    network = TransportNetwork(name="Sample City Network")
    
    # Add nodes (stations)
    nodes = [
        TransportNode("n1", 0, 0, "Downtown"),
        TransportNode("n2", 10, 0, "East End"),
        TransportNode("n3", 0, 10, "North Side"),
        TransportNode("n4", 10, 10, "Northeast"),
        TransportNode("n5", 5, 5, "Central"),
    ]
    
    for node in nodes:
        network.add_node(node)
    
    # Add links (routes)
    links = [
        TransportLink("l1", "n1", "n2", 10000, 1800, 15, {"type": "main_road"}),
        TransportLink("l2", "n2", "n1", 10000, 1800, 15, {"type": "main_road"}),
        TransportLink("l3", "n1", "n3", 10000, 1200, 12, {"type": "secondary"}),
        TransportLink("l4", "n3", "n1", 10000, 1200, 12, {"type": "secondary"}),
        TransportLink("l5", "n2", "n4", 10000, 1200, 12, {"type": "secondary"}),
        TransportLink("l6", "n4", "n2", 10000, 1200, 12, {"type": "secondary"}),
        TransportLink("l7", "n3", "n4", 10000, 1500, 13, {"type": "main_road"}),
        TransportLink("l8", "n4", "n3", 10000, 1500, 13, {"type": "main_road"}),
        TransportLink("l9", "n1", "n5", 7000, 1000, 10, {"type": "local"}),
        TransportLink("l10", "n5", "n1", 7000, 1000, 10, {"type": "local"}),
        TransportLink("l11", "n2", "n5", 7000, 1000, 10, {"type": "local"}),
        TransportLink("l12", "n5", "n2", 7000, 1000, 10, {"type": "local"}),
        TransportLink("l13", "n3", "n5", 7000, 1000, 10, {"type": "local"}),
        TransportLink("l14", "n5", "n3", 7000, 1000, 10, {"type": "local"}),
        TransportLink("l15", "n4", "n5", 7000, 1000, 10, {"type": "local"}),
        TransportLink("l16", "n5", "n4", 7000, 1000, 10, {"type": "local"}),
    ]
    
    for link in links:
        network.add_link(link)
    
    # Add weights for different time periods
    time_periods = ["morning_peak", "midday", "evening_peak", "night"]
    
    # Set initial weights
    for link_id, link in network.links.items():
        # Morning peak: heavy traffic from residential to downtown
        if link.to_node == "n1" and link.from_node in ["n2", "n3", "n4"]:
            network.add_weight_data(link_id, "morning_peak", 3.0)
        elif link.from_node == "n1" and link.to_node in ["n2", "n3", "n4"]:
            network.add_weight_data(link_id, "morning_peak", 1.0)
        else:
            network.add_weight_data(link_id, "morning_peak", 1.5)
        
        # Midday: balanced traffic
        network.add_weight_data(link_id, "midday", 1.5)
        
        # Evening peak: heavy traffic from downtown to residential
        if link.from_node == "n1" and link.to_node in ["n2", "n3", "n4"]:
            network.add_weight_data(link_id, "evening_peak", 3.0)
        elif link.to_node == "n1" and link.from_node in ["n2", "n3", "n4"]:
            network.add_weight_data(link_id, "evening_peak", 1.0)
        else:
            network.add_weight_data(link_id, "evening_peak", 1.5)
        
        # Night: light traffic
        network.add_weight_data(link_id, "night", 0.5)
    
    return network

def weight_evolution_model(link_id, current_weight, time_step):
    """
    Simple model for weight evolution over time
    
    This model:
    1. Adds a small random fluctuation
    2. Gradually increases weights for main roads
    3. Gradually decreases weights for local roads
    """
    # Get link type from the first character of the link_id
    link_type = int(link_id[1:])
    
    # Random fluctuation
    random_factor = random.uniform(0.95, 1.05)
    
    # Trend factor based on link type
    if link_type <= 8:  # Main roads
        trend_factor = 1.01  # Slight increase
    else:  # Local roads
        trend_factor = 0.99  # Slight decrease
    
    # Calculate new weight
    new_weight = current_weight * random_factor * trend_factor
    
    # Ensure weight stays within reasonable bounds
    new_weight = max(0.1, min(5.0, new_weight))
    
    return new_weight

def main():
    # Create output directory
    os.makedirs("output", exist_ok=True)
    
    # Create sample network
    network = create_sample_network()
    
    # Save network to CSV
    write_network_to_csv(network, "output")
    
    # Visualize initial network for different time periods
    for period in ["morning_peak", "midday", "evening_peak", "night"]:
        plt = network.visualize(period)
        plt.savefig(f"output/initial_network_{period}.png")
        plt.close()
    
    # Create simulation
    simulation = TransportSimulation(network)
    
    # Simulate 20 time steps
    simulation.simulate_time_evolution(
        num_steps=20,
        weight_evolution_model=weight_evolution_model
    )
    
    # Visualize evolution for morning peak
    plt = simulation.visualize_evolution(
        time_steps=[0, 5, 10, 15, 20],
        time_period="morning_peak",
        figsize=(20, 10),
        rows=1
    )
    plt.savefig("output/evolution_morning_peak.png")
    plt.close()
    
    # Visualize evolution for evening peak
    plt = simulation.visualize_evolution(
        time_steps=[0, 5, 10, 15, 20],
        time_period="evening_peak",
        figsize=(20, 10),
        rows=1
    )
    plt.savefig("output/evolution_evening_peak.png")
    plt.close()
    
    # Plot weight evolution for selected links
    plt = simulation.plot_weight_evolution(
        link_ids=["l1", "l3", "l9", "l15"],
        time_period="morning_peak"
    )
    plt.savefig("output/weight_evolution_morning.png")
    plt.close()
    
    print("Simulation completed successfully!")
    print("Output files saved to the 'output' directory.")

if __name__ == "__main__":
    main()
