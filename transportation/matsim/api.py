from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel
from typing import List, Dict, Optional
import os
import json

from .network import TransportNetwork, TransportNode, TransportLink
from .simulation import TransportSimulation
from .io import read_network_from_csv, write_network_to_csv

app = FastAPI(title="Oasis Transportation API")

# Store simulations in memory
simulations = {}

class NodeModel(BaseModel):
    node_id: str
    x: float
    y: float
    name: Optional[str] = None
    attributes: Optional[Dict] = None

class LinkModel(BaseModel):
    link_id: str
    from_node: str
    to_node: str
    length: float
    capacity: float
    free_speed: float
    attributes: Optional[Dict] = None

class WeightModel(BaseModel):
    link_id: str
    time_period: str
    weight: float

class NetworkModel(BaseModel):
    name: str
    nodes: List[NodeModel]
    links: List[LinkModel]
    weights: Optional[List[WeightModel]] = None

@app.get("/")
def read_root():
    return {"message": "Welcome to Oasis Transportation API"}

@app.post("/networks/")
def create_network(network: NetworkModel):
    """Create a new transport network"""
    transport_network = TransportNetwork(name=network.name)
    
    # Add nodes
    for node_data in network.nodes:
        node = TransportNode(
            node_id=node_data.node_id,
            x=node_data.x,
            y=node_data.y,
            name=node_data.name or node_data.node_id,
            attributes=node_data.attributes or {}
        )
        transport_network.add_node(node)
    
    # Add links
    for link_data in network.links:
        link = TransportLink(
            link_id=link_data.link_id,
            from_node=link_data.from_node,
            to_node=link_data.to_node,
            length=link_data.length,
            capacity=link_data.capacity,
            free_speed=link_data.free_speed,
            attributes=link_data.attributes or {}
        )
        transport_network.add_link(link)
    
    # Add weights
    if network.weights:
        for weight_data in network.weights:
            transport_network.add_weight_data(
                link_id=weight_data.link_id,
                time_period=weight_data.time_period,
                weight=weight_data.weight
            )
    
    # Save network to CSV files
    output_dir = f"output/{network.name}"
    write_network_to_csv(transport_network, output_dir)
    
    return {"message": f"Network '{network.name}' created successfully", "output_dir": output_dir}

@app.post("/simulations/")
def create_simulation(network_name: str, num_steps: int = 10, seed: int = 42):
    """Create a new simulation based on a network"""
    # Check if network exists
    input_dir = f"output/{network_name}"
    if not os.path.exists(input_dir):
        raise HTTPException(status_code=404, detail=f"Network '{network_name}' not found")
    
    # Load network from CSV files
    nodes_file = os.path.join(input_dir, "nodes.csv")
    links_file = os.path.join(input_dir, "links.csv")
    weights_file = os.path.join(input_dir, "weights.csv")
    
    if not os.path.exists(weights_file):
        weights_file = None
    
    network = read_network_from_csv(nodes_file, links_file, weights_file)
    
    # Create simulation
    simulation_id = f"{network_name}_{seed}"
    simulation = TransportSimulation(network, seed=seed)
    
    # Store simulation
    simulations[simulation_id] = simulation
    
    return {"message": f"Simulation '{simulation_id}' created successfully", "simulation_id": simulation_id}

@app.post("/simulations/{simulation_id}/run")
def run_simulation(simulation_id: str, num_steps: int = 10):
    """Run a simulation for a specified number of steps"""
    if simulation_id not in simulations:
        raise HTTPException(status_code=404, detail=f"Simulation '{simulation_id}' not found")
    
    simulation = simulations[simulation_id]
    
    # Define a simple weight evolution model
    def weight_evolution_model(link_id, current_weight, time_step):
        # Simple model: random fluctuation around current weight
        import random
        random.seed(int(link_id[1:]) + time_step)
        fluctuation = random.uniform(0.95, 1.05)
        return current_weight * fluctuation
    
    # Run simulation
    simulation.simulate_time_evolution(
        num_steps=num_steps,
        weight_evolution_model=weight_evolution_model
    )
    
    # Save output visualization
    output_dir = f"output/simulations/{simulation_id}"
    os.makedirs(output_dir, exist_ok=True)
    
    # Visualize final state for each time period
    time_periods = list(simulation.base_network.time_periods)
    for period in time_periods:
        plt = simulation.get_network_at_time(num_steps).visualize(period)
        plt.savefig(f"{output_dir}/final_state_{period}.png")
        plt.close()
    
    # Visualize evolution for first time period
    if time_periods:
        plt = simulation.visualize_evolution(
            time_steps=list(range(0, num_steps+1, max(1, num_steps//5))),
            time_period=time_periods[0]
        )
        plt.savefig(f"{output_dir}/evolution_{time_periods[0]}.png")
        plt.close()
    
    return {
        "message": f"Simulation '{simulation_id}' completed successfully",
        "num_steps": num_steps,
        "output_dir": output_dir,
        "time_periods": time_periods
    }

@app.get("/simulations/{simulation_id}/network/{time_step}")
def get_network_at_time(simulation_id: str, time_step: int):
    """Get the network state at a specific time step"""
    if simulation_id not in simulations:
        raise HTTPException(status_code=404, detail=f"Simulation '{simulation_id}' not found")
    
    simulation = simulations[simulation_id]
    
    try:
        network = simulation.get_network_at_time(time_step)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    
    # Convert network to JSON-serializable format
    result = {
        "name": network.name,
        "nodes": [],
        "links": [],
        "weights": []
    }
    
    for node_id, node in network.nodes.items():
        result["nodes"].append({
            "node_id": node.node_id,
            "x": node.x,
            "y": node.y,
            "name": node.name,
            "attributes": node.attributes
        })
    
    for link_id, link in network.links.items():
        result["links"].append({
            "link_id": link.link_id,
            "from_node": link.from_node,
            "to_node": link.to_node,
            "length": link.length,
            "capacity": link.capacity,
            "free_speed": link.free_speed,
            "attributes": link.attributes
        })
        
        for time_period, weight in link.weights_history.items():
            result["weights"].append({
                "link_id": link.link_id,
                "time_period": time_period,
                "weight": weight
            })
    
    return result 