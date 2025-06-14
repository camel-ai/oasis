#!/usr/bin/env python3
"""
Demo script for the TrafficSimulator with public transit action space
"""

import os
import sys
import json
from sumo_llm import SUMOEnvironment, TrafficSimulator

def create_sample_environment():
    """Create a sample SUMO environment with mock data for demonstration"""
    # Create a mock SUMO environment
    env = SUMOEnvironment(
        network_file="mock_network.net.xml",
        route_file="mock_routes.rou.xml", 
        config_file="mock.sumocfg",
        output_dir="demo_output"
    )
    
    # Add some sample public transit data
    env.public_transit = {
        "stops": {
            "bus_stop_1": {
                "type": "bus",
                "position": {"x": 100, "y": 200},
                "capacity": 15,
                "served_by": ["bus_route_1", "bus_route_2"]
            },
            "bus_stop_2": {
                "type": "bus", 
                "position": {"x": 300, "y": 200},
                "capacity": 12,
                "served_by": ["bus_route_1"]
            },
            "train_stop_1": {
                "type": "train",
                "position": {"x": 150, "y": 400},
                "capacity": 100,
                "served_by": ["train_line_red", "train_line_blue"]
            },
            "train_stop_2": {
                "type": "train",
                "position": {"x": 500, "y": 400}, 
                "capacity": 80,
                "served_by": ["train_line_red"]
            }
        },
        "buses": {
            "bus_001": {
                "route": ["bus_stop_1", "bus_stop_2", "bus_stop_3"],
                "current_stop": "bus_stop_1",
                "status": "at_stop",
                "capacity": 30,
                "passengers": [],
                "upcoming_stops": ["bus_stop_2", "bus_stop_3"]
            },
            "bus_002": {
                "route": ["bus_stop_2", "bus_stop_1"],
                "current_stop": "bus_stop_2", 
                "status": "at_stop",
                "capacity": 25,
                "passengers": [],
                "upcoming_stops": ["bus_stop_1"]
            }
        },
        "trains": {
            "train_red_01": {
                "route": ["train_stop_1", "train_stop_2", "train_stop_3"],
                "current_stop": "train_stop_1",
                "status": "at_stop", 
                "capacity": 200,
                "passengers": [],
                "upcoming_stops": ["train_stop_2", "train_stop_3"]
            },
            "train_blue_01": {
                "route": ["train_stop_1", "train_stop_4", "train_stop_5"],
                "current_stop": "train_stop_1",
                "status": "at_stop",
                "capacity": 180,
                "passengers": [],
                "upcoming_stops": ["train_stop_4", "train_stop_5"]
            }
        }
    }
    
    return env

def demonstrate_traffic_simulator():
    """Demonstrate the TrafficSimulator functionality"""
    print("🚌🚂 Traffic Simulator Demo 🚂🚌")
    print("=" * 50)
    
    # Create sample environment and simulator
    env = create_sample_environment()
    simulator = TrafficSimulator(env)
    
    print("\n📍 Available Stops:")
    stops = simulator.get_available_stops()
    print(f"Bus stops: {list(stops['bus_stops'].keys())}")
    print(f"Train stops: {list(stops['train_stops'].keys())}")
    
    # Test scenario: Agent travels using public transit
    agent_id = "agent_alice"
    
    print(f"\n👤 Agent '{agent_id}' starts journey")
    
    # 1. Agent gets onto bus
    print("\n🚌 Action 1: Getting onto bus at bus_stop_1")
    result = simulator.get_onto_bus(agent_id, "bus_stop_1")
    print(f"Result: {json.dumps(result, indent=2)}")
    
    # Check agent status
    status = simulator.get_agent_status(agent_id)
    print(f"\nAgent status: {json.dumps(status, indent=2)}")
    
    # 2. Agent gets off bus
    print("\n🚌 Action 2: Getting off bus at bus_stop_2")
    # First, simulate bus moving to bus_stop_2
    simulator.buses["bus_001"]["current_stop"] = "bus_stop_2"
    
    result = simulator.get_off_bus(agent_id, "bus_stop_2")
    print(f"Result: {json.dumps(result, indent=2)}")
    
    # 3. Agent gets onto train
    print("\n🚂 Action 3: Getting onto train at train_stop_1")
    result = simulator.get_onto_train(agent_id, "train_stop_1")
    print(f"Result: {json.dumps(result, indent=2)}")
    
    # Check agent status again
    status = simulator.get_agent_status(agent_id)
    print(f"\nAgent status: {json.dumps(status, indent=2)}")
    
    # 4. Agent gets off train
    print("\n🚂 Action 4: Getting off train at train_stop_2")
    # Simulate train moving to train_stop_2
    simulator.trains["train_red_01"]["current_stop"] = "train_stop_2"
    
    result = simulator.get_off_train(agent_id, "train_stop_2")
    print(f"Result: {json.dumps(result, indent=2)}")
    
    # Final status
    status = simulator.get_agent_status(agent_id)
    print(f"\nFinal agent status: {json.dumps(status, indent=2)}")
    
    # Test error scenarios
    print("\n❌ Testing Error Scenarios:")
    
    # Try to board non-existent bus
    print("\n1. Trying to board at non-existent stop:")
    result = simulator.get_onto_bus("agent_bob", "non_existent_stop")
    print(f"Result: {result['error']}")
    
    # Try to get off when not on vehicle
    print("\n2. Trying to get off bus when not on bus:")
    result = simulator.get_off_bus("agent_bob", "bus_stop_1")
    print(f"Result: {result['error']}")
    
    # Try to board when no vehicle available
    print("\n3. Trying to board when no bus available:")
    # Make bus unavailable
    simulator.buses["bus_001"]["status"] = "en_route"
    result = simulator.get_onto_bus("agent_charlie", "bus_stop_1")
    print(f"Result: {result['error']}")
    print(f"Estimated wait time: {result.get('wait_time', 'N/A')} seconds")
    
    print("\n✅ Demo completed!")

def test_multiple_agents():
    """Test multiple agents using the system simultaneously"""
    print("\n👥 Multi-Agent Test")
    print("-" * 30)
    
    env = create_sample_environment()
    simulator = TrafficSimulator(env)
    
    agents = ["agent_1", "agent_2", "agent_3"]
    
    # All agents try to board the same bus
    print("Multiple agents boarding bus_001:")
    for agent in agents:
        result = simulator.get_onto_bus(agent, "bus_stop_1")
        print(f"{agent}: {'Success' if result['success'] else result['error']}")
    
    # Check passenger count
    bus_passengers = simulator.buses["bus_001"]["passengers"]
    print(f"\nBus passengers: {bus_passengers}")
    print(f"Total passengers on bus: {len(bus_passengers)}")

if __name__ == "__main__":
    try:
        demonstrate_traffic_simulator()
        test_multiple_agents()
    except Exception as e:
        print(f"Error running demo: {e}")
        import traceback
        traceback.print_exc()
