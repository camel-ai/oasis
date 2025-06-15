import os
import json
import random
import requests
import traci
import subprocess
from typing import List, Dict, Any, Optional, Tuple
import xml.etree.ElementTree as ET
import tempfile
import shutil
import sumo_agent
from sumo_env import SUMOEnvironment
def modify_routes_file(original_route_file: str, output_route_file: str, 
                      route_adjustments: Dict[str, Dict[str, Any]]):
    """Modify a SUMO route file based on LLM suggestions, including multimodal options
    
    Args:
        original_route_file: Path to the original route XML file
        output_route_file: Path to write the modified route file
        route_adjustments: Dictionary with vehicle IDs and route adjustments
    """
    try:
        # Parse the original route file
        tree = ET.parse(original_route_file)
        root = tree.getroot()
        
        # Track park and ride activities for multimodal journeys
        park_and_ride_stops = {}
        
        # Process each vehicle element
        for vehicle in root.findall(".//vehicle"):
            vehicle_id = vehicle.get("id")
            
            if vehicle_id in route_adjustments:
                adjustments = route_adjustments[vehicle_id]
                
                # Find the route element
                route_elem = vehicle.find("route")
                if route_elem is not None:
                    # Get current edges
                    edges = route_elem.get("edges", "").split()
                    
                    # Apply standard route adjustments
                    for adjustment in adjustments.get("route_adjustments", []):
                        edge_to_avoid = adjustment.get("edge_to_avoid")
                        alternative_edge = adjustment.get("alternative_edge")
                        
                        if edge_to_avoid in edges and alternative_edge is not None:
                            # Replace the edge to avoid with the alternative
                            index = edges.index(edge_to_avoid)
                            edges[index] = alternative_edge
                    
                    # Handle multimodal journey if recommended
                    multimodal = adjustments.get("multimodal_strategy", {})
                    if multimodal.get("use_multimodal", False):
                        # Process transitions between modes
                        for transition in multimodal.get("transitions", []):
                            from_mode = transition.get("from_mode")
                            to_mode = transition.get("to_mode")
                            location = transition.get("at_location")
                            
                            # Find matching public transit recommendations
                            for pt_rec in adjustments.get("public_transit_recommendations", []):
                                if pt_rec.get("type") == to_mode and pt_rec.get("board_at_stop") == location:
                                    # Record park and ride location
                                    park_and_ride_stops[vehicle_id] = {
                                        "stop_id": location,
                                        "vehicle_id": vehicle_id,
                                        "pt_type": to_mode,
                                        "pt_id": pt_rec.get("id"),
                                        "board_at": pt_rec.get("board_at_stop"),
                                        "alight_at": pt_rec.get("alight_at_stop")
                                    }
                                    
                                    # Modify route to end at the park and ride location
                                    # In a real implementation, we'd look up the edge ID for the stop
                                    # For now, just truncate the route at a reasonable point
                                    if len(edges) > 3:
                                        # Keep route up to about halfway through (simulating driving to P+R)
                                        truncate_point = len(edges) // 2
                                        edges = edges[:truncate_point]
                    
                    # Update route with the modified edges
                    route_elem.set("edges", " ".join(edges))
        
        # Create public transit trips for vehicles that switch to public transport
        if park_and_ride_stops:
            # In a real implementation, we would:
            # 1. Add person definitions for each driver who leaves their vehicle
            # 2. Add walk legs to connect from the parking spot to transit stop
            # 3. Add public transit trips using the specific bus/train IDs
            # 4. Add walk legs to destination
            
            print(f"Multimodal journeys planned for {len(park_and_ride_stops)} vehicles")
            for vehicle_id, pr_data in park_and_ride_stops.items():
                print(f"  - Vehicle {vehicle_id} will switch to {pr_data['pt_type']} at {pr_data['board_at']}")
                
                # Create a person element for the driver
                person_id = f"person_{vehicle_id}"
                person_elem = ET.SubElement(root, "person", id=person_id)
                
                # Add a plan element
                plan_elem = ET.SubElement(person_elem, "plan")
                
                # Add the initial drive leg (handled by the vehicle route adjustment above)
                
                # Add activity for parking
                ET.SubElement(plan_elem, "stop", 
                             location=pr_data["board_at"],
                             actType="parking")
                
                # Add a walk leg to the transit stop
                ET.SubElement(plan_elem, "walk", 
                             from_=pr_data["board_at"],
                             to=pr_data["board_at"],
                             duration="60")  # 60 seconds walking time
                
                # Add wait for transit
                ET.SubElement(plan_elem, "stop", 
                             location=pr_data["board_at"],
                             actType="waiting")
                
                # Add the public transit leg
                ET.SubElement(plan_elem, "ride", 
                             from_=pr_data["board_at"],
                             to=pr_data["alight_at"],
                             lines=pr_data["pt_id"])
                
                # Add final walk leg to destination
                ET.SubElement(plan_elem, "walk", 
                             from_=pr_data["alight_at"],
                             to="destination",  # In real implementation, use actual destination
                             duration="120")  # 120 seconds walking time
        
        # Write modified XML to output file
        tree.write(output_route_file)
        print(f"Modified routes written to {output_route_file}")
        
    except Exception as e:
        print(f"Error modifying routes file: {e}")
        
        # On failure, copy the original file
        shutil.copy(original_route_file, output_route_file)
        print(f"Copied original routes to {output_route_file}")


def run_simulation(network_file: str, route_file: str, config_file: str,
                  output_dir: str, sumo_binary: str, openai_api_key: str, 
                  num_iterations: int, additional_files: List[str] = None):
    """Run the simulation for multiple iterations
    
    Args:
        network_file: Path to network XML file
        route_file: Path to route XML file
        config_file: Path to SUMO config file
        output_dir: Directory to store output files
        sumo_binary: Path to SUMO or SUMO-GUI binary
        openai_api_key: OpenAI API key for LLM (can be empty)
        num_iterations: Number of iterations to run
        additional_files: List of additional files (e.g., public transit schedules)
    """
    # Initialize environment
    env = SUMOEnvironment(network_file, route_file, config_file, output_dir, additional_files)
    
    # Initial route file
    current_route_file = route_file
    
    # Run simulation iterations
    for i in range(num_iterations):
        print(f"\nIteration {i+1}/{num_iterations}")
        
        # Run SUMO and get network state
        network_state = env.run_iteration(sumo_binary)
        
        # Only create LLM agents after the first iteration when we have vehicle data
        if i > 0 or env.vehicles:
            # Create LLM agents for vehicles seen in the simulation
            agents = []
            for vehicle_id in env.vehicles:
                agents.append(sumo_agent.LLMAgent(vehicle_id, env, openai_api_key))
            
            print(f"Created {len(agents)} LLM agents")
            
            # Update vehicle routes based on network state using LLM
            route_adjustments = {}
            for agent in agents:
                adjustment = agent.update_route(network_state)
                if adjustment:
                    route_adjustments[agent.vehicle_id] = adjustment
            
            # Create updated route file for the next iteration
            next_route_file = os.path.join(output_dir, f"updated_routes_{i}.xml")
            modify_routes_file(current_route_file, next_route_file, route_adjustments)
            
            # Use the new route file for the next iteration
            current_route_file = next_route_file
    
    print(f"Simulation completed after {num_iterations} iterations")


class TrafficSimulator:
    """Traffic simulator with public transit action space"""
    
    def __init__(self, sumo_environment: SUMOEnvironment):
        """Initialize the traffic simulator
        
        Args:
            sumo_environment: SUMOEnvironment instance for underlying simulation
        """
        self.env = sumo_environment
        self.agents = {}  # Dict to store agent states
        self.bus_stops = {}  # Dict to store bus stop info
        self.train_stops = {}  # Dict to store train stop info
        self.buses = {}  # Dict to store bus info
        self.trains = {}  # Dict to store train info
        
        # Initialize transit infrastructure
        self._initialize_transit_infrastructure()
    
    def _initialize_transit_infrastructure(self):
        """Initialize bus and train stops from SUMO environment"""
        # Extract bus stops from public transit data
        if self.env.public_transit.get("stops"):
            for stop_id, stop_data in self.env.public_transit["stops"].items():
                if stop_data.get("type") == "bus":
                    self.bus_stops[stop_id] = {
                        "id": stop_id,
                        "position": stop_data.get("position"),
                        "capacity": stop_data.get("capacity", 10),
                        "waiting_agents": [],
                        "served_by": stop_data.get("served_by", [])
                    }
                elif stop_data.get("type") == "train":
                    self.train_stops[stop_id] = {
                        "id": stop_id,
                        "position": stop_data.get("position"),
                        "capacity": stop_data.get("capacity", 50),
                        "waiting_agents": [],
                        "served_by": stop_data.get("served_by", [])
                    }
        
        # Extract bus and train info
        if self.env.public_transit.get("buses"):
            self.buses = self.env.public_transit["buses"].copy()
        if self.env.public_transit.get("trains"):
            self.trains = self.env.public_transit["trains"].copy()
    
    def get_onto_bus(self, agent_id: str, bus_stop_id: str) -> Dict[str, Any]:
        """Agent gets onto bus at specified stop
        
        This function simulates an agent boarding a bus at a specific bus stop.
        It performs comprehensive validation, capacity checking, and state management.
        
        Step-by-step process:
        1. Validate that the requested bus stop exists in the network
        2. Initialize agent state if this is a new agent
        3. Find all buses currently at the stop that have available capacity
        4. Use SUMO TraCI (if available) to get real-time capacity data
        5. Select the best bus (most available space) for boarding
        6. Update passenger lists and agent state
        7. Remove agent from waiting lists if applicable
        8. Return detailed result with capacity information
        
        Args:
            agent_id: Unique identifier for the agent attempting to board
            bus_stop_id: Unique identifier for the bus stop location
            
        Returns:
            Dictionary with action result, success status, and detailed information:
            - success: Boolean indicating if boarding was successful
            - error: Error message if boarding failed
            - bus_id: ID of the boarded bus (if successful)
            - capacity_info: Real-time capacity details
            - wait_time: Estimated wait time if no bus available
        """
        try:
            # STEP 1: Validate bus stop existence
            # Check if the requested bus stop exists in our transit network
            if bus_stop_id not in self.bus_stops:
                return {
                    "success": False,
                    "error": f"Bus stop {bus_stop_id} does not exist",
                    "agent_id": agent_id,
                    "action": "get_onto_bus"
                }
            
            # STEP 2: Initialize agent state if needed
            # Create agent record if this is their first action in the system
            if agent_id not in self.agents:
                self.agents[agent_id] = {
                    "current_location": None,
                    "mode": "walking",  # Default mode is walking
                    "vehicle_id": None,
                    "destination": None
                }
            
            # STEP 3: Find available buses at the stop
            # Search for buses that are:
            # - Currently at the requested stop
            # - In "at_stop" status (not departing)
            # - Have available capacity for more passengers
            available_buses = []
            for bus_id, bus_data in self.buses.items():
                if (bus_data.get("current_stop") == bus_stop_id and 
                    bus_data.get("status") == "at_stop"):
                    
                    # STEP 4: Check capacity using SUMO TraCI if simulation is running
                    if self.env.sim_running:
                        try:
                            # Use real-time SUMO data for accurate capacity info
                            capacity = traci.vehicle.getPersonCapacity(bus_id)
                            current_passengers = traci.vehicle.getPersonNumber(bus_id)
                            if current_passengers < capacity:
                                available_buses.append({
                                    "id": bus_id,
                                    "capacity": capacity,
                                    "current_passengers": current_passengers,
                                    "available_space": capacity - current_passengers
                                })
                        except traci.exceptions.TraCIException:
                            # STEP 4a: Fallback to simulated data if TraCI fails
                            capacity = bus_data.get("capacity", 30)
                            current_passengers = len(bus_data.get("passengers", []))
                            if current_passengers < capacity:
                                available_buses.append({
                                    "id": bus_id,
                                    "capacity": capacity,
                                    "current_passengers": current_passengers,
                                    "available_space": capacity - current_passengers
                                })
                    else:
                        # STEP 4b: Use simulated data when SUMO is not running
                        capacity = bus_data.get("capacity", 30)
                        current_passengers = len(bus_data.get("passengers", []))
                        if current_passengers < capacity:
                            available_buses.append({
                                "id": bus_id,
                                "capacity": capacity,
                                "current_passengers": current_passengers,
                                "available_space": capacity - current_passengers
                            })
            
            # STEP 5: Handle case when no buses have available capacity
            if not available_buses:
                # Check if buses exist but are at capacity vs no buses at all
                buses_at_stop = []
                for bus_id, bus_data in self.buses.items():
                    if (bus_data.get("current_stop") == bus_stop_id and 
                        bus_data.get("status") == "at_stop"):
                        buses_at_stop.append(bus_id)
                
                if buses_at_stop:
                    # Buses exist but are full - agent must wait for space or next bus
                    return {
                        "success": False,
                        "error": f"All buses at stop {bus_stop_id} are at capacity",
                        "agent_id": agent_id,
                        "action": "get_onto_bus",
                        "wait_time": random.randint(30, 300),  # Wait for next bus or space
                        "buses_at_capacity": buses_at_stop
                    }
                else:
                    # No buses at the stop at all - agent must wait for arrival
                    return {
                        "success": False,
                        "error": f"No available bus at stop {bus_stop_id}",
                        "agent_id": agent_id,
                        "action": "get_onto_bus",
                        "wait_time": random.randint(30, 300)  # Estimated wait time in seconds
                    }
            
            # STEP 6: Select the best bus for boarding
            # Choose the bus with the most available space to optimize passenger distribution
            best_bus = max(available_buses, key=lambda x: x["available_space"])
            bus_id = best_bus["id"]
            
            # STEP 7: Add passenger to bus and update tracking
            if self.env.sim_running:
                try:
                    # In SUMO, passengers are managed through person simulation
                    # For now, we track manually and sync with SUMO state
                    if "passengers" not in self.buses[bus_id]:
                        self.buses[bus_id]["passengers"] = []
                    self.buses[bus_id]["passengers"].append(agent_id)
                except traci.exceptions.TraCIException as e:
                    print(f"Warning: Could not sync passenger data with SUMO: {e}")
                    # Continue with manual tracking even if SUMO sync fails
                    if "passengers" not in self.buses[bus_id]:
                        self.buses[bus_id]["passengers"] = []
                    self.buses[bus_id]["passengers"].append(agent_id)
            else:
                # Manual tracking when SUMO is not running
                if "passengers" not in self.buses[bus_id]:
                    self.buses[bus_id]["passengers"] = []
                self.buses[bus_id]["passengers"].append(agent_id)
            
            # STEP 8: Update agent state to reflect boarding
            self.agents[agent_id].update({
                "current_location": bus_stop_id,
                "mode": "bus",  # Agent is now on a bus
                "vehicle_id": bus_id,  # Track which specific bus
                "boarded_at": bus_stop_id  # Remember where they got on
            })
            
            # STEP 9: Clean up waiting lists
            # Remove agent from bus stop waiting list if they were waiting
            if agent_id in self.bus_stops[bus_stop_id]["waiting_agents"]:
                self.bus_stops[bus_stop_id]["waiting_agents"].remove(agent_id)
            
            # STEP 10: Return success response with detailed information
            return {
                "success": True,
                "agent_id": agent_id,
                "action": "get_onto_bus",
                "bus_stop_id": bus_stop_id,
                "bus_id": bus_id,
                "bus_route": self.buses[bus_id].get("route", []),
                "next_stops": self.buses[bus_id].get("upcoming_stops", []),
                "capacity_info": {
                    "total_capacity": best_bus["capacity"],
                    "current_passengers": best_bus["current_passengers"] + 1,  # +1 for this agent
                    "remaining_space": best_bus["available_space"] - 1
                }
            }
            
        except Exception as e:
            # Handle any unexpected errors gracefully
            return {
                "success": False,
                "error": f"Error boarding bus: {str(e)}",
                "agent_id": agent_id,
                "action": "get_onto_bus"
            }
    
    def get_off_bus(self, agent_id: str, bus_stop_id: str) -> Dict[str, Any]:
        """Agent gets off bus at specified stop
        
        This function simulates an agent alighting (getting off) a bus at a specific stop.
        It performs validation to ensure the agent is on a bus and the bus is at the correct stop.
        
        Step-by-step process:
        1. Validate that the agent exists and is currently on a bus
        2. Verify the bus stop exists in the network
        3. Check that the agent's current bus is actually at the requested stop
        4. Remove the agent from the bus passenger list
        5. Update agent state to reflect they are now walking
        6. Optionally add agent to stop waiting list if they plan to transfer
        7. Return detailed result with updated agent status
        
        Args:
            agent_id: Unique identifier for the agent attempting to alight
            bus_stop_id: Unique identifier for the bus stop where agent wants to get off
            
        Returns:
            Dictionary with action result, success status, and detailed information:
            - success: Boolean indicating if alighting was successful
            - error: Error message if alighting failed
            - previous_bus_id: ID of the bus the agent was on
            - new_location: Updated location after alighting
            - new_mode: Updated mode after alighting
            - transfer_options: Available transit options at this stop
        """
        try:
            # STEP 1: Validate agent exists and is on a bus
            # Check if agent exists in our system
            if agent_id not in self.agents:
                return {
                    "success": False,
                    "error": f"Agent {agent_id} does not exist",
                    "agent_id": agent_id,
                    "action": "get_off_bus"
                }
            
            agent = self.agents[agent_id]
            
            # Verify agent is currently on a bus (not walking or on other transport)
            if agent.get("mode") != "bus":
                return {
                    "success": False,
                    "error": f"Agent {agent_id} is not currently on a bus (mode: {agent.get('mode')})",
                    "agent_id": agent_id,
                    "action": "get_off_bus"
                }
            
            # Verify agent has a valid bus ID
            bus_id = agent.get("vehicle_id")
            if not bus_id:
                return {
                    "success": False,
                    "error": f"Agent {agent_id} does not have a valid bus ID",
                    "agent_id": agent_id,
                    "action": "get_off_bus"
                }
            
            # STEP 2: Validate bus stop exists
            if bus_stop_id not in self.bus_stops:
                return {
                    "success": False,
                    "error": f"Bus stop {bus_stop_id} does not exist",
                    "agent_id": agent_id,
                    "action": "get_off_bus"
                }
            
            # STEP 3: Verify the bus is actually at the requested stop
            # Check if the bus exists in our system
            if bus_id not in self.buses:
                return {
                    "success": False,
                    "error": f"Bus {bus_id} not found in system",
                    "agent_id": agent_id,
                    "action": "get_off_bus"
                }
            
            bus = self.buses[bus_id]
            
            # Verify the bus is currently at the stop where agent wants to get off
            if bus.get("current_stop") != bus_stop_id:
                return {
                    "success": False,
                    "error": f"Bus {bus_id} is not at stop {bus_stop_id} (currently at: {bus.get('current_stop')})",
                    "agent_id": agent_id,
                    "action": "get_off_bus",
                    "bus_current_stop": bus.get("current_stop")
                }
            
            # Verify the bus is in a state where passengers can alight
            if bus.get("status") != "at_stop":
                return {
                    "success": False,
                    "error": f"Bus {bus_id} is not stopped at {bus_stop_id} (status: {bus.get('status')})",
                    "agent_id": agent_id,
                    "action": "get_off_bus"
                }
            
            # STEP 4: Remove agent from bus passenger list
            # Initialize passengers list if it doesn't exist
            if "passengers" not in bus:
                bus["passengers"] = []
            
            # Check if agent is actually on the passenger list
            if agent_id not in bus["passengers"]:
                return {
                    "success": False,
                    "error": f"Agent {agent_id} is not listed as passenger on bus {bus_id}",
                    "agent_id": agent_id,
                    "action": "get_off_bus"
                }
            
            # Remove agent from bus passenger list
            bus["passengers"].remove(agent_id)
            
            # Sync with SUMO if simulation is running
            if self.env.sim_running:
                try:
                    # In SUMO, passenger alighting is managed through person simulation
                    # For now, we track manually and would sync with SUMO state
                    current_passengers = traci.vehicle.getPersonNumber(bus_id)
                    # Note: In full SUMO integration, we would use person commands here
                except traci.exceptions.TraCIException as e:
                    print(f"Warning: Could not sync passenger data with SUMO: {e}")
                    # Continue with manual tracking even if SUMO sync fails
            
            # STEP 5: Update agent state to reflect alighting
            previous_bus_id = agent["vehicle_id"]
            boarded_at = agent.get("boarded_at")
            
            self.agents[agent_id].update({
                "current_location": bus_stop_id,
                "mode": "walking",  # Agent is now walking
                "vehicle_id": None,  # No longer on any vehicle
                "previous_bus": previous_bus_id,  # Track for history/analytics
                "alighted_at": bus_stop_id  # Remember where they got off
            })
            
            # STEP 6: Handle potential transfers
            # Get available transfer options at this stop
            transfer_options = {
                "buses": [],
                "trains": []
            }
            
            # Find other buses at this stop
            for other_bus_id, other_bus_data in self.buses.items():
                if (other_bus_id != bus_id and  # Don't include the bus they just got off
                    other_bus_data.get("current_stop") == bus_stop_id and
                    other_bus_data.get("status") == "at_stop"):
                    
                    # Check capacity for potential boarding
                    capacity = other_bus_data.get("capacity", 30)
                    current_passengers = len(other_bus_data.get("passengers", []))
                    if current_passengers < capacity:
                        transfer_options["buses"].append({
                            "bus_id": other_bus_id,
                            "route": other_bus_data.get("route", []),
                            "available_space": capacity - current_passengers
                        })
            
            # Find trains at nearby stations (if this stop connects to rail)
            stop_info = self.bus_stops[bus_stop_id]
            if stop_info.get("connects_to_rail"):
                # Logic for finding connected train stations would go here
                # For now, simplified approach
                for train_stop_id, train_stop_data in self.train_stops.items():
                    if train_stop_data.get("connects_to_bus_stop") == bus_stop_id:
                        for train_id, train_data in self.trains.items():
                            if (train_data.get("current_stop") == train_stop_id and
                                train_data.get("status") == "at_stop"):
                                
                                capacity = train_data.get("capacity", 200)
                                current_passengers = len(train_data.get("passengers", []))
                                if current_passengers < capacity:
                                    transfer_options["trains"].append({
                                        "train_id": train_id,
                                        "station_id": train_stop_id,
                                        "route": train_data.get("route", []),
                                        "available_space": capacity - current_passengers
                                    })
            
            # STEP 7: Return success response with detailed information
            return {
                "success": True,
                "agent_id": agent_id,
                "action": "get_off_bus",
                "bus_stop_id": bus_stop_id,
                "previous_bus_id": previous_bus_id,
                "boarded_at": boarded_at,
                "new_location": bus_stop_id,
                "new_mode": "walking",
                "transfer_options": transfer_options,
                "journey_segment": {
                    "from": boarded_at,
                    "to": bus_stop_id,
                    "vehicle": previous_bus_id,
                    "mode": "bus"
                }
            }
            
        except Exception as e:
            # Handle any unexpected errors gracefully
            return {
                "success": False,
                "error": f"Error alighting from bus: {str(e)}",
                "agent_id": agent_id,
                "action": "get_off_bus"
            }
    
    def get_onto_train(self, agent_id: str, train_stop_id: str) -> Dict[str, Any]:
        """Agent gets onto train at specified stop
        
        This function simulates an agent boarding a train at a specific train station.
        It performs comprehensive validation, capacity checking, and state management.
        Trains typically have higher capacity than buses and longer boarding times.
        
        Step-by-step process:
        1. Validate that the requested train station exists in the network
        2. Initialize agent state if this is a new agent
        3. Find all trains currently at the station that have available capacity
        4. Use SUMO TraCI (if available) to get real-time capacity data
        5. Select the best train (most available space) for boarding
        6. Update passenger lists and agent state
        7. Remove agent from waiting lists if applicable
        8. Return detailed result with capacity and route information
        
        Args:
            agent_id: Unique identifier for the agent attempting to board
            train_stop_id: Unique identifier for the train station location
            
        Returns:
            Dictionary with action result, success status, and detailed information:
            - success: Boolean indicating if boarding was successful
            - error: Error message if boarding failed
            - train_id: ID of the boarded train (if successful)
            - capacity_info: Real-time capacity details
            - wait_time: Estimated wait time if no train available
            - route_info: Train route and upcoming stations
        """
        try:
            # STEP 1: Validate train station existence
            # Check if the requested train station exists in our transit network
            if train_stop_id not in self.train_stops:
                return {
                    "success": False,
                    "error": f"Train stop {train_stop_id} does not exist",
                    "agent_id": agent_id,
                    "action": "get_onto_train"
                }
            
            # STEP 2: Initialize agent state if needed
            # Create agent record if this is their first action in the system
            if agent_id not in self.agents:
                self.agents[agent_id] = {
                    "current_location": None,
                    "mode": "walking",  # Default mode is walking
                    "vehicle_id": None,
                    "destination": None
                }
            
            # STEP 3: Find available trains at the station
            # Search for trains that are:
            # - Currently at the requested station
            # - In "at_stop" status (not departing)
            # - Have available capacity for more passengers
            available_trains = []
            for train_id, train_data in self.trains.items():
                if (train_data.get("current_stop") == train_stop_id and 
                    train_data.get("status") == "at_stop"):
                    
                    # STEP 4: Check capacity using SUMO TraCI if simulation is running
                    if self.env.sim_running:
                        try:
                            # Use real-time SUMO data for accurate capacity info
                            capacity = traci.vehicle.getPersonCapacity(train_id)
                            current_passengers = traci.vehicle.getPersonNumber(train_id)
                            if current_passengers < capacity:
                                available_trains.append({
                                    "id": train_id,
                                    "capacity": capacity,
                                    "current_passengers": current_passengers,
                                    "available_space": capacity - current_passengers
                                })
                        except traci.exceptions.TraCIException:
                            # STEP 4a: Fallback to simulated data if TraCI fails
                            capacity = train_data.get("capacity", 200)
                            current_passengers = len(train_data.get("passengers", []))
                            if current_passengers < capacity:
                                available_trains.append({
                                    "id": train_id,
                                    "capacity": capacity,
                                    "current_passengers": current_passengers,
                                    "available_space": capacity - current_passengers
                                })
                    else:
                        # STEP 4b: Use simulated data when SUMO is not running
                        capacity = train_data.get("capacity", 200)
                        current_passengers = len(train_data.get("passengers", []))
                        if current_passengers < capacity:
                            available_trains.append({
                                "id": train_id,
                                "capacity": capacity,
                                "current_passengers": current_passengers,
                                "available_space": capacity - current_passengers
                            })
            
            # STEP 5: Handle case when no trains have available capacity
            if not available_trains:
                # Check if trains exist but are at capacity vs no trains at all
                trains_at_station = []
                for train_id, train_data in self.trains.items():
                    if (train_data.get("current_stop") == train_stop_id and 
                        train_data.get("status") == "at_stop"):
                        trains_at_station.append(train_id)
                
                if trains_at_station:
                    # Trains exist but are full - agent must wait for space or next train
                    return {
                        "success": False,
                        "error": f"All trains at station {train_stop_id} are at capacity",
                        "agent_id": agent_id,
                        "action": "get_onto_train",
                        "wait_time": random.randint(60, 600),  # Trains have longer intervals
                        "trains_at_capacity": trains_at_station
                    }
                else:
                    # No trains at the station at all - agent must wait for arrival
                    return {
                        "success": False,
                        "error": f"No available train at station {train_stop_id}",
                        "agent_id": agent_id,
                        "action": "get_onto_train",
                        "wait_time": random.randint(60, 600)  # Estimated wait time in seconds
                    }
            
            # STEP 6: Select the best train for boarding
            # Choose the train with the most available space to optimize passenger distribution
            best_train = max(available_trains, key=lambda x: x["available_space"])
            train_id = best_train["id"]
            
            # STEP 7: Add passenger to train and update tracking
            if self.env.sim_running:
                try:
                    # In SUMO, passengers are managed through person simulation
                    # For now, we track manually and sync with SUMO state
                    if "passengers" not in self.trains[train_id]:
                        self.trains[train_id]["passengers"] = []
                    self.trains[train_id]["passengers"].append(agent_id)
                except traci.exceptions.TraCIException as e:
                    print(f"Warning: Could not sync passenger data with SUMO: {e}")
                    # Continue with manual tracking even if SUMO sync fails
                    if "passengers" not in self.trains[train_id]:
                        self.trains[train_id]["passengers"] = []
                    self.trains[train_id]["passengers"].append(agent_id)
            else:
                # Manual tracking when SUMO is not running
                if "passengers" not in self.trains[train_id]:
                    self.trains[train_id]["passengers"] = []
                self.trains[train_id]["passengers"].append(agent_id)
            
            # STEP 8: Update agent state to reflect boarding
            self.agents[agent_id].update({
                "current_location": train_stop_id,
                "mode": "train",  # Agent is now on a train
                "vehicle_id": train_id,  # Track which specific train
                "boarded_at": train_stop_id  # Remember where they got on
            })
            
            # STEP 9: Clean up waiting lists
            # Remove agent from train station waiting list if they were waiting
            if agent_id in self.train_stops[train_stop_id]["waiting_agents"]:
                self.train_stops[train_stop_id]["waiting_agents"].remove(agent_id)
            
            # STEP 10: Return success response with detailed information
            return {
                "success": True,
                "agent_id": agent_id,
                "action": "get_onto_train",
                "train_stop_id": train_stop_id,
                "train_id": train_id,
                "train_route": self.trains[train_id].get("route", []),
                "next_stops": self.trains[train_id].get("upcoming_stops", []),
                "capacity_info": {
                    "total_capacity": best_train["capacity"],
                    "current_passengers": best_train["current_passengers"] + 1,  # +1 for this agent
                    "remaining_space": best_train["available_space"] - 1
                },
                "travel_info": {
                    "vehicle_type": "train",
                    "boarding_time": "30-60 seconds",  # Trains typically take longer to board
                    "frequency": "5-15 minutes"  # Train frequency is usually lower than buses
                }
            }
            
        except Exception as e:
            # Handle any unexpected errors gracefully
            return {
                "success": False,
                "error": f"Error boarding train: {str(e)}",
                "agent_id": agent_id,
                "action": "get_onto_train"
            }
    
    def get_off_train(self, agent_id: str, train_stop_id: str) -> Dict[str, Any]:
        """Agent gets off train at specified stop
        
        This function simulates an agent alighting (getting off) a train at a specific station.
        It performs validation to ensure the agent is on a train and the train is at the correct station.
        Trains typically have longer boarding/alighting times and serve longer distance routes.
        
        Step-by-step process:
        1. Validate that the agent exists and is currently on a train
        2. Verify the train station exists in the network
        3. Check that the agent's current train is actually at the requested station
        4. Remove the agent from the train passenger list
        5. Update agent state to reflect they are now walking
        6. Handle potential transfers to buses or other trains
        7. Return detailed result with journey information and transfer options
        
        Args:
            agent_id: Unique identifier for the agent attempting to alight
            train_stop_id: Unique identifier for the train station where agent wants to get off
            
        Returns:
            Dictionary with action result, success status, and detailed information:
            - success: Boolean indicating if alighting was successful
            - error: Error message if alighting failed
            - previous_train_id: ID of the train the agent was on
            - new_location: Updated location after alighting
            - transfer_options: Available transit options at this station
            - journey_segment: Details about the completed journey segment
        """
        try:
            # STEP 1: Validate agent exists and is on a train
            # Check if agent exists in our system
            if agent_id not in self.agents:
                return {
                    "success": False,
                    "error": f"Agent {agent_id} does not exist",
                    "agent_id": agent_id,
                    "action": "get_off_train"
                }
            
            agent = self.agents[agent_id]
            
            # Verify agent is currently on a train (not walking or on other transport)
            if agent.get("mode") != "train":
                return {
                    "success": False,
                    "error": f"Agent {agent_id} is not currently on a train (mode: {agent.get('mode')})",
                    "agent_id": agent_id,
                    "action": "get_off_train"
                }
            
            # Verify agent has a valid train ID
            train_id = agent.get("vehicle_id")
            if not train_id:
                return {
                    "success": False,
                    "error": f"Agent {agent_id} does not have a valid train ID",
                    "agent_id": agent_id,
                    "action": "get_off_train"
                }
            
            # STEP 2: Validate train station exists
            if train_stop_id not in self.train_stops:
                return {
                    "success": False,
                    "error": f"Train station {train_stop_id} does not exist",
                    "agent_id": agent_id,
                    "action": "get_off_train"
                }
            
            # STEP 3: Verify the train is actually at the requested station
            # Check if the train exists in our system
            if train_id not in self.trains:
                return {
                    "success": False,
                    "error": f"Train {train_id} not found in system",
                    "agent_id": agent_id,
                    "action": "get_off_train"
                }
            
            train = self.trains[train_id]
            
            # Verify the train is currently at the station where agent wants to get off
            if train.get("current_stop") != train_stop_id:
                return {
                    "success": False,
                    "error": f"Train {train_id} is not at station {train_stop_id} (currently at: {train.get('current_stop')})",
                    "agent_id": agent_id,
                    "action": "get_off_train",
                    "train_current_stop": train.get("current_stop")
                }
            
            # Verify the train is in a state where passengers can alight
            if train.get("status") != "at_stop":
                return {
                    "success": False,
                    "error": f"Train {train_id} is not stopped at {train_stop_id} (status: {train.get('status')})",
                    "agent_id": agent_id,
                    "action": "get_off_train"
                }
            
            # STEP 4: Remove agent from train passenger list
            # Initialize passengers list if it doesn't exist
            if "passengers" not in train:
                train["passengers"] = []
            
            # Check if agent is actually on the passenger list
            if agent_id not in train["passengers"]:
                return {
                    "success": False,
                    "error": f"Agent {agent_id} is not listed as passenger on train {train_id}",
                    "agent_id": agent_id,
                    "action": "get_off_train"
                }
            
            # Remove agent from train passenger list
            train["passengers"].remove(agent_id)
            
            # Sync with SUMO if simulation is running
            if self.env.sim_running:
                try:
                    # In SUMO, passenger alighting is managed through person simulation
                    # For now, we track manually and would sync with SUMO state
                    current_passengers = traci.vehicle.getPersonNumber(train_id)
                    # Note: In full SUMO integration, we would use person commands here
                except traci.exceptions.TraCIException as e:
                    print(f"Warning: Could not sync passenger data with SUMO: {e}")
                    # Continue with manual tracking even if SUMO sync fails
            
            # STEP 5: Update agent state to reflect alighting
            previous_train_id = agent["vehicle_id"]
            boarded_at = agent.get("boarded_at")
            
            self.agents[agent_id].update({
                "current_location": train_stop_id,
                "mode": "walking",  # Agent is now walking
                "vehicle_id": None,  # No longer on any vehicle
                "previous_train": previous_train_id,  # Track for history/analytics
                "alighted_at": train_stop_id  # Remember where they got off
            })
            
            # STEP 6: Handle potential transfers and connections
            # Get available transfer options at this station
            transfer_options = {
                "trains": [],
                "buses": [],
                "connections": []
            }
            
            # Find other trains at this station
            for other_train_id, other_train_data in self.trains.items():
                if (other_train_id != train_id and  # Don't include the train they just got off
                    other_train_data.get("current_stop") == train_stop_id and
                    other_train_data.get("status") == "at_stop"):
                    
                    # Check capacity for potential boarding
                    capacity = other_train_data.get("capacity", 200)
                    current_passengers = len(other_train_data.get("passengers", []))
                    if current_passengers < capacity:
                        transfer_options["trains"].append({
                            "train_id": other_train_id,
                            "route": other_train_data.get("route", []),
                            "line": other_train_data.get("line", "Unknown"),
                            "available_space": capacity - current_passengers,
                            "direction": other_train_data.get("direction", "Unknown")
                        })
            
            # Find buses at connected bus stops
            station_info = self.train_stops[train_stop_id]
            if station_info.get("connects_to_bus"):
                # Logic for finding connected bus stops
                connected_bus_stops = station_info.get("connected_bus_stops", [])
                for bus_stop_id in connected_bus_stops:
                    if bus_stop_id in self.bus_stops:
                        for bus_id, bus_data in self.buses.items():
                            if (bus_data.get("current_stop") == bus_stop_id and
                                bus_data.get("status") == "at_stop"):
                                
                                capacity = bus_data.get("capacity", 30)
                                current_passengers = len(bus_data.get("passengers", []))
                                if current_passengers < capacity:
                                    transfer_options["buses"].append({
                                        "bus_id": bus_id,
                                        "bus_stop_id": bus_stop_id,
                                        "route": bus_data.get("route", []),
                                        "available_space": capacity - current_passengers,
                                        "walking_distance": "2-5 minutes"  # Typical station to bus stop distance
                                    })
            
            # Add information about station connections and facilities
            transfer_options["connections"] = {
                "walking_paths": station_info.get("walking_connections", []),
                "facilities": station_info.get("facilities", []),
                "accessibility": station_info.get("accessibility", {}),
                "platform_info": station_info.get("platforms", {})
            }
            
            # STEP 7: Return success response with detailed information
            return {
                "success": True,
                "agent_id": agent_id,
                "action": "get_off_train",
                "train_stop_id": train_stop_id,
                "previous_train_id": previous_train_id,
                "boarded_at": boarded_at,
                "new_location": train_stop_id,
                "new_mode": "walking",
                "transfer_options": transfer_options,
                "journey_segment": {
                    "from": boarded_at,
                    "to": train_stop_id,
                    "vehicle": previous_train_id,
                    "mode": "train",
                    "vehicle_type": "rail_transport"
                },
                "station_info": {
                    "station_name": station_info.get("name", train_stop_id),
                    "zone": station_info.get("zone", "Unknown"),
                    "services": station_info.get("services", []),
                    "next_departure_estimate": random.randint(300, 900)  # 5-15 minutes typical
                }
            }
            
        except Exception as e:
            # Handle any unexpected errors gracefully
            return {
                "success": False,
                "error": f"Error alighting from train: {str(e)}",
                "agent_id": agent_id,
                "action": "get_off_train"
            }
    
    def get_agent_status(self, agent_id: str) -> Dict[str, Any]:
        """Get current status of an agent
        
        Args:
            agent_id: ID of the agent
            
        Returns:
            Dictionary with agent status information
        """
        if agent_id not in self.agents:
            return {
                "exists": False,
                "agent_id": agent_id
            }
        
        agent = self.agents[agent_id]
        status = {
            "exists": True,
            "agent_id": agent_id,
            "current_location": agent.get("current_location"),
            "mode": agent.get("mode"),
            "vehicle_id": agent.get("vehicle_id"),
            "destination": agent.get("destination")
        }
        
        # Add vehicle-specific information
        if agent.get("mode") == "bus" and agent.get("vehicle_id"):
            bus_id = agent["vehicle_id"]
            if bus_id in self.buses:
                status["vehicle_info"] = {
                    "route": self.buses[bus_id].get("route", []),
                    "next_stops": self.buses[bus_id].get("upcoming_stops", []),
                    "passenger_count": len(self.buses[bus_id].get("passengers", []))
                }
        elif agent.get("mode") == "train" and agent.get("vehicle_id"):
            train_id = agent["vehicle_id"]
            if train_id in self.trains:
                status["vehicle_info"] = {
                    "route": self.trains[train_id].get("route", []),
                    "next_stops": self.trains[train_id].get("upcoming_stops", []),
                    "passenger_count": len(self.trains[train_id].get("passengers", []))
                }
        
        return status
    
    def get_available_stops(self) -> Dict[str, Any]:
        """Get information about all available stops
        
        Returns:
            Dictionary with bus and train stop information
        """
        return {
            "bus_stops": {stop_id: {
                "id": stop_data["id"],
                "position": stop_data["position"],
                "waiting_agents": len(stop_data["waiting_agents"]),
                "served_by": stop_data["served_by"]
            } for stop_id, stop_data in self.bus_stops.items()},
            "train_stops": {stop_id: {
                "id": stop_data["id"],
                "position": stop_data["position"],
                "waiting_agents": len(stop_data["waiting_agents"]),
                "served_by": stop_data["served_by"]
            } for stop_id, stop_data in self.train_stops.items()}
        }
    
    def update_simulation_state(self, network_state: Dict[str, Any]):
        """Update simulator state based on SUMO network state
        
        Args:
            network_state: Network state from SUMO environment
        """
        # Update public transit vehicles positions and schedules
        if "public_transit" in network_state:
            transit_data = network_state["public_transit"]
            
            # Update bus positions
            if "buses" in transit_data:
                for bus_id, bus_data in transit_data["buses"].items():
                    if bus_id in self.buses:
                        self.buses[bus_id].update(bus_data)
            
            # Update train positions
            if "trains" in transit_data:
                for train_id, train_data in transit_data["trains"].items():
                    if train_id in self.trains:
                        self.trains[train_id].update(train_data)


if __name__ == "__main__":
    # Example usage with the test files we've created
    network_file = "example_network.net.xml"
    route_file = "example_routes.rou.xml"
    config_file = "example.sumocfg"
    output_dir = "output"
    
    # Update this path to point to your SUMO binary
    # Use sumo-gui for visualization or sumo for headless simulation
    # If SUMO is installed properly, you can just use the command name
    sumo_binary = "sumo"  # or "sumo-gui" for visualization
    
    # Replace with your actual OpenAI API key (or leave empty)
    openai_api_key = ""  # Optional if not using LLM features
    
    # Additional files for public transit
    additional_files = ["bus_schedule.add.xml"]
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Run simulation for 3 iterations (reduced for faster testing)
    run_simulation(network_file, route_file, config_file, 
                  output_dir, sumo_binary, openai_api_key, num_iterations=3, additional_files=additional_files)
