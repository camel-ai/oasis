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

class SUMOEnvironment:
    """Environment for SUMO simulation with LLM agent feedback"""
    
    def __init__(self, network_file: str, route_file: str, 
                 config_file: str, output_dir: str,
                 additional_files: List[str] = None):
        """Initialize the SUMO environment
        
        Args:
            network_file: Path to the network XML file
            route_file: Path to the route XML file
            config_file: Path to the SUMO config file
            output_dir: Directory to store output files
            additional_files: List of additional files (e.g., public transit schedules)
        """
        self.network_file = network_file
        self.route_file = route_file
        self.config_file = config_file
        self.output_dir = output_dir
        self.additional_files = additional_files if additional_files else []
        self.iteration = 0
        self.sim_running = False
        self.vehicles = {}  # Dictionary to store vehicle data
        self.public_transit = {
            "buses": {},    # Bus data
            "trains": {},   # Train data
            "stops": {}     # Public transit stop data
        }
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
    def run_iteration(self, sumo_binary: str) -> Dict[str, Any]:
        """Run a single SUMO iteration
        
        Args:
            sumo_binary: Path to the SUMO or SUMO-GUI binary
            
        Returns:
            Dictionary containing network state information
        """
        # Create iteration-specific output directory
        iter_output_dir = os.path.join(self.output_dir, f"iteration_{self.iteration}")
        os.makedirs(iter_output_dir, exist_ok=True)
        
        # Prepare configuration for this iteration
        iter_config_file = os.path.join(iter_output_dir, f"config_iter_{self.iteration}.sumocfg")
        self._prepare_iteration_config(iter_config_file, iter_output_dir)
        
        # Run SUMO using TraCI
        try:
            # Start the simulation
            print(f"Running SUMO iteration {self.iteration}...")
            print(f"  - Config file: {iter_config_file}")
            print(f"  - Output directory: {iter_output_dir}")
            
            traci_port = self._get_free_port()
            sumo_cmd = [
                sumo_binary,
                "-c", iter_config_file,
                "--remote-port", str(traci_port),
                "--no-step-log", "true",
                "--no-warnings", "true",
                "--output-prefix", f"iter_{self.iteration}_",
                "--netstate-dump", os.path.join(iter_output_dir, "network_state.xml"),
                "--netstate-dump.empty-edges", "true",
                "--netstate-dump.max-parts", "1"
            ]
            
            # Add additional files if specified
            if self.additional_files:
                additional_param = ",".join(self.additional_files)
                sumo_cmd.extend(["--additional-files", additional_param])
            
            # Try to start SUMO
            try:
                sumo_process = subprocess.Popen(sumo_cmd)
                traci.init(traci_port)
                self.sim_running = True
            except (subprocess.SubprocessError, traci.exceptions.FatalTraCIError) as e:
                print(f"Error starting SUMO: {e}")
                print("Falling back to simulated data...")
                return self._generate_simulated_network_state()
            
            # Run the simulation for a fixed number of steps
            network_state = {}
            step_count = 0
            max_steps = 100  # Number of simulation steps to run
            
            while step_count < max_steps:
                traci.simulationStep()
                
                # Update vehicle data
                vehicle_ids = traci.vehicle.getIDList()
                for veh_id in vehicle_ids:
                    if veh_id not in self.vehicles:
                        # New vehicle - initialize its data
                        route_id = traci.vehicle.getRouteID(veh_id)
                        vehicle_type = traci.vehicle.getTypeID(veh_id)
                        self.vehicles[veh_id] = {
                            "id": veh_id,
                            "route_id": route_id,
                            "type": vehicle_type,
                            "edge_history": [],
                            "travel_times": {},
                            "is_public_transit": veh_id.startswith("bus_") or veh_id.startswith("train_")
                        }
                    
                    # Update vehicle data
                    edge_id = traci.vehicle.getRoadID(veh_id)
                    if edge_id.startswith(":"):
                        # Skip junction edges
                        continue
                        
                    speed = traci.vehicle.getSpeed(veh_id)
                    position = traci.vehicle.getPosition(veh_id)
                    lane_id = traci.vehicle.getLaneID(veh_id)
                    
                    # Update edge history
                    veh_data = self.vehicles[veh_id]
                    if edge_id not in veh_data["edge_history"]:
                        veh_data["edge_history"].append(edge_id)
                    
                    # Update travel time for this edge
                    if edge_id in veh_data["travel_times"]:
                        veh_data["travel_times"][edge_id]["time"] += 1
                    else:
                        veh_data["travel_times"][edge_id] = {
                            "entry_time": step_count,
                            "time": 1,
                            "lane_id": lane_id
                        }
                
                # Collect public transit data
                try:
                    # Track buses (assuming bus IDs start with "bus_")
                    bus_ids = [veh_id for veh_id in vehicle_ids if veh_id.startswith("bus_")]
                    for bus_id in bus_ids:
                        if bus_id not in self.public_transit["buses"]:
                            route_id = traci.vehicle.getRouteID(bus_id)
                            line_id = traci.vehicle.getLine(bus_id) if hasattr(traci.vehicle, 'getLine') else "unknown"
                            self.public_transit["buses"][bus_id] = {
                                "id": bus_id,
                                "route_id": route_id,
                                "line": line_id,
                                "stops": [],
                                "current_edge": traci.vehicle.getRoadID(bus_id),
                                "next_stop": None,
                                "schedule": {}
                            }
                        
                        # Update bus data
                        self.public_transit["buses"][bus_id]["current_edge"] = traci.vehicle.getRoadID(bus_id)
                        
                        # Get the next stop (if available in this SUMO version)
                        if hasattr(traci.vehicle, 'getNextStops'):
                            next_stops = traci.vehicle.getNextStops(bus_id)
                            if next_stops:
                                self.public_transit["buses"][bus_id]["next_stop"] = next_stops[0]
                                
                    # Track trains (assuming train IDs start with "train_")
                    train_ids = [veh_id for veh_id in vehicle_ids if veh_id.startswith("train_")]
                    for train_id in train_ids:
                        if train_id not in self.public_transit["trains"]:
                            route_id = traci.vehicle.getRouteID(train_id)
                            line_id = traci.vehicle.getLine(train_id) if hasattr(traci.vehicle, 'getLine') else "unknown"
                            self.public_transit["trains"][train_id] = {
                                "id": train_id,
                                "route_id": route_id,
                                "line": line_id,
                                "stops": [],
                                "current_edge": traci.vehicle.getRoadID(train_id),
                                "next_stop": None,
                                "schedule": {}
                            }
                        
                        # Update train data
                        self.public_transit["trains"][train_id]["current_edge"] = traci.vehicle.getRoadID(train_id)
                        
                        # Get the next stop (if available in this SUMO version)
                        if hasattr(traci.vehicle, 'getNextStops'):
                            next_stops = traci.vehicle.getNextStops(train_id)
                            if next_stops:
                                self.public_transit["trains"][train_id]["next_stop"] = next_stops[0]
                                
                    # Track public transit stops 
                    if hasattr(traci, 'busstop'):
                        stop_ids = traci.busstop.getIDList()
                        for stop_id in stop_ids:
                            if stop_id not in self.public_transit["stops"]:
                                # Initialize stop data
                                lane_id = traci.busstop.getLaneID(stop_id)
                                position = traci.busstop.getPosition(stop_id)
                                self.public_transit["stops"][stop_id] = {
                                    "id": stop_id,
                                    "lane_id": lane_id,
                                    "position": position,
                                    "waiting_vehicles": [],
                                    "edge_id": lane_id.split("_")[0] if "_" in lane_id else "unknown"
                                }
                            
                            # Update stop data
                            if hasattr(traci.busstop, 'getVehicleIDs'):
                                waiting_vehicles = traci.busstop.getVehicleIDs(stop_id)
                                self.public_transit["stops"][stop_id]["waiting_vehicles"] = waiting_vehicles
                                
                except Exception as e:
                    print(f"Error collecting public transit data: {e}")
                    # Continue with simulation even if we can't collect transit data
                
                # Collect edge data for network state
                for edge_id in traci.edge.getIDList():
                    if edge_id.startswith(":"):
                        # Skip junction edges
                        continue
                        
                    mean_speed = traci.edge.getLastStepMeanSpeed(edge_id)
                    occupancy = traci.edge.getLastStepOccupancy(edge_id)
                    vehicle_count = traci.edge.getLastStepVehicleNumber(edge_id)
                    travel_time = traci.edge.getTraveltime(edge_id)
                    
                    if edge_id not in network_state:
                        network_state[edge_id] = {
                            "mean_speed": [],
                            "occupancy": [],
                            "vehicle_count": [],
                            "travel_time": []
                        }
                    
                    network_state[edge_id]["mean_speed"].append(mean_speed)
                    network_state[edge_id]["occupancy"].append(occupancy)
                    network_state[edge_id]["vehicle_count"].append(vehicle_count)
                    network_state[edge_id]["travel_time"].append(travel_time)
                
                step_count += 1
            
            # Calculate averages for the network state
            for edge_id, data in network_state.items():
                for key, values in data.items():
                    if values:  # Ensure the list is not empty
                        data[key] = sum(values) / len(values)
                    else:
                        data[key] = 0
            
            # Add public transit information to network state
            network_state["public_transit"] = {
                "buses": self.public_transit["buses"],
                "trains": self.public_transit["trains"],
                "stops": self.public_transit["stops"]
            }
            
            # Close the TraCI connection
            traci.close()
            self.sim_running = False
            
            self.iteration += 1
            return network_state
            
        except Exception as e:
            print(f"Error during SUMO simulation: {e}")
            if self.sim_running:
                try:
                    traci.close()
                except:
                    pass
                self.sim_running = False
            
            print("Falling back to simulated data...")
            return self._generate_simulated_network_state()

    def _prepare_iteration_config(self, config_file: str, output_dir: str):
        """Prepare a SUMO configuration file for the current iteration
        
        Args:
            config_file: Path to write the new config file
            output_dir: Output directory for this iteration
        """
        # Create a basic SUMO configuration file
        # In a real implementation, you'd want to modify an existing config
        root = ET.Element("configuration")
        
        input_section = ET.SubElement(root, "input")
        ET.SubElement(input_section, "net-file", value=self.network_file)
        ET.SubElement(input_section, "route-files", value=self.route_file)
        
        if self.additional_files:
            ET.SubElement(input_section, "additional-files", value=",".join(self.additional_files))
        
        output_section = ET.SubElement(root, "output")
        ET.SubElement(output_section, "tripinfo-output", 
                     value=os.path.join(output_dir, "tripinfo.xml"))
        ET.SubElement(output_section, "summary-output", 
                     value=os.path.join(output_dir, "summary.xml"))
        
        time_section = ET.SubElement(root, "time")
        ET.SubElement(time_section, "begin", value="0")
        ET.SubElement(time_section, "end", value="3600")  # 1 hour simulation
        
        # Write the configuration to file
        tree = ET.ElementTree(root)
        tree.write(config_file, encoding="utf-8", xml_declaration=True)

    def _get_free_port(self) -> int:
        """Get a free port for TraCI connection"""
        import socket
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.bind(('', 0))
        port = s.getsockname()[1]
        s.close()
        return port
        
    def _generate_simulated_network_state(self) -> Dict[str, Any]:
        """Generate simulated network state when SUMO is unavailable
        
        Returns:
            Dictionary with network state data
        """
        print("Generating simulated network data...")
        
        # Try to parse the network file to get edge IDs
        edge_ids = []
        try:
            tree = ET.parse(self.network_file)
            root = tree.getroot()
            for edge in root.findall(".//edge"):
                if "id" in edge.attrib and not edge.attrib["id"].startswith(":"):
                    edge_ids.append(edge.attrib["id"])
        except:
            # If network file can't be parsed, generate some dummy edge IDs
            edge_ids = [f"edge_{i}" for i in range(10)]
        
        # Generate simulated data for each edge
        network_state = {}
        for edge_id in edge_ids:
            # Generate realistic traffic data
            # Lower speeds and higher occupancy for some edges to simulate congestion
            is_congested = random.random() < 0.3
            
            if is_congested:
                mean_speed = random.uniform(5.0, 15.0)  # m/s
                occupancy = random.uniform(0.4, 0.8)
                vehicle_count = random.randint(5, 20)
                travel_time = random.uniform(60.0, 180.0)  # seconds
            else:
                mean_speed = random.uniform(20.0, 33.0)  # m/s
                occupancy = random.uniform(0.05, 0.3)
                vehicle_count = random.randint(0, 7)
                travel_time = random.uniform(20.0, 45.0)  # seconds
            
            network_state[edge_id] = {
                "mean_speed": mean_speed,
                "occupancy": occupancy,
                "vehicle_count": vehicle_count,
                "travel_time": travel_time
            }
        
        # Add simulated public transit data
        network_state["public_transit"] = self._generate_simulated_public_transit(edge_ids)
        
        return network_state
        
    def _generate_simulated_public_transit(self, edge_ids: List[str]) -> Dict[str, Any]:
        """Generate simulated public transit data
        
        Args:
            edge_ids: List of edge IDs in the network
            
        Returns:
            Dictionary with simulated public transit data
        """
        # Generate simulated bus data
        buses = {}
        for i in range(5):  # Simulate 5 buses
            bus_id = f"bus_{i}"
            route = random.sample(edge_ids, min(5, len(edge_ids)))
            current_edge = route[random.randint(0, len(route)-1)]
            
            buses[bus_id] = {
                "id": bus_id,
                "route_id": f"bus_route_{i}",
                "line": f"line_{i}",
                "stops": [f"stop_{j}" for j in range(3)],
                "current_edge": current_edge,
                "next_stop": f"stop_{random.randint(0, 2)}",
                "schedule": {
                    f"stop_{j}": random.randint(0, 600) for j in range(3)
                }
            }
        
        # Generate simulated train data
        trains = {}
        for i in range(3):  # Simulate 3 trains
            train_id = f"train_{i}"
            route = random.sample(edge_ids, min(4, len(edge_ids)))
            current_edge = route[random.randint(0, len(route)-1)]
            
            trains[train_id] = {
                "id": train_id,
                "route_id": f"train_route_{i}",
                "line": f"line_{i+10}",
                "stops": [f"station_{j}" for j in range(2)],
                "current_edge": current_edge,
                "next_stop": f"station_{random.randint(0, 1)}",
                "schedule": {
                    f"station_{j}": random.randint(0, 900) for j in range(2)
                }
            }
        
        # Generate simulated stop data
        stops = {}
        for i in range(8):  # Simulate 8 stops (5 bus stops, 3 train stations)
            stop_type = "station" if i >= 5 else "stop"
            stop_id = f"{stop_type}_{i % 5 if stop_type == 'stop' else i % 3}"
            
            # Select a random edge for this stop
            edge_id = random.choice(edge_ids)
            lane_id = f"{edge_id}_0"
            
            stops[stop_id] = {
                "id": stop_id,
                "lane_id": lane_id,
                "position": (random.uniform(0, 1000), random.uniform(0, 1000)),
                "waiting_vehicles": [],
                "edge_id": edge_id
            }
            
            # Randomly add waiting vehicles
            if random.random() < 0.4:
                vehicle_type = "train" if stop_type == "station" else "bus"
                vehicle_id = f"{vehicle_type}_{random.randint(0, 2 if vehicle_type == 'train' else 4)}"
                stops[stop_id]["waiting_vehicles"].append(vehicle_id)
        
        return {
            "buses": buses,
            "trains": trains,
            "stops": stops
        }


class LLMAgent:
    """Agent that uses LLM to make decisions about route updates, including multimodal options"""
    
    def __init__(self, vehicle_id: str, environment: SUMOEnvironment, openai_api_key: str):
        """Initialize agent
        
        Args:
            vehicle_id: ID of the vehicle in the SUMO simulation
            environment: SUMOEnvironment instance
            openai_api_key: API key for OpenAI
        """
        self.vehicle_id = vehicle_id
        self.environment = environment
        self.openai_api_key = openai_api_key
        self.route_history = []
        self.multimodal_journey = False  # Whether this agent is using multimodal transport
        self.transport_modes = []  # List of transport modes used
        
    def update_route(self, network_state: Dict[str, Any]) -> Dict[str, Any]:
        """Update the agent's route based on network state using LLM
        
        Args:
            network_state: Dictionary with network state information
            
        Returns:
            Dictionary with route update information
        """
        print(f"Updating route for vehicle {self.vehicle_id}")
        
        # Check if vehicle exists in environment
        if self.vehicle_id not in self.environment.vehicles:
            print(f"  Vehicle {self.vehicle_id} not found in simulation")
            return {}
        
        # Extract vehicle data
        vehicle_data = self._extract_vehicle_data()
        
        # Extract relevant network data
        network_data = self._extract_relevant_network_data(network_state)
        
        # Extract public transit data
        transit_data = self._extract_public_transit_data(network_state)
        
        # Generate prompt for LLM
        prompt = self._generate_prompt(vehicle_data, network_data, transit_data)
        
        # Query LLM
        try:
            llm_response = self._query_llm(prompt)
            
            # Parse and apply LLM suggestions
            route_update = self._parse_llm_suggestions(llm_response)
            return route_update
            
        except Exception as e:
            print(f"Error querying LLM: {e}")
            return {}
    
    def _extract_vehicle_data(self) -> Dict[str, Any]:
        """Extract relevant data from a vehicle
        
        Returns:
            Dictionary with vehicle data
        """
        vehicle_data = self.environment.vehicles.get(self.vehicle_id, {})
        
        # Format the data for the LLM prompt
        formatted_data = {
            "vehicle_id": self.vehicle_id,
            "vehicle_type": vehicle_data.get("type", "unknown"),
            "current_route": vehicle_data.get("route_id", "unknown"),
            "edge_history": vehicle_data.get("edge_history", []),
            "travel_times": vehicle_data.get("travel_times", {}),
            "is_public_transit": vehicle_data.get("is_public_transit", False),
            "multimodal_journey": self.multimodal_journey,
            "transport_modes": self.transport_modes
        }
        
        return formatted_data
    
    def _extract_relevant_network_data(self, network_state: Dict[str, Any]) -> Dict[str, Any]:
        """Extract network data relevant to this agent's routes
        
        Args:
            network_state: Full network state
            
        Returns:
            Dictionary with relevant network data
        """
        relevant_data = {}
        
        # Skip public transit key if present
        if "public_transit" in network_state:
            network_state_edges = {k: v for k, v in network_state.items() if k != "public_transit"}
        else:
            network_state_edges = network_state
        
        # Get vehicle's edge history
        vehicle_data = self.environment.vehicles.get(self.vehicle_id, {})
        edge_history = vehicle_data.get("edge_history", [])
        
        # Include data for edges the vehicle has traveled on
        for edge_id in edge_history:
            if edge_id in network_state_edges:
                relevant_data[edge_id] = network_state_edges[edge_id]
        
        # Include data for all edges if we have limited history
        if len(relevant_data) < 5:
            # Add some of the most congested edges
            congested_edges = sorted(
                [(edge_id, data["travel_time"]) for edge_id, data in network_state_edges.items()],
                key=lambda x: x[1],
                reverse=True
            )[:10]
            
            for edge_id, _ in congested_edges:
                if edge_id not in relevant_data:
                    relevant_data[edge_id] = network_state_edges[edge_id]
        
        return relevant_data
    
    def _extract_public_transit_data(self, network_state: Dict[str, Any]) -> Dict[str, Any]:
        """Extract public transit data from network state
        
        Args:
            network_state: Full network state
            
        Returns:
            Dictionary with public transit data
        """
        transit_data = {
            "nearby_stops": [],
            "relevant_buses": [],
            "relevant_trains": []
        }
        
        # Skip if public transit data is not available
        if "public_transit" not in network_state:
            return transit_data
        
        pt_data = network_state["public_transit"]
        
        # Get vehicle's current edge
        vehicle_data = self.environment.vehicles.get(self.vehicle_id, {})
        current_edge = vehicle_data.get("edge_history", [])[-1] if vehicle_data.get("edge_history") else None
        
        if not current_edge:
            return transit_data
        
        # Find nearby stops (stops on the current edge or recent history)
        edge_history = vehicle_data.get("edge_history", [])[-5:]  # Last 5 edges
        
        for stop_id, stop_data in pt_data.get("stops", {}).items():
            stop_edge = stop_data.get("edge_id")
            if stop_edge in edge_history:
                transit_data["nearby_stops"].append({
                    "id": stop_id,
                    "edge_id": stop_edge,
                    "position": stop_data.get("position"),
                    "waiting_vehicles": stop_data.get("waiting_vehicles", [])
                })
        
        # Find relevant buses (buses that pass through nearby stops)
        for bus_id, bus_data in pt_data.get("buses", {}).items():
            next_stop = bus_data.get("next_stop")
            for stop in transit_data["nearby_stops"]:
                if next_stop == stop["id"] or stop["id"] in bus_data.get("stops", []):
                    transit_data["relevant_buses"].append({
                        "id": bus_id,
                        "line": bus_data.get("line", "unknown"),
                        "current_edge": bus_data.get("current_edge"),
                        "next_stop": next_stop
                    })
                    break
        
        # Find relevant trains (trains that pass through nearby stops)
        for train_id, train_data in pt_data.get("trains", {}).items():
            next_stop = train_data.get("next_stop")
            for stop in transit_data["nearby_stops"]:
                if next_stop == stop["id"] or stop["id"] in train_data.get("stops", []):
                    transit_data["relevant_trains"].append({
                        "id": train_id,
                        "line": train_data.get("line", "unknown"),
                        "current_edge": train_data.get("current_edge"),
                        "next_stop": next_stop
                    })
                    break
        
        return transit_data
    
    def _generate_prompt(self, vehicle_data: Dict[str, Any], 
                        network_data: Dict[str, Any], 
                        transit_data: Dict[str, Any]) -> str:
        """Generate a prompt for the LLM
        
        Args:
            vehicle_data: Vehicle's current data
            network_data: Relevant network state
            transit_data: Public transit data
            
        Returns:
            Prompt string
        """
        prompt = f"""
You are an intelligent navigation system for vehicle {vehicle_data['vehicle_id']} of type {vehicle_data['vehicle_type']}.
Your task is to provide optimal routing recommendations, including multimodal options with buses and trains when appropriate.

CURRENT VEHICLE STATUS:
- Vehicle ID: {vehicle_data['vehicle_id']}
- Vehicle Type: {vehicle_data['vehicle_type']}
- Current Route: {vehicle_data['current_route']}
- Edge History: {', '.join(vehicle_data['edge_history'][-5:]) if vehicle_data['edge_history'] else 'No history yet'}
- Currently Using Multimodal Journey: {vehicle_data['multimodal_journey']}
- Transport Modes Used: {', '.join(vehicle_data['transport_modes']) if vehicle_data['transport_modes'] else 'Private vehicle only'}

TRAFFIC NETWORK STATUS:
"""
        
        # Add information about network edges
        for edge_id, edge_data in network_data.items():
            prompt += f"""
Edge ID: {edge_id}
- Mean Speed: {edge_data['mean_speed']:.2f} m/s
- Occupancy: {edge_data['occupancy']:.2f}
- Vehicle Count: {edge_data['vehicle_count']:.1f}
- Travel Time: {edge_data['travel_time']:.2f} seconds
"""
        
        # Add travel time information if available
        if vehicle_data['travel_times']:
            prompt += "\nVEHICLE TRAVEL TIMES:\n"
            for edge_id, time_data in vehicle_data['travel_times'].items():
                prompt += f"- Edge {edge_id}: {time_data['time']} seconds\n"
        
        # Add public transit information
        prompt += "\nPUBLIC TRANSIT OPTIONS:\n"
        
        if transit_data["nearby_stops"]:
            prompt += "Nearby Transit Stops:\n"
            for stop in transit_data["nearby_stops"]:
                prompt += f"- Stop {stop['id']} on edge {stop['edge_id']}\n"
        else:
            prompt += "No nearby transit stops detected.\n"
        
        if transit_data["relevant_buses"]:
            prompt += "\nAvailable Buses:\n"
            for bus in transit_data["relevant_buses"]:
                prompt += f"- Bus {bus['id']} (Line {bus['line']}), currently on edge {bus['current_edge']}, next stop: {bus['next_stop']}\n"
        
        if transit_data["relevant_trains"]:
            prompt += "\nAvailable Trains:\n"
            for train in transit_data["relevant_trains"]:
                prompt += f"- Train {train['id']} (Line {train['line']}), currently on edge {train['current_edge']}, next stop: {train['next_stop']}\n"
        
        # Request routing recommendations including multimodal options
        prompt += """
Based on the traffic network status and public transit options, please provide routing recommendations for this vehicle.
Analyze the traffic conditions and suggest improvements to the vehicle's journey, including multimodal options with buses and trains when appropriate.

INSTRUCTIONS:
1. Identify congested edges (high travel time, low speed, high occupancy)
2. Consider both private vehicle routing and public transit options
3. Recommend switching to public transit if it would be more efficient
4. Consider the vehicle's type and history in your recommendations
5. Provide a brief explanation for your recommendations

Output your response in JSON format as follows:
```json
{
  "route_adjustments": [
    {
      "edge_to_avoid": "edge_id",
      "alternative_edge": "edge_id",
      "confidence": 0.8
    }
  ],
  "public_transit_recommendations": [
    {
      "type": "bus",
      "id": "bus_id",
      "board_at_stop": "stop_id",
      "alight_at_stop": "stop_id",
      "confidence": 0.9
    }
  ],
  "multimodal_strategy": {
    "use_multimodal": true,
    "transport_modes": ["car", "bus", "train"],
    "transitions": [
      {
        "from_mode": "car",
        "to_mode": "bus",
        "at_location": "stop_id",
        "time_saved_estimate": 300
      }
    ]
  },
  "explanation": "Brief explanation of your recommendations"
}
```
"""
        return prompt
    
    def _query_llm(self, prompt: str) -> str:
        """Query LLM with the prompt
        
        Args:
            prompt: Prompt string
            
        Returns:
            Response from LLM
        """
        if not self.openai_api_key:
            # Generate a simulated response if no API key
            return self._generate_simulated_llm_response()
        
        try:
            # Send request to OpenAI API
            response = requests.post(
                "https://api.openai.com/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {self.openai_api_key}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": "gpt-4",
                    "messages": [
                        {"role": "system", "content": "You are a traffic navigation assistant for SUMO traffic simulation."},
                        {"role": "user", "content": prompt}
                    ],
                    "temperature": 0.7,
                    "max_tokens": 500
                },
                timeout=30
            )
            
            # Parse the response
            response_data = response.json()
            if "choices" in response_data and len(response_data["choices"]) > 0:
                return response_data["choices"][0]["message"]["content"]
            else:
                print("Unexpected response format from OpenAI API")
                return self._generate_simulated_llm_response()
                
        except Exception as e:
            print(f"Error calling OpenAI API: {e}")
            return self._generate_simulated_llm_response()
    
    def _generate_simulated_llm_response(self) -> str:
        """Generate a simulated LLM response when API is unavailable
        
        Returns:
            Simulated response string
        """
        print("  Generating simulated LLM response")
        
        # Create a reasonable simulated response that includes multimodal options
        edge_to_avoid = f"edge_{random.randint(1, 9)}"
        alternative_edge = f"edge_{random.randint(1, 9)}"
        confidence = round(random.uniform(0.6, 0.95), 2)
        
        # Randomly decide whether to recommend public transit
        recommend_transit = random.random() < 0.4
        
        response = {
            "route_adjustments": [
                {
                    "edge_to_avoid": edge_to_avoid,
                    "alternative_edge": alternative_edge,
                    "confidence": confidence
                }
            ],
            "explanation": "Based on traffic conditions, I've identified congestion on the specified edge."
        }
        
        if recommend_transit:
            # Add public transit recommendations
            transit_type = "bus" if random.random() < 0.7 else "train"
            transit_id = f"{transit_type}_{random.randint(0, 2)}"
            board_stop = f"stop_{random.randint(0, 4)}" if transit_type == "bus" else f"station_{random.randint(0, 2)}"
            alight_stop = f"stop_{random.randint(0, 4)}" if transit_type == "bus" else f"station_{random.randint(0, 2)}"
            
            response["multimodal_strategy"] = {
                "use_multimodal": True,
                "transport_modes": ["car", transit_type],
                "transitions": [
                    {
                        "from_mode": "car",
                        "to_mode": transit_type,
                        "at_location": board_stop,
                        "time_saved_estimate": random.randint(100, 500)
                    }
                ]
            }
            
            response["explanation"] += f" I recommend switching to {transit_type} at {board_stop} to save time and avoid traffic congestion."
        else:
            response["public_transit_recommendations"] = []
            response["multimodal_strategy"] = {
                "use_multimodal": False,
                "transport_modes": ["car"],
                "transitions": []
            }
            response["explanation"] += " The private vehicle route is currently optimal, so no public transit recommendations at this time."
        
        return f"```json\n{json.dumps(response, indent=2)}\n```"
    
    def _parse_llm_suggestions(self, llm_response: str) -> Dict[str, Any]:
        """Parse LLM response and extract route adjustment suggestions
        
        Args:
            llm_response: String response from LLM
            
        Returns:
            Dictionary with route adjustments and multimodal strategy
        """
        try:
            # Extract JSON from markdown code blocks if present
            json_start = llm_response.find("```json")
            if json_start != -1:
                json_start = llm_response.find("\n", json_start) + 1
                json_end = llm_response.find("```", json_start)
                json_str = llm_response[json_start:json_end].strip()
            else:
                # Try to find any JSON object in the response
                json_start = llm_response.find("{")
                json_end = llm_response.rfind("}") + 1
                if json_start == -1 or json_end == 0:
                    print("  No JSON found in LLM response")
                    return {}
                json_str = llm_response[json_start:json_end].strip()
            
            # Parse the JSON
            recommendations = json.loads(json_str)
            
            print(f"  Received routing recommendations:")
            print(f"  Explanation: {recommendations.get('explanation', 'No explanation provided')}")
            
            # Process route adjustments
            for adjustment in recommendations.get("route_adjustments", []):
                edge_to_avoid = adjustment.get("edge_to_avoid")
                alternative_edge = adjustment.get("alternative_edge")
                confidence = adjustment.get("confidence", 0.0)
                
                print(f"  - Avoid {edge_to_avoid}, use {alternative_edge} instead (confidence: {confidence:.2f})")
            
            # Process public transit recommendations
            for transit_rec in recommendations.get("public_transit_recommendations", []):
                transit_type = transit_rec.get("type")
                transit_id = transit_rec.get("id")
                board_stop = transit_rec.get("board_at_stop")
                alight_stop = transit_rec.get("alight_at_stop")
                confidence = transit_rec.get("confidence", 0.0)
                
                print(f"  - Take {transit_type} {transit_id} from {board_stop} to {alight_stop} (confidence: {confidence:.2f})")
            
            # Process multimodal strategy
            multimodal = recommendations.get("multimodal_strategy", {})
            if multimodal.get("use_multimodal", False):
                self.multimodal_journey = True
                self.transport_modes = multimodal.get("transport_modes", ["car"])
                
                print(f"  - Multimodal journey recommended with modes: {', '.join(self.transport_modes)}")
                
                for transition in multimodal.get("transitions", []):
                    from_mode = transition.get("from_mode")
                    to_mode = transition.get("to_mode")
                    location = transition.get("at_location")
                    time_saved = transition.get("time_saved_estimate", 0)
                    
                    print(f"  - Transit from {from_mode} to {to_mode} at {location} (saves ~{time_saved} seconds)")
            
            return recommendations
            
        except Exception as e:
            print(f"Error parsing LLM suggestions: {e}")
            return {}


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
                agents.append(LLMAgent(vehicle_id, env, openai_api_key))
            
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
