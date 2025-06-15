# TODO reconfirm the simulation process with stone
import os
import subprocess
import traci
from typing import List, Dict, Any

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
        # dictionary to store public transit data
        self.public_transit = {
            "buses": {},    # Bus data
            "trains": {},   # Train data
            "stops": {}     # Public transit stop data
        }
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
    async def run_iteration(self, sumo_binary: str) -> Dict[str, Any]:
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
                # replace it with customized data
                return self._generate_simulated_network_state()
            
            # Run the simulation for a fixed number of steps
            network_state = {}
            step_count = 0
            max_steps = 100  # Number of simulation steps to run
            
            while step_count < max_steps:
                traci.simulationStep()
                
                # Update vehicle data
                vehicle_ids: List[str] = traci.vehicle.getIDList()
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
                    edge_id: str = traci.vehicle.getRoadID(veh_id)
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
                    bus_ids: List[str] = [veh_id for veh_id in vehicle_ids if veh_id.startswith("bus_")]
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
                                position = traci.busstop.getStartPos(stop_id)
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
        print("Config file: " + config_file)
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
                "schedule": {}
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
                "schedule": {}
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
